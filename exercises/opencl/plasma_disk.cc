/* plasma_disk.cpp
 */
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "./opencl_error.h"

int NTICKS = 1000;  // Number of ticks.
int NPARTICLES = 256;  // Number of particles.

#ifdef GPU
int GROUP_SIZE = 256;
#else
int GROUP_SIZE = 1;
#endif

unsigned int _seed = 1234;
unsigned int *seed = &_seed;

class NBodySimulation {
 private:
    cl::Context      *const context;
    cl::CommandQueue *queue;
    cl::Kernel       *kernel_eom;
    cl::Program      *program_eom;

    cl::Buffer *gposold;  // compute device: electron positions
    cl::Buffer *gvelold;  // compute device: electron velocities
    cl::Buffer *gposnew;  // compute device: store temp electron positions
    cl::Buffer *gvelnew;  // compute device: store temp electron velocities

    float *hposold;
    float *hvelold;
    float *hposnew;
    float *hvelnew;

 public:
    explicit NBodySimulation(cl::Context *const context);
    ~NBodySimulation();
    void init();
    void step(float DeltaTime);
    void write(int NumBodies, float *hposnew, float *hvelnew);
    void read(int NumBodies, float *hposnew, float *hvelnew);
    void toFile(const char* filename);
    void run(const int ticks);
};

// Constructor.
NBodySimulation::NBodySimulation(cl::Context *const context)
        : context(context) {
    hposold = new float[4 * NPARTICLES];
    hvelold = new float[4 * NPARTICLES];
    hposnew = new float[4 * NPARTICLES];
    hvelnew = new float[4 * NPARTICLES];

    // Allocate buffers in device memory.
    gposold = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NPARTICLES * sizeof(float));
    gvelold = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NPARTICLES * sizeof(float));
    gposnew = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NPARTICLES * sizeof(float));
    gvelnew = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NPARTICLES * sizeof(float));
}


// Destructor.
NBodySimulation::~NBodySimulation() {
    delete gposold;
    delete gvelold;
    delete gposnew;
    delete gvelnew;
}


// Initialise data.
void NBodySimulation::init() {
    // Initialise.
    for (int ip = 0; ip < NPARTICLES; ip++) {
        // generate two random variables
        float rand_zerotoone = static_cast<float>(rand_r(seed)) /
                static_cast<float>(RAND_MAX);
        float rand_zeroto2pi = static_cast<float>(rand_r(seed)) /
                static_cast<float>(RAND_MAX) * M_PI * 2.0;

        hposold[ip * 4] = sqrt(2.0 * NPARTICLES * rand_zerotoone)
                * cos(rand_zeroto2pi);
        hposold[ip * 4 + 1] = sqrt(2.0 * NPARTICLES * rand_zerotoone)
                * sin(rand_zeroto2pi);
        hposold[ip * 4 + 2] = 0.0f;
        hposold[ip * 4 + 3] = 1.0f;
    }

    // Initialise OpenCL.
    try {
        // Get a list of devices on this platform
        std::vector<cl::Device> devices =
                context->getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        queue = new cl::CommandQueue(*context, devices[0]);

        // Read source file
        std::ifstream sourceFile_eom("integrate_eom_kernel.cl");
        std::string sourceCode_eom(
            std::istreambuf_iterator<char>(sourceFile_eom),
            (std::istreambuf_iterator<char>()));
        cl::Program::Sources source_eom
                (1, std::make_pair(sourceCode_eom.c_str(),
                                   sourceCode_eom.length() + 1));
        // Make program of the source code in the context
        program_eom = new cl::Program(*context, source_eom);
        // Build program for these specific devices, compiler argument
        // to include local directory for header files (.h).
        program_eom->build(devices, "-I.");
        // Make kernel
        kernel_eom = new cl::Kernel(*program_eom, "integrate_eom");

        // Create memory buffers on GPU and populate them with the initial data
        queue->enqueueWriteBuffer(*gposold, CL_TRUE, 0,
                                  4 * NPARTICLES * sizeof(float), hposold);
        queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0,
                                  4 * NPARTICLES * sizeof(float), hvelold);
    } catch(cl::Error error) {
        std::cerr << error.what()
                  << "(" << oclErrorString(error.err()) << ")"
                  << std::endl;
        exit(1);
    }
}

// Perform a single step of simulation.
void NBodySimulation::step(float DeltaTime) {
    try {
        // Set arguments to kernel
        kernel_eom->setArg(0, *gposold);
        kernel_eom->setArg(1, *gvelold);
        kernel_eom->setArg(2, *gposnew);
        kernel_eom->setArg(3, *gvelnew);
        kernel_eom->setArg(4, cl::Local(GROUP_SIZE * 4 * sizeof(float)));
        kernel_eom->setArg(5, NPARTICLES);
        kernel_eom->setArg(6, DeltaTime);

        // Run the kernel on specific ND range
        cl::NDRange global(NPARTICLES);
        cl::NDRange local(GROUP_SIZE);

        cl::Event eventA;
        queue->enqueueNDRangeKernel(*kernel_eom, cl::NullRange, global,
                                    local, 0, &eventA);
        eventA.wait();

        // set pointers to particles to kernel in reverse order to copy new->old

        kernel_eom->setArg(0, *gposnew);
        kernel_eom->setArg(1, *gvelnew);
        kernel_eom->setArg(2, *gposold);
        kernel_eom->setArg(3, *gvelold);

        cl::Event eventB;
        queue->enqueueNDRangeKernel(*kernel_eom, cl::NullRange, global,
                                    local, 0, &eventB);
        eventB.wait();
    } catch(cl::Error error) {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        exit(1);
    }
}

// Dump state to file.
void NBodySimulation::toFile(const char* filename) {
    queue->enqueueReadBuffer(*gposold, CL_TRUE, 0,
                             4 * NPARTICLES * sizeof(float), hposold);
    queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0,
                             4 * NPARTICLES * sizeof(float), hvelold);

    FILE *fd = fopen(filename, "w");
    for (int i = 0; i < NPARTICLES; i++) {
        fprintf(fd, "%e %e %e %e",
                hposold[i * 4],
                hposold[i * 4 + 1],
                hposold[i * 4 + 2],
                hposold[i * 4 + 3]);
        fprintf(fd, " %e %e %e %e\n",
                hvelold[i * 4],
                hvelold[i * 4 + 1],
                hvelold[i * 4 + 2],
                hvelold[i * 4 + 3]);
    }
    fclose(fd);
}

// Read state from device memory to CPU.
void NBodySimulation::read(int NumBodies,
                           float *pos,
                           float *vel) {
    // transfer memory from GPU to CPU
    queue->enqueueReadBuffer(*gposold, CL_TRUE, 0,
                             4 * NumBodies * sizeof(float), pos);
    queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0,
                             4 * NumBodies * sizeof(float), vel);
    queue->finish();
}

// Write state from CPU to device memory.
void NBodySimulation::write(int NumBodies,
                            float *pos,
                            float *vel) {
    queue->enqueueWriteBuffer(*gposold, CL_TRUE, 0,
                              4 * NumBodies * sizeof(float), pos);
    queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0,
                              4 * NumBodies * sizeof(float), vel);
    queue->finish();
}

// Run simulation.
void NBodySimulation::run(const int ticks) {
    float deltaTime = 1.0e-3f;

    time_t c0, c1;

    init();
    write(NPARTICLES, hposold, hvelold);

    time(&c0);
    for (int it = 0; it < ticks; it++)  // Main propagation loop.
        step(deltaTime);

    time(&c1);
#ifdef GPU
    fprintf(stderr,
            "GPU propagation for %d particles "
            "over %d doublesteps took %.0f s\n",
            NPARTICLES, ticks, difftime(c1, c0));
#else
    fprintf(stderr,
            "CPU propagation for %d particles "
            "over %d doublesteps took %.0f s\n",
            NPARTICLES, ticks, difftime(c1, c0));
#endif
}

int main(int argc, char *argv[]) {
    cl::Context *context;

    if (argc == 4) {
        NTICKS = atoi(argv[1]);
        GROUP_SIZE = atoi(argv[2]);
        NPARTICLES = GROUP_SIZE * atoi(argv[3]);
    }

    // Get available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    // Select the default platform and create a context using this
    // platform and the GPU.
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };
#ifdef GPU
    context = new cl::Context(CL_DEVICE_TYPE_GPU, cps);
#else
    context = new cl::Context(CL_DEVICE_TYPE_CPU, cps);
#endif

    NBodySimulation simulation(context);

    simulation.run(NTICKS);

    return 0;
}
