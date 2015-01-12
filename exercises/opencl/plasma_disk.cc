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

int NUMPART = 256;

#ifdef GPU
int GROUP_SIZE = 256;
#else
int GROUP_SIZE = 1;
#endif

unsigned int _seed = 1234;
unsigned int *seed = &_seed;

class PropagateNBodySystem {
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
    explicit PropagateNBodySystem(cl::Context *const context);
    void init();
    void PropagateDoubleStep(float DeltaTime);
    void PutElectrons(int NumBodies, float *hposnew, float *hvelnew);
    void GetElectrons(int NumBodies, float *hposnew, float *hvelnew);
    void WriteState(const char* filename);
    void RunSimulation(void);
};

PropagateNBodySystem::PropagateNBodySystem(cl::Context *const context)
        : context(context) {
    hposold = new float[4 * NUMPART];
    hvelold = new float[4 * NUMPART];
    hposnew = new float[4 * NUMPART];
    hvelnew = new float[4 * NUMPART];

    // Create memory buffers on GPU and populate them with the initial
    // data.
    gposold = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NUMPART * sizeof(float));
    gvelold = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NUMPART * sizeof(float));
    gposnew = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NUMPART * sizeof(float));
    gvelnew = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                             4 * NUMPART * sizeof(float));
}


void PropagateNBodySystem::init() {
    // Initialise.
    for (int ip = 0; ip < NUMPART; ip++) {
        // generate two random variables
        float rand_zerotoone = static_cast<float>(rand_r(seed)) /
                static_cast<float>(RAND_MAX);
        float rand_zeroto2pi = static_cast<float>(rand_r(seed)) /
                static_cast<float>(RAND_MAX) * M_PI * 2.0;

        hposold[ip * 4] = sqrt(2.0 * NUMPART * rand_zerotoone)
                * cos(rand_zeroto2pi);
        hposold[ip * 4 + 1] = sqrt(2.0 * NUMPART * rand_zerotoone)
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
                                  4 * NUMPART * sizeof(float), hposold);
        queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0,
                                  4 * NUMPART * sizeof(float), hvelold);
    } catch(cl::Error error) {
        std::cerr << error.what()
                  << "(" << oclErrorString(error.err()) << ")"
                  << std::endl;
        exit(1);
    }
}

void PropagateNBodySystem::PropagateDoubleStep(float DeltaTime) {
    try {
        // Set arguments to kernel
        kernel_eom->setArg(0, *gposold);
        kernel_eom->setArg(1, *gvelold);
        kernel_eom->setArg(2, *gposnew);
        kernel_eom->setArg(3, *gvelnew);
        kernel_eom->setArg(4, cl::Local(GROUP_SIZE * 4 * sizeof(float)));
        kernel_eom->setArg(5, NUMPART);
        kernel_eom->setArg(6, DeltaTime);

        // Run the kernel on specific ND range
        cl::NDRange global(NUMPART);
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

void PropagateNBodySystem::WriteState(const char* filename) {
    // transfer memory back to CPU
    queue->enqueueReadBuffer(*gposold, CL_TRUE, 0,
                             4 * NUMPART * sizeof(float), hposold);
    queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0,
                             4 * NUMPART * sizeof(float), hvelold);
    FILE *fd = fopen(filename, "w");
    for (int i = 0; i < NUMPART; i++) {
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

void PropagateNBodySystem::GetElectrons(int NumBodies, float *pos, float *vel) {
    // transfer memory from GPU to CPU
    queue->enqueueReadBuffer(*gposold, CL_TRUE, 0,
                             4 * NumBodies * sizeof(float), pos);
    queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0,
                             4 * NumBodies * sizeof(float), vel);
    queue->finish();
}

void PropagateNBodySystem::PutElectrons(int NumBodies, float *pos, float *vel) {
    // transfer memory from CPU to GPU
    queue->enqueueWriteBuffer(*gposold, CL_TRUE, 0,
                              4 * NumBodies * sizeof(float), pos);
    queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0,
                              4 * NumBodies * sizeof(float), vel);
    queue->finish();
}

void PropagateNBodySystem::RunSimulation(void) {
    float deltaTime = 1.0e-3f;
    int nt = 1001;

    time_t c0, c1;

    init();
    PutElectrons(NUMPART, hposold, hvelold);

    time(&c0);
    for (int it = 0; it < nt; it++)  // Main propagation loop.
        PropagateDoubleStep(deltaTime);

    time(&c1);
#ifdef GPU
    fprintf(stderr,
            "GPU propagation for %d particles "
            "over %d doublesteps took %.0f s\n",
            NUMPART, nt, difftime(c1, c0));
#else
    fprintf(stderr,
            "CPU propagation for %d particles "
            "over %d doublesteps took %.0f s\n",
            NUMPART, nt, difftime(c1, c0));
#endif
}

int main(int argc, char *argv[]) {
    cl::Context *context;

    if (argc == 3) {
        GROUP_SIZE = atoi(argv[1]);
        NUMPART = GROUP_SIZE * atoi(argv[2]);
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

    PropagateNBodySystem Plasma(context);

    Plasma.RunSimulation();

    return 0;
}
