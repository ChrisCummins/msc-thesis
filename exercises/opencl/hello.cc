/*
 * hello.cc - "Hello World" using OpenCL.
 *
 * Source taken from "Introl OpenCL Tutorial", from AMD. See:
 *
 *   http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/introductory-tutorial-to-opencl/
 */

#include "./hello.h"

int main(int argc, char *argv[]) {
  cl_int err;

  // Get a list of available platforms.
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Check that list of platforms contains elements.
  checkError(platforms.size() ? CL_SUCCESS : -1, "platforms.size  0");
  std::cerr << "Platform number is: " << platforms.size() << std::endl;

  // Print platform info.
  std::string platformVendor;
  platforms[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
  std::cerr << "Platform info: " << platformVendor << std::endl;

  // Create OpenCL context.
  cl_context_properties cprops[3] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)(platforms[0])(),
    0
  };

  cl::Context context(CL_DEVICE_TYPE_CPU, cprops, NULL, NULL, &err);
  checkError(err, "Context::Context()");

  // Create results buffer.
  char *buffer = new char[hw.length() + 1];
  cl::Buffer outCL(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                   hw.length() + 1, buffer, &err);
  checkError(err, "Buffer::Buffer()");

  // Get devices.
  std::vector<cl::Device> devices;
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  checkError(devices.size() ? CL_SUCCESS : -1, "devices.size() != 0");

  // Compile kernel "hello.cl".
  std::ifstream file("hello.cl");
  checkError(file.is_open() ? CL_SUCCESS : -1, "hello.cl");
  std::string prog(std::istreambuf_iterator<char>(file),
                   (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(1, std::make_pair(prog.c_str(),
                                                prog.length() + 1));
  cl::Program program(context, source);
  err = program.build(devices, "");
  checkError(err, "Program::build()");

  cl::Kernel kernel(program, "hello", &err);
  checkError(err, "Kernel::Kernel()");
  err = kernel.setArg(0, outCL);
  checkError(err, "Kernel::setArg()");

  // Enqueue command for execution.
  cl::CommandQueue queue(context, devices[0], 0, &err);
  checkError(err, "CommandQueue::CommandQueue()");
  cl::Event event;
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(hw.length() + 1),
                                   cl::NDRange(1, 1), NULL, &event);
  checkError(err, "CommandQueue:enqueueNDRAngeKernel()");

  // Wait for command to complete.
  event.wait();

  // Read results buffer.
  err = queue.enqueueReadBuffer(outCL, CL_TRUE, 0, hw.length() + 1, buffer);
  checkError(err, "CommandQueue::enqueueReadBuffer()");
  std::cout << buffer;

  // Free buffer.
  delete[] buffer;

  return EXIT_SUCCESS;
}
