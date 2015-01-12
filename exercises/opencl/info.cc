/*
 * info.cc - Print OpenCL platform and device info.
 */
#include <iostream>
#include <vector>

#include <CL/cl.hpp>  // NOLINT(build/include_order)

// Display Device information.
void printDeviceInfo(const int index, const cl::Device &device) {
    std::cout
        << "\n   Device " << index << ": "
        <<          device.getInfo<CL_DEVICE_NAME>()
        << "\n\t Device Version     : "
        <<          device.getInfo<CL_DEVICE_VERSION>()
        << "\n\t OpenCL C Version   : "
        <<          device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()
        << "\n\t Compute Units      : "
        <<          device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
        << "\n\t Max Work Group Size: "
        <<          device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
        << "\n\t Clock Frequency    : "
        <<          device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
        << "\n\t Local Memory Size  : "
        <<          device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()
        << "\n\t Global Memory Size : "
        <<          device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    // Check if the device supports double precision.
    std::string str = device.getInfo<CL_DEVICE_EXTENSIONS>();
    size_t found = str.find("cl_khr_fp64");
    std::cout << "\n\t Double Precision   : ";
    if (found != std::string::npos)
      std::cout << "yes\n";
    else
      std::cout <<  "no\n";
}

// Display Platform information.
void printPlatformInfo(const int index, const cl::Platform &platform) {
  // Display the platform information.
  std::cout << "----------------------------------------------"
            << "\nPlatform " << index << ": "
            << platform.getInfo<CL_PLATFORM_NAME>()
            << "\nVendor    : " << platform.getInfo<CL_PLATFORM_VENDOR>()
            << "\nVersion   : " << platform.getInfo<CL_PLATFORM_VERSION>()
            << "\n----------------------------------------------\n";

  // Get the devices on the current platform.
  std::vector <cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL , & devices);

  // Loop over the devices.
  for (size_t i = 0; i < devices.size(); i++)
    printDeviceInfo(i + 1, devices[i]);
  std::cout << "\n----------------------------------------------\n";
}

// Print information for all platforms.
int main() {
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    for (size_t i = 0; i < platforms.size(); i++)
      printPlatformInfo(i + 1, platforms[i]);

    return 0;
}
