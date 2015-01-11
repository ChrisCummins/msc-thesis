#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// The string that to "compute".
__constant char hw[] = "Hello OpenCL!\n";

__kernel void hello(__global char *out) {
  size_t tid = get_global_id(0);
  out[tid] = hw[tid];
}
