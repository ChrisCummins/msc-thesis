/*! \file mapoverlap_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "mapoverlap_kernels.h"
#include "mapoverlap_convol_kernels.h"

#include "device_mem_pointer_cl.h"

namespace skepu
{




/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic MapOverlap kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename MapOverlapFunc>
void MapOverlap<MapOverlapFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_mapOverlapFunc->func_CL;
   std::string kernelName;

   std::string kernelName_MatRow, kernelName_MatCol, kernelName_MatColMulti;

   if(m_mapOverlapFunc->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   
   kernelSource = MapOverlapKernel_CL+ MapOverlapKernel_CL_Matrix_Col + MapOverlapKernel_CL_Matrix_Row + MapOverlapKernel_CL_Matrix_ColMulti;

   kernelName = "MapOverlapKernel_" + m_mapOverlapFunc->funcName_CL;

   kernelName_MatRow = "MapOverlapKernel_MatRowWise_" + m_mapOverlapFunc->funcName_CL;

   kernelName_MatCol = "MapOverlapKernel_MatColWise_" + m_mapOverlapFunc->funcName_CL;

   kernelName_MatColMulti = "MapOverlapKernel_MatColWiseMulti_" + m_mapOverlapFunc->funcName_CL;

   replaceTextInString(kernelSource, std::string("TYPE"), m_mapOverlapFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_mapOverlapFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_mapOverlapFunc->funcName_CL);

   // check for extra user-supplied opencl code for custome datatype
   totalSource.append( read_file_into_string(OPENCL_SOURCE_FILE_NAME) );

   totalSource.append(funcSource);
   totalSource.append(kernelSource);
   
   if(sizeof(size_t) <= 4)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned int ");
   else if(sizeof(size_t) <= 8)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned long ");
   else
      SKEPU_ERROR("OpenCL code compilation issue: sizeof(size_t) is bigger than 8 bytes: " << sizeof(size_t));

   DEBUG_TEXT_LEVEL3(totalSource);

   //Builds the code and creates kernel for all devices
   for(std::vector<Device_CL*>::iterator it = m_environment->m_devices_CL.begin(); it != m_environment->m_devices_CL.end(); ++it)
   {
      const char* c_src = totalSource.c_str();
      cl_int err;
      cl_program temp_program;
      cl_kernel temp_kernel;

      temp_program = clCreateProgramWithSource((*it)->getContext(), 1, &c_src, NULL, &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating OpenCLsource!!\n");
      }

      err = clBuildProgram(temp_program, 0, NULL, NULL, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         #if SKEPU_DEBUG > 0
            cl_build_status build_status;
            clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
            if (build_status != CL_SUCCESS)
            {
               char *build_log;
               size_t ret_val_size;
               clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
               build_log = new char[ret_val_size+1];
               clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

               // to be carefully, terminate with \0
               // there's no information in the reference whether the string is 0 terminated or not
               build_log[ret_val_size] = '\0';

               SKEPU_ERROR("Error building OpenCL program!!\n" <<err <<"\nBUILD LOG:\n" << build_log);

               delete[] build_log;
            }
         #else
            SKEPU_ERROR("Error building OpenCL program. " << err <<"\n");
         #endif
      }

      temp_kernel = clCreateKernel(temp_program, kernelName.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelName_MatRow.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_MatRow.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_Mat_Row_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelName_MatCol.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_MatCol.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_Mat_Col_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelName_MatColMulti.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_MatColMulti.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_Mat_ColMulti_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
}

/*!
 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
 *  The function uses only \em one device which is decided by a parameter. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID Integer deciding which device to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   BackEndParams bp = m_execPlan->find_(inputEnd-inputBegin);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t n = inputEnd-inputBegin;
   size_t overlap = m_mapOverlapFunc->overlap;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   size_t out_offset = 0;
   size_t out_numelements = n;

   int _poly;
   typename InputIterator::value_type _pad;

   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = 0;
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = 0;
   }

   // Constructs a wrap vector, which is used if cyclic edge policy is specified.
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);
   std::vector<typename InputIterator::value_type> wrap(2*overlap);
   if(poly == CYCLIC)
   {
      inputBegin.getParent().updateHostAndInvalidateDevice();
      for(size_t i = 0; i < overlap; ++i)
      {
         wrap[i] = inputEndMinusOverlap(i);
         wrap[overlap+i] = inputBegin(i);
      }
   }

   // Copy wrap vector to device.
   DeviceMemPointer_CL<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_kernels_CL.at(deviceID).second);
   wrap_mem_p.copyHostToDevice();

   numThreads = std::min(maxThreads, (size_t)n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), n, m_kernels_CL.at(deviceID).second, true);
   typename OutputIterator::device_pointer_type_cl out_mem_p = outputBegin.getParent().updateDevice_CL(outputBegin.getAddress(), n, m_kernels_CL.at(deviceID).second, false);

   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p->getDeviceDataPointer();
   cl_mem wrap_p = wrap_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_kernels_CL.at(deviceID).first;

   // Sets the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
   clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
   clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_offset);
   clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&out_numelements);
   clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
   clSetKernelArg(kernel, 8, sizeof(typename InputIterator::value_type), (void*)&_pad);
   clSetKernelArg(kernel, 9, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // Launches the kernel (asynchronous)
   err = clEnqueueNDRangeKernel(m_kernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}

/*!
 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
 *  The function uses a variable number of devices, dividing the range of elemets equally among the participating devices each mapping
 *  its part. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = inputEnd - inputBegin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   size_t overlap = m_mapOverlapFunc->overlap;

   int _poly;
   typename InputIterator::value_type _pad;

   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = 0;
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = 0;
   }

//    //Need to get new values from other devices so that the overlap between devices is up to date.
//    //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
// Acutally we dont need to validate the data here as for normal cases, it should work fine. We are not writing to InputBegin
//    inputBegin.getParent().updateHostAndInvalidateDevice();

   // Constructs a wrap vector, which is used if cyclic edge policy is specified.
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);
   std::vector<typename InputIterator::value_type> wrap(2*overlap);

   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t i = 0; i < overlap; ++i)
      {
         wrap[i] = inputEndMinusOverlap(i);
         wrap[overlap+i] = inputBegin(i);
      }
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      BackEndParams bp = m_execPlan->find_(totalNumElements);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];
      size_t out_offset, out_numelements, n;

      // Copy wrap vector to device.
      DeviceMemPointer_CL<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_kernels_CL.at(i).second);
      wrap_mem_p.copyHostToDevice();
      cl_mem wrap_p = wrap_mem_p.getDeviceDataPointer();

      typename InputIterator::device_pointer_type_cl in_mem_p;
      typename OutputIterator::device_pointer_type_cl out_mem_p;

      // Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
      if(i == 0)
      {
         in_mem_p = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem+overlap, m_kernels_CL.at(i).second, true);
         out_offset = 0;
         out_numelements = numElem;
         n = numElem+overlap;
      }
      else if(i == numDevices-1)
      {
         in_mem_p = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+overlap, m_kernels_CL.at(i).second, true);
         out_offset = overlap;
         out_numelements = numElem;
         n = numElem+overlap;
      }
      else
      {
         in_mem_p = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+2*overlap, m_kernels_CL.at(i).second, true);
         out_offset = overlap;
         out_numelements = numElem;
         n = numElem+2*overlap;
      }

      // Allocate memory for output.
      out_mem_p = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);

      numThreads = std::min(maxThreads, (size_t)n);
      numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

      cl_mem in_p = in_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_offset);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&out_numelements);
      clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
      clSetKernelArg(kernel, 8, sizeof(typename InputIterator::value_type), (void*)&_pad);
      clSetKernelArg(kernel, 9, sharedMemSize, NULL);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();
   }
}

/*!
 *  Performs the MapOverlap on a whole Vector. With a seperate output vector. Wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CL(Vector<T>& input, Vector<T>& output, EdgePolicy poly, T pad, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CL(input.begin(), input.end(), output.begin(), poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
 *  on multiple devices, calling mapOverlapNumDevices_CL.
 *  Using \em OpenCL as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingle_CL(inputBegin, inputEnd, outputBegin, poly, pad, 0);
   }
   else
   {
      mapOverlapNumDevices_CL(inputBegin, inputEnd, outputBegin, poly, pad, numDevices);
   }
}

/*!
 *  Performs the MapOverlap on a whole Vector. With itself as output. A wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU). Using \em OpenCL as backend.
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CL(Vector<T>& input, EdgePolicy poly, T pad, int useNumGPU)
{
   CL(input.begin(), input.end(), poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output. Since a seperate output is needed, a
 *  temporary output vector is created and copied to the input vector at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   Vector<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputEnd-inputBegin);

   CL(inputBegin, inputEnd, output.begin(), poly, pad, useNumGPU);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}






//*******************************************************************
//*******************************************************************
// ------------------------------------------------------------------
// ------------------------------------------------------------------
//*******************************************************************
//*******************************************************************

//*******************************************************************
//*******************************************************************
// ------------------------------------------------------------------
// ------------------------------------------------------------------
//*******************************************************************
//*******************************************************************/

/*!
 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CL() method.
 *
 * \param numThreads Number of threads in a thread block.
 * \param deviceID The device ID.
 */
template <typename MapOverlapFunc>
template <typename T>
bool MapOverlap<MapOverlapFunc>::sharedMemAvailable_CL(size_t &numThreads, unsigned int deviceID)
{
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t maxShMem = ((m_environment->m_devices_CL.at(deviceID)->getSharedMemPerBlock())/sizeof(T))-SHMEM_SAFITY_BUFFER; // little buffer for other usage

   size_t orgThreads = numThreads;

   numThreads = ( ((numThreads+2*overlap)<maxShMem)? numThreads : maxShMem-(2*overlap) );

   if(orgThreads == numThreads) // return true when nothing changed because of overlap constraint
      return true;

   if(numThreads<8) // not original numThreads then atleast 8 threads should be there
   {
      SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted!!!\n");
      numThreads = -1;
   }
   return false;
}



/*!
 * Helper method used for calculating optimal thread count. For row- or column-wise overlap,
 * it determines a thread block size with perfect division of problem size.
 *
 * \param width The problem size.
 * \param numThreads Number of threads in a thread block.
 * \param deviceID The device ID.
 */
template <typename MapOverlapFunc>
template <typename T>
size_t MapOverlap<MapOverlapFunc>::getThreadNumber_CL(size_t width, size_t &numThreads, unsigned int deviceID)
{
   // first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
   if(!sharedMemAvailable_CL<T>(numThreads, deviceID) && numThreads<1)
   {
      SKEPU_ERROR("Too lower overlap size to continue.\n");
   }

   if( (width % numThreads) == 0)
      return (width / numThreads);

   for(size_t i=numThreads-1; i>=1; i--) // decreament numThreads and see which one is a perfect divisor
   {
      if( (width % numThreads)==0)
         return (width / numThreads);
   }

   return -1;
}




/*!
 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
 *  The function uses only \em one device which is decided by a parameter. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID, OverlapPolicy overlapPolicy)
{
   if(overlapPolicy == OVERLAP_ROW_COL_WISE)
   {
      Matrix<typename InputIterator::value_type> tmp_m(outputBegin.getParent().total_rows(), outputBegin.getParent().total_cols());
      mapOverlapSingle_CL_Row(inputBegin, inputEnd, tmp_m.begin(), poly, pad, deviceID);
      mapOverlapSingle_CL_Col(tmp_m.begin(), tmp_m.end(), outputBegin, poly, pad, deviceID);
   }
   else if(overlapPolicy == OVERLAP_COL_WISE)
   {
      mapOverlapSingle_CL_Col(inputBegin, inputEnd, outputBegin, poly, pad, deviceID);
   }
   else
      mapOverlapSingle_CL_Row(inputBegin, inputEnd, outputBegin, poly, pad, deviceID);
}


/*!
 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID The integer specifying OpenCL device to execute the operation.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   cl_int err;
   size_t n = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;

   size_t numBlocks;
   size_t numThreads;

   size_t overlap = m_mapOverlapFunc->overlap;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   size_t out_offset = 0;
   size_t out_numelements = n;

   size_t numrows = inputBegin.getParent().total_rows();
   size_t rowWidth = inputBegin.getParent().total_cols(); // same as numcols

   size_t trdsize= rowWidth;
   size_t blocksPerRow = 1;
   if(rowWidth> maxThreads)
   {
      size_t tmp= getThreadNumber_CL<typename InputIterator::value_type>(rowWidth, maxThreads, deviceID);
      if(tmp==-1 || (tmp*numrows)>maxBlocks)
      {
         SKEPU_ERROR("ERROR! Row width is larger than maximum thread size."<<rowWidth<<"  "<<maxThreads<<" \n");
      }
      blocksPerRow = tmp;
      trdsize = rowWidth / blocksPerRow;

      if(trdsize<overlap)
      {
         SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.\n");
      }
   }

   int _poly;
   typename InputIterator::value_type _pad;

   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = typename InputIterator::value_type();
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = typename InputIterator::value_type();
   }

   // Constructs a wrap vector, which is used if cyclic edge policy is specified.
   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*numrows);


   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t row=0; row< numrows; row++)
      {
         inputEndTemp = inputBeginTemp+rowWidth;

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(row*twoOverlap)] = inputEndTemp(i-overlap);// inputEndMinusOverlap(i);
            wrap[(overlap+i)+(row*twoOverlap)] = inputBeginTemp(i);
         }
         inputBeginTemp += rowWidth;
      }
   }
   /*    else
         wrap.resize(1); // not used so minimize overhead;*/

   // Copy wrap vector to device.
   DeviceMemPointer_CL<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_kernels_Mat_Row_CL.at(deviceID).second);
   wrap_mem_p.copyHostToDevice();


   numThreads = trdsize; //std::min(maxThreads, rowWidth);
   numBlocks = std::max(1, (int)std::min( (blocksPerRow * numrows), maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), numrows, rowWidth, m_kernels_Mat_Row_CL.at(deviceID).second, true);
   typename OutputIterator::device_pointer_type_cl out_mem_p = outputBegin.getParent().updateDevice_CL(outputBegin.getAddress(), numrows, rowWidth, m_kernels_Mat_Row_CL.at(deviceID).second, false);



   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p->getDeviceDataPointer();
   cl_mem wrap_p = wrap_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_kernels_Mat_Row_CL.at(deviceID).first;

   // Sets the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
   clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
   clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_offset);
   clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&out_numelements);
   clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
   clSetKernelArg(kernel, 8, sizeof(typename InputIterator::value_type), (void*)&_pad);
   clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&blocksPerRow);
   clSetKernelArg(kernel, 10, sizeof(size_t), (void*)&rowWidth);
   clSetKernelArg(kernel, 11, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // Launches the kernel (asynchronous)
   err = clEnqueueNDRangeKernel(m_kernels_Mat_Row_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}



/*!
 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL_RowMulti(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices)
{
   cl_int err;

   size_t totalElems = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(totalElems/numDevices);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;

   size_t numBlocks;
   size_t numThreads;

   size_t overlap = m_mapOverlapFunc->overlap;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   size_t numrows = inputBegin.getParent().total_rows();
   size_t rowWidth = inputBegin.getParent().total_cols();

   size_t trdsize= rowWidth;
   size_t blocksPerRow = 1;
   if(rowWidth> maxThreads)
   {
      size_t tmp= getThreadNumber_CL<typename InputIterator::value_type>(rowWidth, maxThreads, 0);
      if(tmp==-1 || (tmp*numrows)>maxBlocks)
      {
         SKEPU_ERROR("ERROR! Row width is larger than maximum thread size!"<<rowWidth<<"  "<<maxThreads<<" \n");
      }
      blocksPerRow = tmp;
      trdsize = rowWidth / blocksPerRow;

      if(trdsize<overlap)
      {
         SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.\n");
      }
   }

   int _poly;
   typename InputIterator::value_type _pad;

   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = typename InputIterator::value_type();
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = typename InputIterator::value_type();
   }

   size_t numRowsPerSlice = (numrows / numDevices);
   size_t numElemPerSlice = numRowsPerSlice*rowWidth;
   size_t restRows = (numrows % numDevices);

   //Need to get new values from other devices so that the overlap between devices is up to date.
   //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
//    inputBegin.getParent().updateHostAndInvalidateDevice();

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*numrows);

   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;

   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t row=0; row< numrows; row++)
      {
         inputEndTemp = inputBeginTemp+rowWidth;

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(row*twoOverlap)] = inputEndTemp(i-overlap);// inputEndMinusOverlap(i);
            wrap[(overlap+i)+(row*twoOverlap)] = inputBeginTemp(i);
         }
         inputBeginTemp += rowWidth;
      }
   }

   numThreads = trdsize; //std::min(maxThreads, rowWidth);

   m_mapOverlapFunc->setStride(1);


   typename InputIterator::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   typename InputIterator::device_pointer_type_cl wrap_mem_p[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      // Copy wrap vector to device.
      wrap_mem_p[i] = new DeviceMemPointer_CL<typename InputIterator::value_type>(&wrap[(i*numRowsPerSlice*overlap*2)], &wrap[(i*numRowsPerSlice*overlap*2)], (numRowsPerSlice*overlap*2), m_kernels_Mat_Row_CL.at(i).second);

      // Copy elements to device and allocate output memory.
      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, m_kernels_Mat_Row_CL.at(i).second, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, m_kernels_Mat_Row_CL.at(i).second, false);
   }

   size_t out_offset = 0;

   // we will divide the computation row-wise... copy data and do operation
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;
      size_t numElems;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      numElems = numRows * rowWidth;

      if(poly == CYCLIC)
      {
         // Copy wrap vector only if it is a CYCLIC overlap policy.
         wrap_mem_p[i]->copyHostToDevice();
      }

      numBlocks = std::max((size_t)1, std::min( (blocksPerRow * numRows), maxBlocks));
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

      // Copy elements to device and allocate output memory.
      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, m_kernels_Mat_Row_CL.at(i).second, true);

      cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();
      cl_mem wrap_p = wrap_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_Mat_Row_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElems);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_offset);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&numElems);
      clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
      clSetKernelArg(kernel, 8, sizeof(typename InputIterator::value_type), (void*)&_pad);
      clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&blocksPerRow);
      clSetKernelArg(kernel, 10, sizeof(size_t), (void*)&rowWidth);
      clSetKernelArg(kernel, 11, sharedMemSize, NULL);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_Mat_Row_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }

   // to properly de-allocate the memory
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete wrap_mem_p[i];
   }
}




/*!
 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID The integer specifying OpenCL device to execute the operation.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   cl_int err;
   size_t n = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;

   size_t numBlocks;
   size_t numThreads;

   size_t overlap = m_mapOverlapFunc->overlap;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   size_t out_offset = 0;
   size_t out_numelements = n;

   size_t colWidth = inputBegin.getParent().total_rows();
   size_t numcols = inputBegin.getParent().total_cols();

   size_t trdsize= colWidth;
   size_t blocksPerCol = 1;
   if(colWidth> maxThreads)
   {
      size_t tmp= getThreadNumber_CL<typename InputIterator::value_type>(colWidth, maxThreads, deviceID);

      if(tmp==-1 || (blocksPerCol*numcols)>maxBlocks)
      {
         SKEPU_ERROR("ERROR! Col width is larger than maximum thread size!"<<colWidth<<"  "<<maxThreads<<" \n");
      }
      blocksPerCol = tmp;
      trdsize = colWidth / blocksPerCol;
      if(trdsize<overlap)
      {
         SKEPU_ERROR("ERROR! Thread size should be larger than overlap width!\n");
      }
   }

   int _poly;
   typename InputIterator::value_type _pad;
   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = typename InputIterator::value_type();
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = typename InputIterator::value_type();
   }

   // Constructs a wrap vector, which is used if cyclic edge policy is specified.
   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*(n/colWidth));

   if(poly == CYCLIC)
   {
      size_t stride = numcols;
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t col=0; col< numcols; col++)
      {
         inputEndTemp = inputBeginTemp+(numcols*(colWidth-1));
         inputEndMinusOverlap = (inputEndTemp - (overlap-1)*stride);

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(col*twoOverlap)] = inputEndMinusOverlap(i*stride);
            wrap[(overlap+i)+(col*twoOverlap)] = inputBeginTemp(i*stride);
         }
         inputBeginTemp++;
      }
   }
   // Copy wrap vector to device.
   DeviceMemPointer_CL<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_kernels_Mat_Col_CL.at(deviceID).second);
   wrap_mem_p.copyHostToDevice();

   numThreads = trdsize; //std::min(maxThreads, rowWidth);
   numBlocks = std::max(1, (int)std::min( (blocksPerCol * numcols), maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), colWidth, numcols, m_kernels_Mat_Col_CL.at(deviceID).second, true);
   typename OutputIterator::device_pointer_type_cl out_mem_p = outputBegin.getParent().updateDevice_CL(outputBegin.getAddress(), colWidth, numcols, m_kernels_Mat_Col_CL.at(deviceID).second, false);

   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p->getDeviceDataPointer();
   cl_mem wrap_p = wrap_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_kernels_Mat_Col_CL.at(deviceID).first;

   // Sets the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
   clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
   clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_offset);
   clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&out_numelements);
   clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
   clSetKernelArg(kernel, 8, sizeof(typename InputIterator::value_type), (void*)&_pad);
   clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&blocksPerCol);
   clSetKernelArg(kernel, 10, sizeof(size_t), (void*)&numcols);
   clSetKernelArg(kernel, 11, sizeof(size_t), (void*)&colWidth);
   clSetKernelArg(kernel, 12, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // Launches the kernel (asynchronous)
   err = clEnqueueNDRangeKernel(m_kernels_Mat_Col_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}




/*!
 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
 *  Used internally by other methods to apply column-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingle_CL_ColMulti(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices)
{
   cl_int err;

   size_t totalElems = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(totalElems/numDevices);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   size_t colWidth = inputBegin.getParent().total_rows();
   size_t numCols = inputBegin.getParent().total_cols();

   size_t numRowsPerSlice = (colWidth / numDevices);
   size_t numElemPerSlice = numRowsPerSlice*numCols;
   size_t restRows = (colWidth % numDevices);

   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);

   std::vector<typename InputIterator::value_type> wrapStartDev(overlap*numCols);
   std::vector<typename InputIterator::value_type> wrapEndDev(overlap*numCols);

   if(poly == CYCLIC)
   {
      size_t stride = numCols;

      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t col=0; col< numCols; col++)
      {
         inputEndTemp = inputBeginTemp+(numCols*(colWidth-1));
         inputEndMinusOverlap = (inputEndTemp - (overlap-1)*stride);

         for(size_t i = 0; i < overlap; ++i)
         {
            wrapStartDev[i+(col*overlap)] = inputEndMinusOverlap(i*stride);
            wrapEndDev[i+(col*overlap)] = inputBeginTemp(i*stride);
         }
         inputBeginTemp++;
      }
   }
   typename InputIterator::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   typename InputIterator::device_pointer_type_cl wrapStartDev_mem_p[MAX_GPU_DEVICES];
   typename InputIterator::device_pointer_type_cl wrapEndDev_mem_p[MAX_GPU_DEVICES];

   size_t trdsize[MAX_GPU_DEVICES];
   size_t blocksPerCol[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;

      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      trdsize[i] = numRows;
      blocksPerCol[i] = 1;

      if(numRows> maxThreads)
      {
         size_t tmp = getThreadNumber_CL<typename InputIterator::value_type>(numRows, maxThreads, i);
         if(tmp ==-1 || (tmp*numCols)>maxBlocks)
         {
            SKEPU_ERROR("ERROR! Col width is larger than maximum thread size!"<<colWidth<<"  "<<maxThreads<<" \n");
         }
         blocksPerCol[i] = tmp;
         trdsize[i] = numRows / blocksPerCol[i];

         if(trdsize[i]<overlap)
         {
            SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.\n");
         }
      }

      size_t overlapElems = overlap*numCols;

      if(i == 0)
         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numRows+overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, false);
      else if(i == numDevices-1)
         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, false);
      else
         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+2*overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, false);

      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numRows, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, false);

      // wrap vector, don't copy just allocates space on OpenCL device.
      wrapStartDev_mem_p[i] = new DeviceMemPointer_CL<typename InputIterator::value_type>(&wrapStartDev[0], &wrapStartDev[0], wrapStartDev.size(), m_kernels_Mat_ColMulti_CL.at(i).second);
      wrapEndDev_mem_p[i] = new DeviceMemPointer_CL<typename InputIterator::value_type>(&wrapEndDev[0], &wrapEndDev[0], wrapEndDev.size(), m_kernels_Mat_ColMulti_CL.at(i).second);
   }

   int _poly;
   int _deviceType[MAX_GPU_DEVICES];
   typename InputIterator::value_type _pad;
   // Sets the pad and edge policy values that are sent to the kernel
   if(poly == CONSTANT)
   {
      _poly = 0;
      _pad = pad;
   }
   else if(poly == CYCLIC)
   {
      _poly = 1;
      _pad = typename InputIterator::value_type();
   }
   else if(poly == DUPLICATE)
   {
      _poly = 2;
      _pad = typename InputIterator::value_type();
   }

   // we will divide the computation row-wise... copy data and do operation
   for(size_t i=0; i< numDevices; i++)
   {
      size_t n;
      size_t numRows;
      size_t numElems;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      size_t in_offset, overlapElems = overlap*numCols;
      numElems = numRows * numCols;

      // Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
      if(i == 0)
      {
         _deviceType[i] = -1;

         if(poly == CYCLIC)
            wrapStartDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?

         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numRows+overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, true);

         in_offset = 0;
         n = numElems; //numElem+overlap;
      }
      else if(i == numDevices-1)
      {
         _deviceType[i] = 1;

         if(poly == CYCLIC)
            wrapEndDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?

         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, true);

         in_offset = overlapElems;
         n = numElems; //+overlap;
      }
      else
      {
         _deviceType[i] = 0;

         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+2*overlap, numCols, m_kernels_Mat_ColMulti_CL.at(i).second, true);

         in_offset = overlapElems;
         n = numElems; //+2*overlap;
      }

      numThreads = trdsize[i]; //std::min(maxThreads, rowWidth);
      numBlocks = std::max(1, (int)std::min( (size_t)(blocksPerCol[i] * numCols), maxBlocks));
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

      m_mapOverlapFunc->setStride(1);

      cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();
      cl_mem wrap_p = wrapStartDev_mem_p[i]->getDeviceDataPointer();

      if(poly == CYCLIC && i==(numDevices-1))
         wrap_p = wrapEndDev_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_Mat_ColMulti_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&wrap_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&overlap);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&in_offset);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&n);
      clSetKernelArg(kernel, 7, sizeof(int), (void*)&_poly);
      clSetKernelArg(kernel, 8, sizeof(int), (void*)&_deviceType[i]);
      clSetKernelArg(kernel, 9, sizeof(typename InputIterator::value_type), (void*)&_pad);
      clSetKernelArg(kernel, 10, sizeof(size_t), (void*)&blocksPerCol);
      clSetKernelArg(kernel, 11, sizeof(size_t), (void*)&numCols);
      clSetKernelArg(kernel, 12, sizeof(size_t), (void*)&numRows);
      clSetKernelArg(kernel, 13, sharedMemSize, NULL);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_Mat_ColMulti_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }

   // to properly de-allocate the memory
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete wrapStartDev_mem_p[i];
      delete wrapEndDev_mem_p[i];
   }
}







/*!
 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
 *  The function uses a variable number of devices, dividing the range of elemets equally among the participating devices each mapping
 *  its part. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices, OverlapPolicy overlapPolicy)
{
   // Currently this method uses one device.
   if(overlapPolicy == OVERLAP_ROW_COL_WISE)
   {
      Matrix<typename InputIterator::value_type> tmp_m(outputBegin.getParent().total_rows(), outputBegin.getParent().total_cols());
      mapOverlapSingle_CL_RowMulti(inputBegin, inputEnd, tmp_m.begin(), poly, pad,numDevices);
      mapOverlapSingle_CL_ColMulti(tmp_m.begin(), tmp_m.end(), outputBegin, poly, pad, 0);
   }
   else if(overlapPolicy == OVERLAP_COL_WISE)
      mapOverlapSingle_CL_ColMulti(inputBegin, inputEnd, outputBegin, poly, pad,numDevices);
   else
      mapOverlapSingle_CL_RowMulti(inputBegin, inputEnd, outputBegin, poly, pad,numDevices);
}

/*!
 *  Performs the MapOverlap on a whole Matrix. With a seperate output matrix. Wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CL(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.total_rows(), input.total_cols());
   }

   CL(input.begin(), input.end(), output.begin(), overlapPolicy, poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
 *  on multiple devices, calling mapOverlapNumDevices_CL.
 *  Using \em OpenCL as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingle_CL(inputBegin, inputEnd, outputBegin, poly, pad, 0, overlapPolicy);
   }
   else
   {
      mapOverlapNumDevices_CL(inputBegin, inputEnd, outputBegin, poly, pad, numDevices, overlapPolicy);
   }
}

/*!
 *  Performs the MapOverlap on a whole Matrix. With itself as output. A wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU). Using \em OpenCL as backend.
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CL(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad, int useNumGPU)
{
   CL(input.begin(), input.end(), overlapPolicy, poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output. Since a seperate output is needed, a
 *  temporary output matrix is created and copied to the input matrix at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead. Using \em OpenCL as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   Matrix<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputEnd.getParent().total_rows(),inputEnd.getParent().total_cols());

   CL(inputBegin, inputEnd, output.begin(), overlapPolicy, poly, pad, useNumGPU);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}




}

#endif

