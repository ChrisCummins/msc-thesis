cl_uint wgsize[2];
...
omnitune::stencil::requestWgSize(deviceName, deviceCount,
                                 _shape, output.columnCount(),
                                 output.rowCount(), TinStr,
                                 ToutStr, _program.getCode(),
                                 maxWgSize, &wgsize[0]);
...
devicePtr->enqueue(kernel,
                   cl::NDRange(global[0], global[1]),
                   cl::NDRange(wgsize[0], wgsize[1]),
                   cl::NDRange(0, global[2]),
                   invokeAfter)
