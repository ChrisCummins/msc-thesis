template <typename Tin, typename Tout>
template <typename T, typename... Args>
void Stencil<Tout(Tin)>::execute(const Matrix<Tin>& input,
                                 Matrix<Tout>& output,
                                 ...) const {
  cl_uint wgsize[2];
  ...
  omnitune::stencil::requestWgsize(deviceName, deviceCount,
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
}
