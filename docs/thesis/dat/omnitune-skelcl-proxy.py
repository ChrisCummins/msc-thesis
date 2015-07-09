class SkelCLProxy(omnitune.Proxy):
  ...
  @dbus.service.method("org.omnitune.skelcl",
                       in_signature='siiiiiiiisss',
                       out_signature='(nn)')
  def RequestWgsize(self, device_name, device_count,
                    north, south, east, west, data_width,
                    data_height, type_in, type_out, source,
                    max_wgsize):
    ...
    return wg_c, wg_r

  @dbus.service.method("org.omnitune.skelcl",
                       in_signature='siiiiiiiisss',
                       out_signature='(nn)')
  def RequestTrainingWgsize(self, device_name, device_count,
                            north, south, east, west, data_width,
                            data_height, type_in, type_out, source,
                            max_wgsize):
    ...
    return training_wg_c, training_wg_r
