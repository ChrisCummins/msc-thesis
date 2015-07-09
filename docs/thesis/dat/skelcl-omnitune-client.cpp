void requestWgSize(const std::string &deviceName,
                   const int deviceCount,
                   const skelcl::StencilShape &shape,
                   const int dataWidth,
                   const int dataHeight,
                   const std::string &Tin,
                   const std::string &Tout,
                   const std::string &source,
                   const size_t maxWgSize,
                   cl_uint *const local) {
  init(); // Initialise DBus connection and proxy objects.

  std::vector<Glib::VariantBase> _args;
  _args.push_back(Glib::Variant<std::string>::create(deviceName));
  _args.push_back(Glib::Variant<int>::create(deviceCount));
  ... // pack additional input arguments
  Glib::VariantContainerBase args
      = Glib::VariantContainerBase::create_tuple(_args);

  // Synchronously get parameter values.
  Glib::VariantContainerBase response;
  if (training)
      response = proxy->call_sync("RequestTrainingWgsize", args);
  else
      response = proxy->call_sync("RequestWgsize", args);
  Glib::VariantIter iterator(response.get_child(0));
  Glib::Variant<int16_t> var;

  // Set workgroup size.
  iterator.next_value(var);
  local[0] = var.get();
  iterator.next_value(var);
  local[1] = var.get();
}
