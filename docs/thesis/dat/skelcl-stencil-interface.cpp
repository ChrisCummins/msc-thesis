template <typename Tin, typename Tout>
class Stencil<Tout(Tin)> : public detail::Skeleton {
public:

  // Construct a stencil object.
  Stencil(const Source& source,
          const std::string& func,
          const StencilShape& shape,
          Padding padding, Tin paddingElement =
            static_cast<Tin>(NULL));

  // Execute stencil.
  template <typename... Args>
  Matrix<Tout> operator()(const Matrix<Tin>& input,
                          Args&&... args) const;

  ... // Further implementation
}
