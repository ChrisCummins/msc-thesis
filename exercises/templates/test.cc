// A simple test program for checking template instantiation.

template<typename T>
T add(T a, T b) {
  return a + b;
}

int main() {
  float x = add(0.0, 0.0);                // Float
  return    add(static_cast<int>(x), 0);  // Integer
}
