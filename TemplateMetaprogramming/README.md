# Template Metaprogramming Examples

This project demonstrates advanced C++ template metaprogramming techniques extracted from the gunrock GPU graph library's `launch_box` implementation.

## Overview

The gunrock `launch_box` uses sophisticated template metaprogramming to select optimal GPU kernel launch parameters at compile time based on the target architecture. This project breaks down and explains each technique used.

## Techniques Demonstrated

### 1. Always False Trick (`always_false_trick.hpp`)
- Deferred `static_assert` that only fires on template instantiation
- Used for providing better compile-time error messages
- Essential for creating "poison pill" types that error gracefully

### 2. Type Selection (`type_selection.hpp`)
- `std::conditional` for compile-time type selection
- Template specialization for pattern matching on types
- Type traits for inspecting properties at compile time
- Architecture-based configuration selection

### 3. Tuple Filtering (`tuple_filtering.hpp`)
- **Core launch_box technique**: filtering parameter packs
- Using `std::tuple_cat` with conditional tuples
- `decltype` and `std::declval` for type computation
- Complete launch_box pattern implementation

### 4. SFINAE (`sfinae_tricks.hpp`)
- Substitution Failure Is Not An Error
- `std::enable_if` for conditional function overloads
- Detecting member functions and types
- Tag dispatching as an alternative

## Building and Running

```bash
cd /path/to/cxx_learn
mkdir -p build && cd build
cmake ..
make
./bin/template_tricks
```

## Key Insights

### How Launch Box Works

1. **Define configurations** for different GPU architectures:
   ```cpp
   launch_params_t<sm_75, dim3_t<256>, dim3_t<128>>
   launch_params_t<sm_80, dim3_t<512>, dim3_t<256>>
   launch_params_t<fallback, dim3_t<128>, dim3_t<64>>
   ```

2. **Filter at compile time** using tuple operations:
   ```cpp
   using Filtered = decltype(std::tuple_cat(
       std::conditional_t<matches_arch, std::tuple<Config>, std::tuple<>>()...
   ));
   ```

3. **Select first match** or trigger compile error:
   ```cpp
   using Selected = std::conditional_t<
       tuple_size == 0,
       raise_error_t,
       tuple_element_t<0, Filtered>
   >;
   ```

4. **Zero runtime overhead** - all selection happens at compile time!

## Real-World Application

In gunrock, this enables writing:

```cpp
using launch_t = launch_box_t<
    launch_params_t<sm_86 | sm_80, dim3_t<512>, dim3_t<256>>,
    launch_params_t<sm_75 | sm_70, dim3_t<256>, dim3_t<128>>,
    launch_params_t<fallback, dim3_t<128>, dim3_t<64>>
>;

launch_t launcher;
launcher.launch(context, my_kernel, args...);
```

The compiler automatically selects the right configuration based on `-DSM_TARGET=75` (or whatever architecture you're compiling for).

## Requirements

- C++17 or later
- CMake 3.18+
- Modern C++ compiler (GCC 7+, Clang 5+, MSVC 2017+)

## Learning Resources

- [C++ Templates: The Complete Guide](https://www.amazon.com/Templates-Complete-Guide-2nd/dp/0321714121)
- [Modern C++ Design](https://www.amazon.com/Modern-Design-Generic-Programming-Patterns/dp/0201704315)
- [cppreference.com - Template metaprogramming](https://en.cppreference.com/w/cpp/language/templates)

## Author

Extracted and documented from [gunrock](https://github.com/gunrock/gunrock) project.

## License

Educational purposes. See gunrock's original license for the source material.
