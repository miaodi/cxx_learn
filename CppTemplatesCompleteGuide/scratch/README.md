# Scratch

This directory is a small playground for trying ideas while studying
*C++ Templates: The Complete Guide*.

Use `playground.cpp` for temporary experiments. Keep polished examples in the
chapter/topic directories such as `../basics/`.

## Run

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target cpp_templates_scratch
./build/CppTemplatesCompleteGuide/cpp_templates_scratch
```
