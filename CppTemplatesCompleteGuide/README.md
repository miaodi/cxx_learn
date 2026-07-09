# C++ Templates: The Complete Guide

## Concept

This directory is for book-guided study of C++ templates, generic programming,
and compile-time techniques from *C++ Templates: The Complete Guide*.

Use this as a structured learning area rather than a reusable template library.
Each example should isolate one template rule, idiom, or design tradeoff.

## Layout

- `basics/`: examples that track the book's basics chapter.
- `notes/`: chapter notes, summaries, and focused explanations.
- `examples/`: small C++20 examples that make one template mechanism visible.
- `scratch/`: editable playground space for temporary experiments.

## Source Reference

The book's companion source is available at
[mpoullet/tmplbook](https://github.com/mpoullet/tmplbook/tree/master). Check it
when adding notes or examples that should track the book's original code.

## What To Run

Configure once from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build and run one basics example:

```sh
cmake --build build --target cpp_templates_basics_max4
./build/CppTemplatesCompleteGuide/cpp_templates_basics_max4

cmake --build build --target cpp_templates_basics_fold_expressions
./build/CppTemplatesCompleteGuide/cpp_templates_basics_fold_expressions

cmake --build build --target cpp_templates_basics_index_sequence_get
./build/CppTemplatesCompleteGuide/cpp_templates_basics_index_sequence_get

cmake --build build --target cpp_templates_scratch
./build/CppTemplatesCompleteGuide/cpp_templates_scratch
```

## Example Map

| Topic | Source | Target |
| --- | --- | --- |
| Non-template overload declared before/after a three-argument function template | `basics/max4.cpp` | `cpp_templates_basics_max4` |
| Four fold expression forms | `basics/fold_expressions.cpp` | `cpp_templates_basics_fold_expressions` |
| Index sequences with a local tuple and `get<I>()` | `basics/index_sequence_get.cpp` | `cpp_templates_basics_index_sequence_get` |

## Learning Map

Good first topics for this directory:

- Function templates, class templates, and non-type template parameters.
- Template argument deduction and overload resolution.
- Specialization, partial specialization, and variable templates.
- Dependent names, two-phase lookup, and `typename` / `template` disambiguators.
- SFINAE, constraints, concepts, and overload control.
- Traits, type transformations, and policy-based design.
- Variadic templates, fold expressions, and compile-time lists.
- Instantiation model, compile-time cost, and diagnostics.

## Caveats

Template examples are especially sensitive to compiler diagnostics and language
standard mode. Keep examples C++20 unless a topic specifically needs an older or
newer rule, and note compiler-specific behavior when diagnostics differ.
