# Basics

Small examples from the book's basics chapter.

| Topic | Source | Target |
| --- | --- | --- |
| Ordinary lookup in a function template body | `max4.cpp` | `cpp_templates_basics_max4` |
| The four fold expression forms | `fold_expressions.cpp` | `cpp_templates_basics_fold_expressions` |
| Index sequences with a local tuple and `get<I>()` | `index_sequence_get.cpp` | `cpp_templates_basics_index_sequence_get` |

## Knowledge Points

- If a non-template overload and a function template are both visible and equally
  good matches, overload resolution prefers the non-template overload.
- Ordinary unqualified lookup inside a template body only sees declarations that
  are already visible at the template definition.
- Fold expressions have four forms: right fold without init, left fold without
  init, right fold with init, and left fold with init.
- Fold direction matters for non-associative operators, and folds with init
  provide an initial value that can make empty packs well-defined.
- A minimal tuple can be represented as `Tuple<Head, Tail...>`: store `head`
  directly and store the remaining values recursively in `tail`.
- A teaching `get<I>()` recursively peels `tail` until `I == 0`, then returns
  `head`.
- `get<Indices>(tuple)...` expands a compile-time index list into selected tuple
  elements, such as `get<2>(tuple), get<1>(tuple), get<3>(tuple)`.
