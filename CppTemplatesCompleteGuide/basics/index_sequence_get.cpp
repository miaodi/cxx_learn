#include <cstddef>
#include <iostream>

template <typename... Ts>
struct Tuple;

template <>
struct Tuple<> {};

template <typename Head, typename... Tail>
struct Tuple<Head, Tail...> {
  Head head;
  Tuple<Tail...> tail;

  constexpr Tuple(Head head_value, Tail... tail_values)
      : head(head_value), tail(tail_values...) {}
};

template <std::size_t I, typename Head, typename... Tail>
constexpr decltype(auto) get(Tuple<Head, Tail...>& tuple) {
  static_assert(I < 1 + sizeof...(Tail));

  if constexpr (I == 0) {
    return (tuple.head);
  } else {
    return get<I - 1>(tuple.tail);
  }
}

template <std::size_t I, typename Head, typename... Tail>
constexpr decltype(auto) get(Tuple<Head, Tail...> const& tuple) {
  static_assert(I < 1 + sizeof...(Tail));

  if constexpr (I == 0) {
    return (tuple.head);
  } else {
    return get<I - 1>(tuple.tail);
  }
}

template <std::size_t... Indices>
struct IndexSequence {};

void print_values() {
  std::cout << '\n';
}

template <typename T, typename... Rest>
void print_values(T const& value, Rest const&... rest) {
  std::cout << value;

  if constexpr (sizeof...(Rest) == 0) {
    std::cout << '\n';
  } else {
    std::cout << ' ';
    print_values(rest...);
  }
}

template <typename TupleType, std::size_t... Indices>
void print_selected(TupleType const& tuple, IndexSequence<Indices...>) {
  print_values(get<Indices>(tuple)...);
}

int main() {
  constexpr Tuple<int, int, int, int> tuple{10, 20, 30, 40};

  std::cout << "Custom Tuple/get plus IndexSequence\n\n";

  std::cout << "Tuple<int,int,int,int>{10, 20, 30, 40}\n";
  std::cout << "  get<0>(tuple) -> " << get<0>(tuple) << "\n";
  std::cout << "  get<1>(tuple) -> " << get<1>(tuple) << "\n";
  std::cout << "  get<2>(tuple) -> " << get<2>(tuple) << "\n";
  std::cout << "  get<3>(tuple) -> " << get<3>(tuple) << "\n\n";

  std::cout << "IndexSequence<2, 1, 3> selects tuple elements by index.\n";
  std::cout << "print_selected(tuple, IndexSequence<2, 1, 3>{}) expands as:\n";
  std::cout << "  print_values(get<2>(tuple), get<1>(tuple), get<3>(tuple))\n";
  std::cout << "Selected values: ";
  print_selected(tuple, IndexSequence<2, 1, 3>{});

  std::cout << "\nKnowledge point 1: Tuple<Head, Tail...> stores one value and "
               "recursively stores the rest in tail.\n";
  std::cout << "Knowledge point 2: get<I>() peels tail until I == 0, then "
               "returns head.\n";
  std::cout << "Knowledge point 3: get<Indices>(tuple)... expands a "
               "compile-time index list into selected tuple elements.\n";
  std::cout << "Caveat: real std::tuple/std::get handles references, const, "
               "rvalues, empty-base optimization, and many constructor cases.\n";
}
