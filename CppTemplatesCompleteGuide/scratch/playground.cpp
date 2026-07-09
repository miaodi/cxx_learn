#include <iostream>
#include <type_traits>

template<typename T>
void print(T value) {
  std::cout << value << std::endl;
}

template<typename T, typename... Args>
void print(T t, Args... args){
  std::cout << t << std::endl;
  print(args...);
}

template <std::size_t... Args>
struct IndexSequence {};


template<typename T, std::size_t... Args>
void printIndexSequence(T t, IndexSequence<Args...>) {
  print(std::get<Args>(t)...);
}


int main() {
  IndexSequence<2, 1, 3> seq;
  auto tuple = std::make_tuple(10, 20, 30, 40);
  printIndexSequence(tuple, seq);
  return 0;
}
