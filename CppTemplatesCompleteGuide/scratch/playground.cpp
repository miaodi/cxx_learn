#include "traits.hpp"
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

template <typename Derived>
class CounterBase {
public:
  using DerivedType = Derived;
  CounterBase() { ++count; }
  CounterBase(const CounterBase &) { ++count; }
  ~CounterBase() { --count; }
  static int getCount() { return count; }
  static int count;
};

class MyClass : public CounterBase<MyClass> {
public:
  MyClass() = default;
  MyClass(const MyClass &) = default;
  ~MyClass() = default;
};

template <typename Derived>
int CounterBase<Derived>::count = 0;

template <typename Derived>
class ShapeBase {
public:
  using DerivedType = Derived;
  void draw() { static_cast<Derived *>(this)->drawImpl(); }
};

class Circle : public ShapeBase<Circle> {
public:
  void drawImpl() { std::cout << "Drawing Circle" << std::endl; }
};

class Square : public ShapeBase<Square> {
public:
  void drawImpl() { std::cout << "Drawing Square" << std::endl; }
};

template <typename T>
void drawShape(ShapeBase<T> &shape) {
  shape.draw();
}

void drawShapes() {
  using ShapeVariant = std::variant<Circle, Square>;
  std::vector<ShapeVariant> shapes;
  shapes.emplace_back(Circle());
  shapes.emplace_back(Square());

  for (auto &shape : shapes) {
    std::visit([](auto &s) { s.draw(); }, shape);
  }
}

int main() {
  MyClass obj1;
  MyClass obj2 = obj1;
  std::cout << "Current count: " << MyClass::getCount() << std::endl;
  auto obj3 = std::make_unique<MyClass>();
  std::cout << "Current count: " << MyClass::getCount() << std::endl;

  obj3.reset();
  std::cout << "Current count: " << MyClass::getCount() << std::endl;

  drawShapes();
  return 0;
}
