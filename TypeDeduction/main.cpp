#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

std::string demangle(const char* name) {
#ifdef __GNUG__
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return status == 0 && res ? res.get() : name;
#else
  return name;
#endif
}

template <typename T>
std::string type_name() {
  using Raw = std::remove_reference_t<T>;
  using Base = std::remove_cv_t<Raw>;

  std::string name = demangle(typeid(Base).name());
  if (std::is_const_v<Raw>) {
    name = "const " + name;
  }
  if (std::is_volatile_v<Raw>) {
    name = "volatile " + name;
  }
  if (std::is_lvalue_reference_v<T>) {
    name += "&";
  } else if (std::is_rvalue_reference_v<T>) {
    name += "&&";
  }
  return name;
}

template <typename T>
void print_type(const std::string& label) {
  std::cout << "  " << label << " -> " << type_name<T>() << '\n';
}

struct Widget {
  int field = 99;
  double value() const { return 3.14; }
};

int& identity_ref(int& v) { return v; }
decltype(auto) forward_field(Widget& w) { return (w.field); }
decltype(auto) value_field(Widget w) { return w.field; }

template <typename T>
void capture_by_value(T param, const std::string& label) {
  std::cout << "  capture_by_value(" << label << "): T = " << type_name<T>()
            << ", param = " << type_name<decltype(param)>() << '\n';
}

template <typename T>
void capture_by_ref(T& param, const std::string& label) {
  std::cout << "  capture_by_ref(" << label << "): T = " << type_name<T>()
            << ", param = " << type_name<decltype(param)>() << '\n';
}

template <typename T>
void capture_forward(T&& param, const std::string& label) {
  std::cout << "  capture_forward(" << label
            << "): T = " << type_name<T>()
            << ", param = " << type_name<decltype(param)>() << '\n';
}

void auto_deduction_examples() {
  std::cout << "=== auto deduction ===\n";

  auto i = 42;
  print_type<decltype(i)>("auto i = 42");

  const int ci = 7;
  auto copy_ci = ci;
  print_type<decltype(copy_ci)>("auto copy_ci = ci (const int)");
  const auto const_ci = ci;
  print_type<decltype(const_ci)>("const auto const_ci = ci");
  auto& ref_ci = ci;
  print_type<decltype(ref_ci)>("auto& ref_ci = ci");
  auto&& fwd_ci = ci;
  print_type<decltype(fwd_ci)>("auto&& fwd_ci = ci");
  auto&& temp = 5;
  print_type<decltype(temp)>("auto&& temp = 5");

  int arr[3] = {1, 2, 3};
  auto arr_copy = arr;
  print_type<decltype(arr_copy)>("auto arr_copy = arr (int[3])");
  auto& arr_ref = arr;
  print_type<decltype(arr_ref)>("auto& arr_ref = arr (int[3])");

  auto init_list = {1, 2, 3};
  print_type<decltype(init_list)>("auto init_list = {1, 2, 3}");

  std::cout << '\n';
}

void template_deduction_examples() {
  std::cout << "=== template parameter deduction ===\n";

  int x = 10;
  const int cx = x;
  capture_by_value(x, "x (int lvalue)");
  capture_by_value(cx, "cx (const int lvalue)");
  capture_by_value(5, "5 (int rvalue)");

  capture_by_ref(x, "x (int lvalue)");
  capture_by_ref(cx, "cx (const int lvalue)");

  capture_forward(x, "x (int lvalue)");
  capture_forward(cx, "cx (const int lvalue)");
  capture_forward(5, "5 (int rvalue)");

  double d = 4.2;
  capture_forward(std::move(d), "std::move(d) (double rvalue)");

  int arr[2] = {0, 1};
  capture_by_value(arr, "arr (int[2])");
  capture_by_ref(arr, "arr (int[2] lvalue)");
  capture_forward(arr, "arr (int[2] lvalue)");

  std::cout << '\n';
}

void decltype_examples() {
  std::cout << "=== decltype / decltype(auto) ===\n";

  int x = 0;
  const int cx = x;

  print_type<decltype(x)>("decltype(x) with int x");
  print_type<decltype((x))>("decltype((x)) (lvalue)");
  print_type<decltype(cx)>("decltype(cx) with const int cx");
  print_type<decltype((cx))>("decltype((cx)) (const lvalue)");
  print_type<decltype(std::move(x))>("decltype(std::move(x))");

  Widget w{};
  print_type<decltype(w.field)>("decltype(w.field) (non-ref data member)");
  print_type<decltype((w.field))>("decltype((w.field)) (data member lvalue)");
  print_type<decltype(w.value())>("decltype(w.value()) (prvalue)");
  print_type<decltype(forward_field(w))>("decltype(forward_field(w))");
  print_type<decltype(value_field(w))>("decltype(value_field(w))");

  int y = 1;
  print_type<decltype(identity_ref(y))>("decltype(identity_ref(y)) returns reference");

  std::cout << '\n';
}

int main() {
  auto_deduction_examples();
  template_deduction_examples();
  decltype_examples();
  return 0;
}
