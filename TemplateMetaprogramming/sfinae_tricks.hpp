/**
 * @file sfinae_tricks.hpp
 * @brief SFINAE (Substitution Failure Is Not An Error) techniques
 * 
 * SFINAE allows function overload resolution based on whether certain
 * expressions are valid for a given type. This is fundamental to many
 * modern C++ template tricks.
 */

#pragma once
#include <type_traits>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

namespace template_tricks {

// ============================================================================
// 1. Basic SFINAE with std::enable_if
// ============================================================================

/**
 * @brief Function enabled only for integral types
 */
template <typename T>
std::enable_if_t<std::is_integral_v<T>, void>
process(T value) {
    std::cout << "Processing integral: " << value << "\n";
}

/**
 * @brief Function enabled only for floating-point types
 */
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, void>
process(T value) {
    std::cout << "Processing floating-point: " << value << "\n";
}

/**
 * @brief Function enabled only for pointer types
 */
template <typename T>
std::enable_if_t<std::is_pointer_v<T>, void>
process(T value) {
    std::cout << "Processing pointer at address: " << static_cast<void*>(value) << "\n";
}

inline void demo_enable_if() {
    std::cout << "\n=== std::enable_if Demo ===\n\n";
    
    process(42);          // Calls integral version
    process(3.14);        // Calls floating-point version
    int x = 10;
    process(&x);          // Calls pointer version
}

// ============================================================================
// 2. Detecting Member Functions
// ============================================================================

/**
 * @brief Check if type T has a size() member function
 */
template <typename T, typename = void>
struct has_size_method : std::false_type {};

template <typename T>
struct has_size_method<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_size_method_v = has_size_method<T>::value;

/**
 * @brief Use size() if available, otherwise estimate
 */
template <typename T>
std::enable_if_t<has_size_method_v<T>, std::size_t>
get_size(const T& container) {
    return container.size();
}

template <typename T>
std::enable_if_t<!has_size_method_v<T>, std::size_t>
get_size(const T&) {
    return 1;  // For non-containers, size is 1
}

inline void demo_member_detection() {
    std::cout << "\n=== Member Function Detection ===\n\n";
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::string str = "hello";
    int num = 42;
    
    std::cout << "vector<int> has size(): " << has_size_method_v<std::vector<int>> << "\n";
    std::cout << "  size = " << get_size(vec) << "\n";
    
    std::cout << "string has size(): " << has_size_method_v<std::string> << "\n";
    std::cout << "  size = " << get_size(str) << "\n";
    
    std::cout << "int has size(): " << has_size_method_v<int> << "\n";
    std::cout << "  size = " << get_size(num) << " (estimated)\n";
}

// ============================================================================
// 3. Detecting Member Types
// ============================================================================

/**
 * @brief Check if type T has a value_type member typedef
 */
template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>>
    : std::true_type {};

/**
 * @brief Check if type T has begin() and end() (is iterable)
 */
template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

template <typename T>
inline constexpr bool is_iterable_v = is_iterable<T>::value;

/**
 * @brief Print container if iterable, otherwise print single value
 */
template <typename T>
std::enable_if_t<is_iterable_v<T>, void>
print_value(const T& container) {
    std::cout << "Container: [";
    bool first = true;
    for (const auto& elem : container) {
        if (!first) std::cout << ", ";
        std::cout << elem;
        first = false;
    }
    std::cout << "]\n";
}

template <typename T>
std::enable_if_t<!is_iterable_v<T>, void>
print_value(const T& value) {
    std::cout << "Single value: " << value << "\n";
}

inline void demo_type_detection() {
    std::cout << "\n=== Type Detection ===\n\n";
    
    std::vector<int> vec = {10, 20, 30};
    std::cout << "vector is iterable: " << is_iterable_v<std::vector<int>> << "\n";
    print_value(vec);
    
    int num = 42;
    std::cout << "\nint is iterable: " << is_iterable_v<int> << "\n";
    print_value(num);
}

// ============================================================================
// 4. Return Type SFINAE
// ============================================================================

/**
 * @brief Function that returns different types based on input
 */
template <typename T>
auto compute(T value) -> std::enable_if_t<std::is_integral_v<T>, double> {
    return value * 2.0;
}

template <typename T>
auto compute(T value) -> std::enable_if_t<std::is_floating_point_v<T>, int> {
    return static_cast<int>(value);
}

inline void demo_return_type_sfinae() {
    std::cout << "\n=== Return Type SFINAE ===\n\n";
    
    auto result1 = compute(42);      // int -> double
    auto result2 = compute(3.14);    // double -> int
    
    std::cout << "compute(42) returns: " << result1 
              << " (type: " << (std::is_same_v<decltype(result1), double> ? "double" : "other") << ")\n";
    std::cout << "compute(3.14) returns: " << result2 
              << " (type: " << (std::is_same_v<decltype(result2), int> ? "int" : "other") << ")\n";
}

// ============================================================================
// 5. Concept-like Constraints (C++17 style)
// ============================================================================

/**
 * @brief Require that T is arithmetic and U is convertible to T
 */
template <typename T, typename U>
using require_arithmetic_convertible = std::enable_if_t<
    std::is_arithmetic_v<T> && std::is_convertible_v<U, T>,
    int
>;

/**
 * @brief Safe addition with compile-time type checking
 */
template <typename T, typename U, require_arithmetic_convertible<T, U> = 0>
T safe_add(T a, U b) {
    return a + static_cast<T>(b);
}

inline void demo_concept_like() {
    std::cout << "\n=== Concept-like Constraints ===\n\n";
    
    auto result1 = safe_add(10, 5);           // int + int
    auto result2 = safe_add(3.14, 2);         // double + int
    auto result3 = safe_add(1.5f, 2.5f);      // float + float
    
    std::cout << "safe_add(10, 5) = " << result1 << "\n";
    std::cout << "safe_add(3.14, 2) = " << result2 << "\n";
    std::cout << "safe_add(1.5f, 2.5f) = " << result3 << "\n";
    
    // This would fail to compile:
    // safe_add("hello", "world");  // strings are not arithmetic
}

// ============================================================================
// 6. Tag Dispatching (Alternative to SFINAE)
// ============================================================================

// Tag types for different categories
struct integral_tag {};
struct floating_tag {};
struct other_tag {};

/**
 * @brief Select tag based on type properties
 */
template <typename T>
using type_category = std::conditional_t<
    std::is_integral_v<T>,
    integral_tag,
    std::conditional_t<
        std::is_floating_point_v<T>,
        floating_tag,
        other_tag
    >
>;

// Implementation for each tag
template <typename T>
void process_impl(T value, integral_tag) {
    std::cout << "  Processing as integral: " << value << "\n";
}

template <typename T>
void process_impl(T value, floating_tag) {
    std::cout << "  Processing as floating: " << value << "\n";
}

template <typename T>
void process_impl(T value, other_tag) {
    std::cout << "  Processing as other type\n";
    (void)value;
}

// Public interface
template <typename T>
void process_with_tags(T value) {
    process_impl(value, type_category<T>{});
}

inline void demo_tag_dispatching() {
    std::cout << "\n=== Tag Dispatching ===\n\n";
    
    std::cout << "42:\n";
    process_with_tags(42);
    
    std::cout << "3.14:\n";
    process_with_tags(3.14);
    
    std::cout << "\"hello\":\n";
    process_with_tags("hello");
}

// ============================================================================
// Demo Function
// ============================================================================

inline void demo_sfinae_tricks() {
    std::cout << "\n=== SFINAE Techniques Demo ===\n";
    
    demo_enable_if();
    demo_member_detection();
    demo_type_detection();
    demo_return_type_sfinae();
    demo_concept_like();
    demo_tag_dispatching();
    
    std::cout << "\n=== Key Insights ===\n";
    std::cout << "1. SFINAE enables/disables functions based on type properties\n";
    std::cout << "2. Can detect member functions, types, and other features\n";
    std::cout << "3. std::void_t and decltype are essential tools\n";
    std::cout << "4. Tag dispatching is a cleaner alternative for simple cases\n";
    std::cout << "5. These techniques enable powerful compile-time polymorphism\n";
}

} // namespace template_tricks
