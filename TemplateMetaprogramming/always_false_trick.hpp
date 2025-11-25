/**
 * @file always_false_trick.hpp
 * @brief Demonstration of the always_false trick for deferred static_assert
 * 
 * This trick is used when you want a static_assert inside a template that only
 * fails when the template is actually instantiated, not when it's first seen.
 */

#pragma once
#include <type_traits>
#include <iostream>

namespace template_tricks {

// ============================================================================
// 1. The Problem: Direct static_assert(false) doesn't work
// ============================================================================

// This would fail to compile immediately:
// template <typename T>
// struct DirectAssert {
//     static_assert(false, "This will fail even if never instantiated!");
// };

// ============================================================================
// 2. Solution 1: always_false - The Classic Trick
// ============================================================================

/**
 * @brief A template that always evaluates to false, but in a type-dependent way
 * 
 * The compiler can't optimize this away because it depends on the template parameter.
 * The compiler must wait until instantiation to evaluate it.
 */
template <typename T>
struct always_false {
    static constexpr bool value = false;
};

// Helper variable template (C++17)
template <typename T>
inline constexpr bool always_false_v = always_false<T>::value;

/**
 * @brief Example: Compile-time error with helpful message
 * 
 * This struct will only trigger a static_assert when you try to instantiate it.
 */
template <typename T>
struct UnsupportedType {
    static_assert(always_false<T>::value, 
                  "This type is not supported! Specialize this template.");
};

// ============================================================================
// 3. Solution 2: Using std::is_same for false condition
// ============================================================================

/**
 * @brief Alternative approach using std::is_same
 * 
 * Compare T with a type that will never match (like void when T is not void)
 */
template <typename T>
struct AlternativeAssert {
    // Only fails if T is not void, but waits for instantiation
    static_assert(!std::is_same_v<T, T> || std::is_same_v<T, void>, 
                  "Alternative deferred assert");
};

// ============================================================================
// 4. Practical Example: Type-based Dispatch with Error Fallback
// ============================================================================

/**
 * @brief Process different types with a compile-time error for unsupported ones
 */
template <typename T>
struct TypeProcessor {
    static void process() {
        static_assert(always_false<T>::value, 
                      "No processor defined for this type. "
                      "Please specialize TypeProcessor<T>");
    }
};

// Specialization for int
template <>
struct TypeProcessor<int> {
    static void process() {
        std::cout << "Processing int type\n";
    }
};

// Specialization for double
template <>
struct TypeProcessor<double> {
    static void process() {
        std::cout << "Processing double type\n";
    }
};

// Specialization for const char*
template <>
struct TypeProcessor<const char*> {
    static void process() {
        std::cout << "Processing string type\n";
    }
};

// ============================================================================
// 5. Advanced: Conditional Compile Error based on Type Traits
// ============================================================================

/**
 * @brief Only allow arithmetic types, fail at compile time for others
 */
template <typename T>
struct ArithmeticOnly {
    static_assert(std::is_arithmetic_v<T> || always_false_v<T>,
                  "This template only accepts arithmetic types (int, float, double, etc.)");
    
    static T compute(T a, T b) {
        return a + b;
    }
};

// ============================================================================
// 6. Conditional Instantiation Error
// ============================================================================

/**
 * @brief Template that errors only for specific conditions
 */
template <typename T, bool Condition>
struct ConditionalError {
    // Only errors when Condition is true AND template is instantiated
    static_assert(!Condition || always_false_v<T>,
                  "Condition violation detected!");
    
    using type = T;
};

// ============================================================================
// Demo Functions
// ============================================================================

inline void demo_always_false_trick() {
    std::cout << "\n=== Always False Trick Demo ===\n\n";
    
    // These will work fine
    std::cout << "1. Type processors for supported types:\n";
    TypeProcessor<int>::process();
    TypeProcessor<double>::process();
    TypeProcessor<const char*>::process();
    
    // This would cause a compile error:
    // TypeProcessor<std::string>::process();
    // Error: "No processor defined for this type"
    
    std::cout << "\n2. Arithmetic operations:\n";
    auto result1 = ArithmeticOnly<int>::compute(5, 10);
    std::cout << "  5 + 10 = " << result1 << "\n";
    
    auto result2 = ArithmeticOnly<double>::compute(3.14, 2.86);
    std::cout << "  3.14 + 2.86 = " << result2 << "\n";
    
    // This would cause a compile error:
    // auto result3 = ArithmeticOnly<std::string>::compute("a", "b");
    // Error: "This template only accepts arithmetic types"
    
    std::cout << "\n3. Conditional errors:\n";
    using NoError [[maybe_unused]] = ConditionalError<int, false>::type;  // OK
    std::cout << "  ConditionalError<int, false> compiles fine\n";
    
    // This would cause a compile error:
    // using Error = ConditionalError<int, true>::type;
    // Error: "Condition violation detected!"
    
    std::cout << "\n=== Key Insight ===\n";
    std::cout << "The always_false trick allows us to write static_assert that:\n";
    std::cout << "  - Only triggers when the template is instantiated\n";
    std::cout << "  - Provides helpful error messages\n";
    std::cout << "  - Enables compile-time type checking and validation\n";
}

} // namespace template_tricks
