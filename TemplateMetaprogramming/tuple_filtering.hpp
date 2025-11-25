/**
 * @file tuple_filtering.hpp
 * @brief Tuple filtering and manipulation - the heart of launch_box selection
 * 
 * This demonstrates the advanced template metaprogramming used in gunrock's
 * launch_box to filter a parameter pack and select matching configurations.
 */

#pragma once
#include <tuple>
#include <type_traits>
#include <iostream>
#include <bitset>
#include <utility>

namespace template_tricks {

// ============================================================================
// 1. Basic Tuple Operations
// ============================================================================

inline void demo_basic_tuples() {
    std::cout << "\n=== Basic Tuple Operations ===\n\n";
    
    std::tuple<int, double, const char*> my_tuple{42, 3.14, "hello"};
    
    std::cout << "Element 0: " << std::get<0>(my_tuple) << "\n";
    std::cout << "Element 1: " << std::get<1>(my_tuple) << "\n";
    std::cout << "Element 2: " << std::get<2>(my_tuple) << "\n";
    std::cout << "Tuple size: " << std::tuple_size_v<decltype(my_tuple)> << "\n";
    
    // Type of element at index 1
    using SecondType = std::tuple_element_t<1, decltype(my_tuple)>;
    std::cout << "Type at index 1: " << (std::is_same_v<SecondType, double> ? "double" : "other") << "\n";
}

// ============================================================================
// 2. Tuple Concatenation
// ============================================================================

inline void demo_tuple_cat() {
    std::cout << "\n=== Tuple Concatenation ===\n\n";
    
    auto t1 = std::make_tuple(1, 2);
    auto t2 = std::make_tuple(3.14, "hello");
    auto t3 = std::make_tuple('x');
    
    // Concatenate all tuples
    auto combined = std::tuple_cat(t1, t2, t3);
    
    std::cout << "Combined tuple size: " << std::tuple_size_v<decltype(combined)> << "\n";
    std::cout << "Elements: " << std::get<0>(combined) << ", " 
              << std::get<1>(combined) << ", "
              << std::get<2>(combined) << ", "
              << std::get<3>(combined) << ", "
              << std::get<4>(combined) << "\n";
}

// ============================================================================
// 3. Conditional Tuple Building (Key Technique!)
// ============================================================================

/**
 * @brief Build a tuple containing only elements that satisfy a condition
 * 
 * This is similar to what match_launch_params_t does in launch_box!
 */
template <bool Condition, typename T>
using ConditionalTuple = std::conditional_t<
    Condition,
    std::tuple<T>,    // Include T if condition is true
    std::tuple<>      // Empty tuple if condition is false
>;

inline void demo_conditional_tuple() {
    std::cout << "\n=== Conditional Tuple Building ===\n\n";
    
    // Build tuple with only even numbers
    auto result = std::tuple_cat(
        ConditionalTuple<true, int>(std::make_tuple(2)),      // 2 is even
        ConditionalTuple<false, int>(),                        // 3 is odd - excluded
        ConditionalTuple<true, int>(std::make_tuple(4)),      // 4 is even
        ConditionalTuple<false, int>(),                        // 5 is odd - excluded
        ConditionalTuple<true, int>(std::make_tuple(6))       // 6 is even
    );
    
    std::cout << "Filtered tuple (only even): ";
    std::cout << std::get<0>(result) << ", " 
              << std::get<1>(result) << ", "
              << std::get<2>(result) << "\n";
    std::cout << "Size: " << std::tuple_size_v<decltype(result)> << " (3 elements)\n";
}

// ============================================================================
// 4. Parameter Pack Filtering (Launch Box Technique!)
// ============================================================================

/**
 * @brief Config with a "flag" that determines if it matches
 */
template <int flag_value, int data_value>
struct Config {
    static constexpr int flag = flag_value;
    static constexpr int data = data_value;
};

/**
 * @brief Filter configs by matching flag against target
 * 
 * This is EXACTLY how launch_box works!
 */
template <int target_flag, typename... Configs>
using FilterConfigs = decltype(std::tuple_cat(
    std::declval<std::conditional_t<
        (Configs::flag & target_flag) != 0,  // Does flag match?
        std::tuple<Configs>,                  // Yes: include it
        std::tuple<>                          // No: skip it
    >>()...                                   // Expand for all Configs
));

// Compile-time sanity checks for the filtering utility.
namespace detail {
using Flag1 = Config<0b0001, 100>;
using Flag2 = Config<0b0010, 200>;
using Flag3 = Config<0b0100, 300>;
using Flag4 = Config<0b1000, 400>;
using FlagFallback = Config<0b1111, 999>;

constexpr int kFilterTarget = 0b0010;
using FilteredFlags = FilterConfigs<
    kFilterTarget,
    Flag1,
    Flag2,
    Flag3,
    Flag4,
    FlagFallback
>;

static_assert(std::tuple_size_v<FilteredFlags> == 2,
              "Expected one exact match plus the fallback.");
using FirstFiltered = std::tuple_element_t<0, FilteredFlags>;
static_assert(std::is_same_v<FirstFiltered, Flag2>,
              "First filtered config should be the first matching flag.");
} // namespace detail

inline void demo_parameter_pack_filtering() {
    std::cout << "\n=== Parameter Pack Filtering (Launch Box Style!) ===\n\n";
    
    // Define some configs with different flags (like sm_75, sm_80, etc.)
    using Config1 = Config<0b0001, 100>;  // flag = 1
    using Config2 = Config<0b0010, 200>;  // flag = 2
    using Config3 = Config<0b0100, 300>;  // flag = 4
    using Config4 = Config<0b1000, 400>;  // flag = 8
    using ConfigFallback = Config<0b1111, 999>;  // flag = all bits (fallback)
    
    // Filter to find configs matching target flag 0b0010 (value 2)
    constexpr int target = 0b0010;
    using Filtered = FilterConfigs<target, Config1, Config2, Config3, Config4, ConfigFallback>;
    
    std::cout << "Total configs: 5\n";
    std::cout << "Target flag: 0b" << std::bitset<4>(target) << " (" << target << ")\n";
    std::cout << "Matching configs: " << std::tuple_size_v<Filtered> << "\n";
    
    // Extract first match
    using FirstMatch = std::tuple_element_t<0, Filtered>;
    std::cout << "First match data value: " << FirstMatch::data << "\n";
    
    // Demonstrate fallback behavior
    std::cout << "\nDemonstrating fallback:\n";
    std::cout << "ConfigFallback has flag 0b1111 (matches any target)\n";
    std::cout << "If no exact match exists, fallback will be selected\n";
}

// ============================================================================
// 5. Complete Launch Box Pattern
// ============================================================================

// Architecture flags (simulating sm_75, sm_80, etc.)
enum GPUArch : unsigned int {
    sm_60 = 1 << 0,   // 0b0001
    sm_70 = 1 << 1,   // 0b0010
    sm_75 = 1 << 2,   // 0b0100
    sm_80 = 1 << 3,   // 0b1000
    sm_86 = 1 << 4,   // 0b10000
    fallback_arch = ~0u    // All bits set
};

// Allow bitwise OR to combine flags
constexpr GPUArch operator|(GPUArch a, GPUArch b) {
    return static_cast<GPUArch>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

// Simulate compile-time target (like SM_TARGET macro)
// Allow overriding via compiler definition to mirror real-world usage.
#ifndef COMPILED_FOR_ARCH
#define COMPILED_FOR_ARCH sm_75
#endif

/**
 * @brief Launch parameters for specific architecture(s)
 */
template <GPUArch arch_flags, int block_size, int grid_size>
struct LaunchParams {
    static constexpr GPUArch flags = arch_flags;
    static constexpr int block_dim = block_size;
    static constexpr int grid_dim = grid_size;
};

/**
 * @brief Filter parameter pack to find matching architectures
 */
template <typename... Params>
using MatchParams = decltype(std::tuple_cat(
    std::declval<std::conditional_t<
        (bool)(Params::flags & COMPILED_FOR_ARCH),
        std::tuple<Params>,
        std::tuple<>
    >>()...
));

/**
 * @brief Select first matching params (or error if none found)
 */
template <typename... Params>
using SelectParams = std::conditional_t<
    (std::tuple_size_v<MatchParams<Params...>> == 0),
    void,  // Could be raise_error_t in real code
    std::tuple_element_t<0, MatchParams<Params...>>
>;

/**
 * @brief LaunchBox that selects params at compile time
 */
template <typename... Params>
struct LaunchBox {
    static_assert(
        std::tuple_size_v<MatchParams<Params...>> > 0,
        "No matching launch parameters for COMPILED_FOR_ARCH. "
        "Add a fallback_arch configuration."
    );

    using selected = SelectParams<Params...>;
    
    static constexpr int block_size = selected::block_dim;
    static constexpr int grid_size = selected::grid_dim;
};

inline void demo_complete_launch_box() {
    std::cout << "\n=== Complete Launch Box Pattern ===\n\n";
    
    // Define multiple launch configs for different architectures
    using Params_sm60 = LaunchParams<sm_60, 128, 64>;
    using Params_sm70 = LaunchParams<sm_70, 256, 128>;
    using Params_sm75 = LaunchParams<sm_75, 512, 256>;  // This will be selected!
    using Params_sm80 = LaunchParams<sm_80 | sm_86, 1024, 512>;
    using Params_fallback = LaunchParams<fallback_arch, 64, 32>;
    
    // Create launch box with all configs
    using MyLaunchBox = LaunchBox<
        Params_sm60,
        Params_sm70,
        Params_sm75,
        Params_sm80,
        Params_fallback
    >;
    
    std::cout << "Compiled for: sm_75\n";
    std::cout << "Selected block size: " << MyLaunchBox::block_size << "\n";
    std::cout << "Selected grid size: " << MyLaunchBox::grid_size << "\n";
    
    std::cout << "\nThis demonstrates the EXACT technique used in gunrock's launch_box!\n";
}

// ============================================================================
// 6. Understanding std::declval
// ============================================================================

inline void demo_declval() {
    std::cout << "\n=== Understanding std::declval ===\n\n";
    
    std::cout << "std::declval<T>() creates a 'fake' instance of T for type deduction\n";
    std::cout << "It's only valid in unevaluated contexts (decltype, sizeof, etc.)\n\n";
    
    // Example: Get return type without creating instance
    struct Foo {
        int compute(double x) { return static_cast<int>(x * 2); }
    };
    
    // We can deduce return type without constructing Foo!
    using ReturnType = decltype(std::declval<Foo>().compute(3.14));
    
    std::cout << "Return type of Foo::compute: " 
              << (std::is_same_v<ReturnType, int> ? "int" : "other") << "\n";
    
    std::cout << "\nIn tuple filtering, we use:\n";
    std::cout << "  decltype(std::tuple_cat(std::declval<...>()...))\n";
    std::cout << "To compute the result type without actually executing anything!\n";
}

// ============================================================================
// Demo Function
// ============================================================================

inline void demo_tuple_filtering() {
    std::cout << "\n=== Tuple Filtering and Manipulation ===\n";
    
    demo_basic_tuples();
    demo_tuple_cat();
    demo_conditional_tuple();
    demo_parameter_pack_filtering();
    demo_complete_launch_box();
    demo_declval();
    
    std::cout << "\n=== Key Insights ===\n";
    std::cout << "1. Tuples can be conditionally built and concatenated\n";
    std::cout << "2. Parameter packs can be filtered using conditional tuples\n";
    std::cout << "3. decltype + std::declval enables type computation without execution\n";
    std::cout << "4. This is how launch_box selects architecture-specific configs!\n";
}

} // namespace template_tricks
