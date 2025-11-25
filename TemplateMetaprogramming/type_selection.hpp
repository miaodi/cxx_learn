/**
 * @file type_selection.hpp
 * @brief Compile-time type selection using std::conditional and template specialization
 * 
 * This demonstrates how to select types at compile time based on conditions,
 * similar to how launch_box selects parameters based on GPU architecture.
 */

#pragma once
#include <cstdint>
#include <type_traits>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>

namespace template_tricks {

// ============================================================================
// 1. Basic std::conditional - Like a compile-time if-else for types
// ============================================================================

/**
 * @brief Select between two types based on a boolean condition
 */
template <typename T>
using SelectStorageType = std::conditional_t<
    sizeof(T) <= 8,    // Condition: small enough for direct storage
    T,                 // True: use T directly
    T*                 // False: use pointer to T
>;

// Example usage
inline void demo_conditional() {
    std::cout << "\n=== std::conditional Demo ===\n\n";
    
    using SmallType = SelectStorageType<int>;        // sizeof(int) = 4, stores int
    using LargeType = SelectStorageType<std::string>; // sizeof(string) > 8, stores string*
    
    std::cout << "sizeof(int) = " << sizeof(int) << " -> ";
    std::cout << (std::is_pointer_v<SmallType> ? "pointer" : "value") << "\n";
    
    std::cout << "sizeof(string) = " << sizeof(std::string) << " -> ";
    std::cout << (std::is_pointer_v<LargeType> ? "pointer" : "value") << "\n";
}

/**
 * @brief Demonstrate using SelectStorageType in a small/large slot
 */
template <typename T>
struct StorageSlot {
    SelectStorageType<T> storage{};
    std::unique_ptr<T> owner{};  // only used when SelectStorageType is a pointer

    explicit StorageSlot(const T& value) {
        if constexpr (std::is_pointer_v<SelectStorageType<T>>) {
            owner = std::make_unique<T>(value);
            storage = owner.get();
        } else {
            storage = value;
        }
    }

    const T& get() const {
        if constexpr (std::is_pointer_v<SelectStorageType<T>>) {
            return *storage;
        } else {
            return storage;
        }
    }
};

inline void demo_storage_slot() {
    std::cout << "\n=== SelectStorageType In Practice ===\n\n";

    StorageSlot<int> small{42};
    StorageSlot<std::string> large{std::string("expensive copy avoided")};

    std::cout << "Small stored inline? "
              << (!std::is_pointer_v<SelectStorageType<int>>) << "\n";
    std::cout << "Large stored via pointer? "
              << (std::is_pointer_v<SelectStorageType<std::string>>) << "\n";
    std::cout << "Values: " << small.get() << " / " << large.get() << "\n";
}

// ============================================================================
// 2. Chained Conditionals - Multiple conditions
// ============================================================================

/**
 * @brief Select optimal integer type based on required range
 */
template <int MaxValue>
using OptimalInt = std::conditional_t<
    (MaxValue <= 127),
    int8_t,                     // -128 to 127
    std::conditional_t<
        (MaxValue <= 32767),
        int16_t,                // -32768 to 32767
        std::conditional_t<
            (MaxValue <= 2147483647),
            int32_t,            // -2^31 to 2^31-1
            int64_t             // -2^63 to 2^63-1
        >
    >
>;

inline void demo_chained_conditional() {
    std::cout << "\n=== Chained Conditional Demo ===\n\n";
    
    using Tiny = OptimalInt<100>;
    using Small = OptimalInt<1000>;
    using Medium = OptimalInt<100000>;
    using Large = OptimalInt<2147483647>;  // Max int32_t
    
    std::cout << "OptimalInt<100>: " << sizeof(Tiny) << " bytes (int8_t)\n";
    std::cout << "OptimalInt<1000>: " << sizeof(Small) << " bytes (int16_t)\n";
    std::cout << "OptimalInt<100000>: " << sizeof(Medium) << " bytes (int32_t)\n";
    std::cout << "OptimalInt<2147483647>: " << sizeof(Large) << " bytes (int32_t)\n";
}

// ============================================================================
// 3. Template Specialization - Pattern Matching on Types
// ============================================================================

/**
 * @brief Primary template (general case)
 */
template <typename T>
struct TypeInfo {
    static const char* name() { return "Unknown Type"; }
    static const char* category() { return "other"; }
};

// Specialization for int
template <>
struct TypeInfo<int> {
    static const char* name() { return "int"; }
    static const char* category() { return "integral"; }
};

// Specialization for float
template <>
struct TypeInfo<float> {
    static const char* name() { return "float"; }
    static const char* category() { return "floating-point"; }
};

// Specialization for any pointer type
template <typename T>
struct TypeInfo<T*> {
    static const char* name() { return "pointer"; }
    static const char* category() { return "pointer"; }
};

// Specialization for any array type
template <typename T, std::size_t N>
struct TypeInfo<T[N]> {
    static const char* name() { return "array"; }
    static const char* category() { return "array"; }
};

inline void demo_specialization() {
    std::cout << "\n=== Template Specialization Demo ===\n\n";
    
    std::cout << "TypeInfo<int>: " << TypeInfo<int>::name() 
              << " (" << TypeInfo<int>::category() << ")\n";
    
    std::cout << "TypeInfo<float>: " << TypeInfo<float>::name() 
              << " (" << TypeInfo<float>::category() << ")\n";
    
    std::cout << "TypeInfo<int*>: " << TypeInfo<int*>::name() 
              << " (" << TypeInfo<int*>::category() << ")\n";
    
    std::cout << "TypeInfo<double[10]>: " << TypeInfo<double[10]>::name() 
              << " (" << TypeInfo<double[10]>::category() << ")\n";
    
    std::cout << "TypeInfo<std::string>: " << TypeInfo<std::string>::name() 
              << " (" << TypeInfo<std::string>::category() << ")\n";
}

// ============================================================================
// 4. SFINAE - Substitution Failure Is Not An Error
// ============================================================================

/**
 * @brief Enable function only for types with begin() method (like containers)
 */
template <typename T>
auto print_size(const T& container) -> decltype(container.begin(), void()) {
    std::cout << "Container size: " << container.size() << "\n";
}

/**
 * @brief Enable function for types without begin() method
 */
template <typename T>
auto print_size(const T& value, int = 0) -> std::enable_if_t<!std::is_class_v<T>> {
    std::cout << "Single value: " << value << "\n";
}

// ============================================================================
// 5. Simplified Architecture Selection Example
// ============================================================================

inline void demo_architecture_selection() {
    std::cout << "\n=== Architecture Selection Demo ===\n\n";
    
    // Simplified demonstration of architecture-based config selection
    // (The full tuple_cat approach is shown in tuple_filtering.hpp)
    
    std::cout << "In a real launch_box:\n";
    std::cout << "1. Define configs for different architectures (sm_60, sm_75, sm_80, etc.)\n";
    std::cout << "2. At compile time, filter configs matching SM_TARGET\n";
    std::cout << "3. Select first match or use fallback\n";
    std::cout << "4. Extract parameters (block size, grid size, shared memory)\n\n";
    
    std::cout << "Example: Compiling for sm_75 selects config with:\n";
    std::cout << "  Cache line size: 64 bytes\n";
    std::cout << "  Data alignment: 8 bytes\n";
    std::cout << "\nSee tuple_filtering.hpp for complete implementation!\n";
}

// ============================================================================
// 6. Type Traits - Compile-time Type Inspection
// ============================================================================

template <typename T>
void inspect_type() {
    std::cout << "\nType inspection for " << typeid(T).name() << ":\n";
    std::cout << "  is_integral: " << std::is_integral_v<T> << "\n";
    std::cout << "  is_floating_point: " << std::is_floating_point_v<T> << "\n";
    std::cout << "  is_pointer: " << std::is_pointer_v<T> << "\n";
    std::cout << "  is_const: " << std::is_const_v<T> << "\n";
    std::cout << "  size: " << sizeof(T) << " bytes\n";
}

inline void demo_type_traits() {
    std::cout << "\n=== Type Traits Demo ===\n";
    inspect_type<int>();
    inspect_type<float>();
    inspect_type<int*>();
    inspect_type<const double>();
}

// ============================================================================
// Demo Function
// ============================================================================

inline void demo_type_selection() {
    std::cout << "\n=== Type Selection Techniques Demo ===\n";
    
    demo_conditional();
    demo_storage_slot();
    demo_chained_conditional();
    demo_specialization();
    demo_architecture_selection();
    demo_type_traits();
    
    std::cout << "\n=== Key Insights ===\n";
    std::cout << "1. std::conditional: compile-time if-else for types\n";
    std::cout << "2. Template specialization: pattern matching on types\n";
    std::cout << "3. Type traits: inspect properties at compile time\n";
    std::cout << "4. Combine these for powerful compile-time dispatching\n";
}

} // namespace template_tricks
