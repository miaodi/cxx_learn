/**
 * @file main.cpp
 * @brief Template Metaprogramming Examples - Main Entry Point
 * 
 * This project demonstrates advanced C++ template metaprogramming techniques
 * extracted from gunrock's launch_box implementation, including:
 * 
 * 1. always_false trick - Deferred static_assert for better error messages
 * 2. Type selection - Compile-time type dispatch and conditional types
 * 3. Tuple filtering - Parameter pack filtering using tuple operations
 * 4. SFINAE - Substitution Failure Is Not An Error techniques
 * 
 * Build and run:
 *   mkdir -p build && cd build
 *   cmake ..
 *   make
 *   ./bin/template_tricks
 */

#include <iostream>
#include "always_false_trick.hpp"
#include "type_selection.hpp"
#include "tuple_filtering.hpp"
#include "sfinae_tricks.hpp"

void print_header(const std::string& title) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << title;
    for (size_t i = title.length(); i < 57; ++i) std::cout << " ";
    std::cout << " ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
}

void print_section(const std::string& section) {
    std::cout << "\n";
    std::cout << "┌───────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << section;
    for (size_t i = section.length(); i < 57; ++i) std::cout << " ";
    std::cout << " │\n";
    std::cout << "└───────────────────────────────────────────────────────────┘\n";
}

int main() {
    print_header("C++ Template Metaprogramming Techniques");
    
    std::cout << "\n";
    std::cout << "This demo shows the techniques used in gunrock's launch_box to\n";
    std::cout << "select GPU kernel launch parameters at compile time based on\n";
    std::cout << "the target architecture.\n";
    
    // 1. Always False Trick
    print_section("1. Always False Trick");
    std::cout << "\nThe always_false trick enables deferred static_assert that only\n";
    std::cout << "fires when a template is actually instantiated, allowing better\n";
    std::cout << "compile-time error messages.\n";
    template_tricks::demo_always_false_trick();
    
    // 2. Type Selection
    print_section("2. Type Selection Techniques");
    std::cout << "\nCompile-time type selection using std::conditional, template\n";
    std::cout << "specialization, and type traits for architecture-aware dispatch.\n";
    template_tricks::demo_type_selection();
    
    // 3. Tuple Filtering
    print_section("3. Tuple Filtering and Parameter Pack Manipulation");
    std::cout << "\nThe core technique behind launch_box: filtering a parameter pack\n";
    std::cout << "to select matching configurations using tuple operations.\n";
    template_tricks::demo_tuple_filtering();
    
    // 4. SFINAE
    print_section("4. SFINAE (Substitution Failure Is Not An Error)");
    std::cout << "\nSFINAE enables compile-time function selection based on type\n";
    std::cout << "properties, member detection, and constraints.\n";
    template_tricks::demo_sfinae_tricks();
    
    // Summary
    print_header("Summary: How Launch Box Works");
    std::cout << "\n";
    std::cout << "The gunrock launch_box combines these techniques:\n\n";
    std::cout << "1. Define multiple launch configurations for different GPU architectures:\n";
    std::cout << "   launch_params_t<sm_75, block_dim, grid_dim, ...>\n";
    std::cout << "   launch_params_t<sm_80, block_dim, grid_dim, ...>\n";
    std::cout << "   launch_params_t<fallback, block_dim, grid_dim, ...>\n\n";
    
    std::cout << "2. Use tuple filtering to select matching configurations:\n";
    std::cout << "   - Check each config's sm_flags against SM_TARGET (bitwise AND)\n";
    std::cout << "   - Build tuple with std::conditional_t (include or skip)\n";
    std::cout << "   - Concatenate with std::tuple_cat to get all matches\n\n";
    
    std::cout << "3. Extract first match using tuple_element_t:\n";
    std::cout << "   - If matches found: use first one\n";
    std::cout << "   - If no matches: trigger compile error with always_false trick\n\n";
    
    std::cout << "4. The selected config provides optimal launch parameters:\n";
    std::cout << "   - Block dimensions (threads per block)\n";
    std::cout << "   - Grid dimensions (blocks per grid)\n";
    std::cout << "   - Shared memory allocation\n";
    std::cout << "   - Items per thread for blocked kernels\n\n";
    
    std::cout << "Result: Architecture-aware, zero-overhead kernel launches!\n\n";
    
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Demo Complete - All techniques explained!                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}
