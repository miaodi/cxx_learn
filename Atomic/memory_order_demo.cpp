#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {

constexpr int kPayloadValue = 42;
constexpr int kDefaultIterations = 200000;

struct LitmusCounts {
  int both_zero = 0;
  int first_zero_second_one = 0;
  int first_one_second_zero = 0;
  int both_one = 0;
};

int parse_iterations(int argc, char **argv) {
  if (argc < 2) {
    return kDefaultIterations;
  }

  char *end = nullptr;
  const long value = std::strtol(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("expected a positive iteration count");
  }

  return static_cast<int>(value);
}

void run_acquire_release_message_passing() {
  int payload = 0;
  int observed = 0;
  std::atomic<bool> ready{false};

  std::thread producer([&] {
    payload = kPayloadValue;
    ready.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    observed = payload;
  });

  producer.join();
  consumer.join();

  assert(observed == kPayloadValue);
  std::cout << "acquire/release message passing: observed payload = "
            << observed << " (guaranteed)\n";
}

template <std::memory_order Order>
LitmusCounts run_store_buffering_litmus(int iterations) {
  std::atomic<int> x{0};
  std::atomic<int> y{0};
  std::atomic<int> r1{0};
  std::atomic<int> r2{0};
  std::barrier start{3};
  std::barrier finish{3};
  LitmusCounts counts;

  std::thread first([&] {
    for (int i = 0; i < iterations; ++i) {
      start.arrive_and_wait();

      x.store(1, Order);
      r1.store(y.load(Order), std::memory_order_relaxed);

      finish.arrive_and_wait();
    }
  });

  std::thread second([&] {
    for (int i = 0; i < iterations; ++i) {
      start.arrive_and_wait();

      y.store(1, Order);
      r2.store(x.load(Order), std::memory_order_relaxed);

      finish.arrive_and_wait();
    }
  });

  for (int i = 0; i < iterations; ++i) {
    x.store(0, Order);
    y.store(0, Order);
    r1.store(-1, std::memory_order_relaxed);
    r2.store(-1, std::memory_order_relaxed);

    start.arrive_and_wait();
    finish.arrive_and_wait();

    const int first_result = r1.load(std::memory_order_relaxed);
    const int second_result = r2.load(std::memory_order_relaxed);

    if (first_result == 0 && second_result == 0) {
      ++counts.both_zero;
    } else if (first_result == 0 && second_result == 1) {
      ++counts.first_zero_second_one;
    } else if (first_result == 1 && second_result == 0) {
      ++counts.first_one_second_zero;
    } else if (first_result == 1 && second_result == 1) {
      ++counts.both_one;
    } else {
      throw std::logic_error("unexpected litmus-test result");
    }
  }

  first.join();
  second.join();
  return counts;
}

void print_litmus_counts(std::string_view name, const LitmusCounts &counts) {
  std::cout << name << " results:\n";
  std::cout << "  r1 = 0, r2 = 0: " << counts.both_zero << '\n';
  std::cout << "  r1 = 0, r2 = 1: " << counts.first_zero_second_one
            << '\n';
  std::cout << "  r1 = 1, r2 = 0: " << counts.first_one_second_zero
            << '\n';
  std::cout << "  r1 = 1, r2 = 1: " << counts.both_one << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int iterations = parse_iterations(argc, argv);

    run_acquire_release_message_passing();

    std::cout << "\nstore-buffering litmus test iterations = " << iterations
              << "\n";
    std::cout << "Each thread stores to one atomic and then loads the other.\n";
    std::cout << "The r1 = 0, r2 = 0 result is allowed with relaxed order "
                 "but forbidden with seq_cst.\n\n";

    const LitmusCounts relaxed =
        run_store_buffering_litmus<std::memory_order_relaxed>(iterations);
    const LitmusCounts seq_cst =
        run_store_buffering_litmus<std::memory_order_seq_cst>(iterations);

    print_litmus_counts("relaxed", relaxed);
    std::cout << '\n';
    print_litmus_counts("seq_cst", seq_cst);

    if (relaxed.both_zero == 0) {
      std::cout << "\nThis run did not observe relaxed r1 = 0, r2 = 0. "
                   "That outcome is still allowed by C++; try more "
                   "iterations or a weaker-memory CPU.\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "memory_order_demo: " << error.what() << '\n';
    return 1;
  }
}
