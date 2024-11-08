#include <algorithm>
#include <atomic>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <semaphore>
#include <thread>

TEST(atomic, memory_order_relaxed) {
  std::atomic<int> r1(0);
  std::atomic<int> r2(0);
  std::atomic<int> x(0);
  std::atomic<int> y(0);
  std::atomic<bool> flag;
  auto f1 = [&r1, &x, &y, &flag]() {
    while (!flag.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    // Thread 1:
    r1 = y.load(std::memory_order_relaxed); // A
    x.store(r1, std::memory_order_relaxed); // B
  };
  auto f2 = [&r2, &x, &y, &flag]() {
    while (!flag.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    // Thread 2:
    r2 = x.load(std::memory_order_relaxed); // C
    y.store(42, std::memory_order_relaxed); // D
  };
  int count = 0;
  while (true) {
    x = 0;
    y = 0;
    r1 = 0;
    r2 = 0;
    flag = false;
    std::thread t1(f1);
    std::thread t2(f2);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    flag.store(true, std::memory_order_relaxed);
    t1.join();
    t2.join();
    std::cout << "r1: " << r1 << ", r2: " << r2 << " x: " << x << " y: " << y
              << " count: " << count++ << std::endl;
    if (r1 == 42 && r2 == 42) {
      break;
    }
  }
}