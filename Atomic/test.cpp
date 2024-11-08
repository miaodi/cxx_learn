#include <algorithm>
#include <atomic>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <thread>
#include <unordered_set>
TEST(atomic, plain_int) {
  int count = 0;
  auto f = [&count]() {
    for (int i = 0; i < 10000; i++) {
      std::this_thread::yield();
      count++;
    }
  };
  std::thread t1(f);
  std::thread t2(f);
  t1.join();
  t2.join();
  std::cout << "count value: " << count << std::endl;
}

TEST(atomic, atomic_int) {
  std::atomic<int> count(0);
  auto f = [&count]() {
    for (int i = 0; i < 10000; i++) {
      std::this_thread::yield();
      count = count + 1;
    }
  };
  std::thread t1(f);
  std::thread t2(f);
  t1.join();
  t2.join();
  std::cout << "count value: " << count << std::endl;
}

TEST(atomic, atomic_int_atomic_op) {
  std::atomic<int> count(0);
  auto f = [&count]() {
    for (int i = 0; i < 10000; i++) {
      std::this_thread::yield();
      count += 1;
    }
  };
  std::thread t1(f);
  std::thread t2(f);
  t1.join();
  t2.join();
  std::cout << "count value: " << count << std::endl;
}