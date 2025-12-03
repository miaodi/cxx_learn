#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include "parallel_sort.h"

#ifdef USE_GTEST
#include <gtest/gtest.h>
#endif

static std::vector<int> make_random_vec(size_t n, int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-100000, 100000);
    std::vector<int> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

static std::vector<std::string> make_random_strings(size_t n, int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> len_dist(0, 20);
    std::uniform_int_distribution<int> ch_dist('a', 'z');
    std::vector<std::string> v(n);
    for (auto &s : v) {
        int len = len_dist(rng);
        s.resize(len);
        for (char &c : s) c = static_cast<char>(ch_dist(rng));
    }
    return v;
}

#ifdef USE_GTEST

TEST(ParallelSort, HandlesEmpty) {
    std::vector<int> v;
    parallel::sort(v.begin(), v.end(), /*nthreads=*/4);
    EXPECT_TRUE(v.empty());
}

TEST(ParallelSort, HandlesSingle) {
    std::vector<int> v{42};
    parallel::sort(v.begin(), v.end(), /*nthreads=*/4);
    EXPECT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 42);
}

TEST(ParallelSort, SmallFallsBackToStdSort) {
    std::vector<int> v{3,2,1};
    parallel::sort(v.begin(), v.end(), /*nthreads=*/8);
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(ParallelSort, LargeRandomIntsDefaultComp) {
    auto v = make_random_vec(200000);
    auto ref = v;
    std::sort(ref.begin(), ref.end());
    parallel::sort(v.begin(), v.end(), /*nthreads=*/8);
    EXPECT_EQ(v, ref);
}

TEST(ParallelSort, RandomStringsCustomComp) {
    auto v = make_random_strings(50000);
    auto ref = v;
    auto comp = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    };
    std::sort(ref.begin(), ref.end(), comp);
    parallel::sort(v.begin(), v.end(), comp, /*nthreads=*/6);
    EXPECT_EQ(v, ref);
}

TEST(ParallelSort, NonPowerOfTwoThreads) {
    auto v = make_random_vec(300000);
    auto ref = v;
    std::sort(ref.begin(), ref.end());
    parallel::sort(v.begin(), v.end(), /*nthreads=*/7);
    EXPECT_EQ(v, ref);
}

#else

int main() {
    {
        std::vector<int> v;
        parallel::sort(v.begin(), v.end(), 4);
        if (!v.empty()) return 1;
    }
    {
        std::vector<int> v{42};
        parallel::sort(v.begin(), v.end(), 4);
        if (v.size()!=1 || v[0]!=42) return 2;
    }
    {
        std::vector<int> v{3,2,1};
        parallel::sort(v.begin(), v.end(), 8);
        if (!std::is_sorted(v.begin(), v.end())) return 3;
    }
    {
        auto v = make_random_vec(200000);
        auto ref = v;
        std::sort(ref.begin(), ref.end());
        parallel::sort(v.begin(), v.end(), 8);
        if (v != ref) return 4;
    }
    {
        auto v = make_random_strings(50000);
        auto ref = v;
        auto comp = [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        };
        std::sort(ref.begin(), ref.end(), comp);
        parallel::sort(v.begin(), v.end(), comp, 6);
        if (v != ref) return 5;
    }
    {
        auto v = make_random_vec(300000);
        auto ref = v;
        std::sort(ref.begin(), ref.end());
        parallel::sort(v.begin(), v.end(), 7);
        if (v != ref) return 6;
    }
    std::cout << "All parallel_sort tests passed\n";
    return 0;
}
#endif
