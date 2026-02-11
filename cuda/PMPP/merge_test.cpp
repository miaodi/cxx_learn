#include "merge.cuh"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// ------------------------------ Merge Tests ------------------------------
class MergeTest : public ::testing::Test {
protected:
  void SetUp() override { rng.seed(2027); }

  void makeSortedInput(std::vector<int> &out, int size, int min_val,
                       int max_val) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    out.resize(size);
    for (int i = 0; i < size; ++i) {
      out[i] = dist(rng);
    }
    std::sort(out.begin(), out.end());
  }

  void verifyMerge(const std::vector<int> &a, const std::vector<int> &b) {
    std::vector<int> expected(a.size() + b.size());
    std::vector<int> actual(a.size() + b.size());

    std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin());
    PMPP::merge(const_cast<int *>(a.data()), static_cast<int>(a.size()),
                const_cast<int *>(b.data()), static_cast<int>(b.size()),
                actual.data());

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], actual[i]) << "Merge mismatch at index " << i;
    }
  }

  void verifyMergeShared(const std::vector<int> &a, const std::vector<int> &b) {
    std::vector<int> expected(a.size() + b.size());
    std::vector<int> actual(a.size() + b.size());

    std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin());
    PMPP::merge_shared(const_cast<int *>(a.data()), static_cast<int>(a.size()),
                       const_cast<int *>(b.data()), static_cast<int>(b.size()),
                       actual.data());

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], actual[i])
          << "MergeShared mismatch at index " << i;
    }
  }

  void verifyMergeSharedPartitioned(const std::vector<int> &a,
                                    const std::vector<int> &b) {
    std::vector<int> expected(a.size() + b.size());
    std::vector<int> actual(a.size() + b.size());

    std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin());
    PMPP::merge_shared_partitioned(
        const_cast<int *>(a.data()), static_cast<int>(a.size()),
        const_cast<int *>(b.data()), static_cast<int>(b.size()), actual.data());

    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], actual[i])
          << "MergeShared mismatch at index " << i;
    }
  }

  std::mt19937 rng;
};

TEST_F(MergeTest, SmallEvenSizes) {
  std::vector<int> a{1, 3, 5, 7};
  std::vector<int> b{2, 4, 6, 8};
  verifyMerge(a, b);
}

TEST_F(MergeTest, SharedSmallEvenSizes) {
  std::vector<int> a{1, 3, 5, 7};
  std::vector<int> b{2, 4, 6, 8};
  verifyMergeShared(a, b);
}

TEST_F(MergeTest, SharedPartitionedSmallEvenSizes) {
  std::vector<int> a{1, 3, 5, 7};
  std::vector<int> b{2, 4, 6, 8};
  verifyMergeSharedPartitioned(a, b);
}

TEST_F(MergeTest, UnevenSizesWithDuplicates) {
  std::vector<int> a{1, 2, 2, 9, 10};
  std::vector<int> b{2, 3, 3, 7};
  verifyMerge(a, b);
}

TEST_F(MergeTest, SharedUnevenSizesWithDuplicates) {
  std::vector<int> a{1, 2, 2, 9, 10};
  std::vector<int> b{2, 3, 3, 7};
  verifyMergeShared(a, b);
}

TEST_F(MergeTest, SharedPartitionedUnevenSizesWithDuplicates) {
  std::vector<int> a{1, 2, 2, 9, 10};
  std::vector<int> b{2, 3, 3, 7};
  verifyMergeSharedPartitioned(a, b);
}

TEST_F(MergeTest, OneEmptyInput) {
  std::vector<int> a;
  std::vector<int> b{1, 2, 3, 4, 5};
  verifyMerge(a, b);
}

TEST_F(MergeTest, SharedOneEmptyInput) {
  std::vector<int> a;
  std::vector<int> b{1, 2, 3, 4, 5};
  verifyMergeShared(a, b);
}

TEST_F(MergeTest, SharedPartitionedOneEmptyInput) {
  std::vector<int> a;
  std::vector<int> b{1, 2, 3, 4, 5};
  verifyMergeSharedPartitioned(a, b);
}

TEST_F(MergeTest, RandomLargeInputs) {
  std::vector<int> a;
  std::vector<int> b;
  makeSortedInput(a, 4096, -10000, 10000);
  makeSortedInput(b, 8192, -10000, 10000);
  verifyMerge(a, b);
}

TEST_F(MergeTest, SharedRandomLargeInputs) {
  std::vector<int> a;
  std::vector<int> b;
  makeSortedInput(a, 4096, -10000, 10000);
  makeSortedInput(b, 8192, -10000, 10000);
  verifyMergeShared(a, b);
}

TEST_F(MergeTest, SharedPartitionedRandomLargeInputs) {
  std::vector<int> a;
  std::vector<int> b;
  makeSortedInput(a, 4096, -10000, 10000);
  makeSortedInput(b, 8192, -10000, 10000);
  verifyMergeSharedPartitioned(a, b);
}

TEST_F(MergeTest, SharedPartitionedRandomLargeInputsLarge) {
  std::vector<int> a;
  std::vector<int> b;
  makeSortedInput(a, 16777216, -10000, 10000);
  makeSortedInput(b, 524288, -10000, 10000);
  verifyMergeSharedPartitioned(a, b);
}