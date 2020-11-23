// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "random.h"

#include "scipp/dataset/bin.h"
#include "scipp/dataset/bins.h"
#include "scipp/dataset/string.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/misc_operations.h"

using namespace scipp;
using namespace scipp::dataset;

class DataArrayBinTest : public ::testing::Test {
protected:
  Variable data = makeVariable<double>(
      Dims{Dim::Event}, Shape{4}, Values{1, 2, 3, 4}, Variances{1, 3, 2, 4});
  Variable x =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, Values{3, 2, 4, 1});
  Variable mask = makeVariable<bool>(Dims{Dim::Event}, Shape{4},
                                     Values{true, false, false, false});
  Variable scalar = makeVariable<double>(Values{1.1});
  DataArray table =
      DataArray(data, {{Dim::X, x}, {Dim("scalar"), scalar}}, {{"mask", mask}});
  Variable edges_x =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{0, 2, 4});
};

TEST_F(DataArrayBinTest, 1d) {
  Variable sorted_data = makeVariable<double>(
      Dims{Dim::Event}, Shape{3}, Values{4, 1, 2}, Variances{4, 1, 3});
  Variable sorted_x =
      makeVariable<double>(Dims{Dim::Event}, Shape{3}, Values{1, 3, 2});
  Variable sorted_mask = makeVariable<bool>(Dims{Dim::Event}, Shape{3},
                                            Values{false, true, false});
  DataArray sorted_table =
      DataArray(sorted_data, {{Dim::X, sorted_x}, {Dim("scalar"), scalar}},
                {{"mask", sorted_mask}});

  const auto bucketed = bin(table, {edges_x});

  EXPECT_EQ(bucketed.dims(), Dimensions(Dim::X, 2));
  EXPECT_EQ(bucketed.coords()[Dim::X], edges_x);
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[0],
            sorted_table.slice({Dim::Event, 0, 1}));
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[1],
            sorted_table.slice({Dim::Event, 1, 3}));
}

TEST_F(DataArrayBinTest, 2d) {
  Variable edges_y =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{0, 1, 3});
  Variable y =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, Values{1, 2, 1, 2});
  table.coords().set(Dim::Y, y);

  Variable sorted_data = makeVariable<double>(
      Dims{Dim::Event}, Shape{3}, Values{4, 1, 2}, Variances{4, 1, 3});
  Variable sorted_x =
      makeVariable<double>(Dims{Dim::Event}, Shape{3}, Values{1, 3, 2});
  Variable sorted_y =
      makeVariable<double>(Dims{Dim::Event}, Shape{3}, Values{2, 1, 2});
  Variable sorted_mask = makeVariable<bool>(Dims{Dim::Event}, Shape{3},
                                            Values{false, true, false});
  DataArray sorted_table = DataArray(
      sorted_data,
      {{Dim::X, sorted_x}, {Dim::Y, sorted_y}, {Dim("scalar"), scalar}},
      {{"mask", sorted_mask}});

  const auto bucketed = bin(table, {edges_x, edges_y});

  EXPECT_EQ(bucketed.dims(), Dimensions({Dim::X, Dim::Y}, {2, 2}));
  EXPECT_EQ(bucketed.coords()[Dim::X], edges_x);
  EXPECT_EQ(bucketed.coords()[Dim::Y], edges_y);
  const auto empty_bucket = sorted_table.slice({Dim::Event, 0, 0});
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[0], empty_bucket);
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[1],
            sorted_table.slice({Dim::Event, 0, 1}));
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[2], empty_bucket);
  EXPECT_EQ(bucketed.values<bucket<DataArray>>()[3],
            sorted_table.slice({Dim::Event, 1, 3}));

  EXPECT_EQ(bin(bin(table, {edges_x}), {edges_y}), bucketed);
}

namespace {
auto make_table(const scipp::index size) {
  Random rand;
  rand.seed(0);
  const Dimensions dims(Dim::Row, size);
  const auto data =
      makeVariable<double>(Dimensions{dims}, Values(rand(dims.volume())));
  const auto x =
      makeVariable<double>(Dimensions{dims}, Values(rand(dims.volume())));
  const auto y =
      makeVariable<double>(Dimensions{dims}, Values(rand(dims.volume())));
  const auto group = astype(
      makeVariable<double>(Dimensions{dims}, Values(rand(dims.volume()))),
      dtype<int64_t>);
  return DataArray(data, {{Dim::X, x}, {Dim::Y, y}, {Dim("group"), group}});
}
} // namespace

class BinTest : public ::testing::TestWithParam<DataArray> {
protected:
  Variable groups = makeVariable<int64_t>(Dims{Dim("group")}, Shape{5},
                                          Values{-2, -1, 0, 1, 2});
  Variable edges_x =
      makeVariable<double>(Dims{Dim::X}, Shape{5}, Values{-2, -1, 0, 1, 2});
  Variable edges_y =
      makeVariable<double>(Dims{Dim::Y}, Shape{5}, Values{-2, -1, 0, 1, 2});
  Variable edges_x_coarse =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{-2, 1, 2});
  Variable edges_y_coarse =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{-2, -1, 2});

  void expect_near(const DataArrayConstView &a, const DataArrayConstView &b) {
    // "round" last digits for approximate floating point comparison
    const auto rounding = 100.0 * units::one * buckets::sum(a);
    EXPECT_EQ(buckets::sum(a) + rounding, buckets::sum(b) + rounding);
  }
};

INSTANTIATE_TEST_SUITE_P(InputSize, BinTest,
                         testing::Values(make_table(0), make_table(1),
                                         make_table(7), make_table(27),
                                         make_table(1233)));

TEST_P(BinTest, group) {
  const auto table = GetParam();
  const auto binned = bin(table, {}, {groups});
  EXPECT_EQ(binned.dims(), groups.dims());
}

TEST_P(BinTest, rebin_coarse_to_fine_1d) {
  const auto table = GetParam();
  EXPECT_EQ(bin(table, {edges_x}),
            bin(bin(table, {edges_x_coarse}), {edges_x}));
}

TEST_P(BinTest, rebin_fine_to_coarse_1d) {
  const auto table = GetParam();
  expect_near(bin(table, {edges_x_coarse}),
              bin(bin(table, {edges_x}), {edges_x_coarse}));
}

TEST_P(BinTest, 2d) {
  const auto table = GetParam();
  const auto x = bin(table, {edges_x});
  const auto x_then_y = bin(x, {edges_y});
  const auto xy = bin(table, {edges_x, edges_y});
  EXPECT_EQ(xy, x_then_y);
}

TEST_P(BinTest, rebin_coarse_to_fine_2d) {
  const auto table = GetParam();
  const auto xy_coarse = bin(table, {edges_x_coarse, edges_y_coarse});
  const auto xy = bin(table, {edges_x, edges_y});
  EXPECT_EQ(bin(xy_coarse, {edges_x, edges_y}), xy);
}

TEST_P(BinTest, rebin_fine_to_coarse_2d) {
  const auto table = GetParam();
  const auto xy_coarse = bin(table, {edges_x_coarse, edges_y_coarse});
  const auto xy = bin(table, {edges_x, edges_y});
  expect_near(bin(xy, {edges_x_coarse, edges_y_coarse}), xy_coarse);
}

TEST_P(BinTest, rebin_coarse_to_fine_2d_inner) {
  const auto table = GetParam();
  const auto xy_coarse = bin(table, {edges_x_coarse, edges_y_coarse});
  const auto xy = bin(table, {edges_x_coarse, edges_y});
  expect_near(bin(xy_coarse, {edges_y}), xy);
}

TEST_P(BinTest, rebin_coarse_to_fine_2d_outer) {
  const auto table = GetParam();
  auto xy_coarse = bin(table, {edges_x_coarse, edges_y});
  auto xy = bin(table, {edges_x, edges_y});
  expect_near(bin(xy_coarse, {edges_x}), xy);
  // Y is inside X and needs to be handled by `bin`, but coord is not required.
  xy_coarse.coords().erase(Dim::Y);
  xy.coords().erase(Dim::Y);
  expect_near(bin(xy_coarse, {edges_x}), xy);
}

TEST_P(BinTest, group_and_bin) {
  const auto table = GetParam();
  const auto x_group = bin(table, {edges_x}, {groups});
  const auto group = bin(table, {}, {groups});
  EXPECT_EQ(bin(group, {edges_x}, {}), x_group);
}
