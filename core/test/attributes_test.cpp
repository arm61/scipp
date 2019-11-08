// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/dataset.h"

using namespace scipp;
using namespace scipp::core;
  // to test:
  // - operations with dataset and item attrs (sum, or directly apply_to_items?)
  // - binary in-place operations preserving dataset and item attrs

class AttributesTest : public ::testing::Test {
protected:
  const Variable scalar = makeVariable<double>(1);
  const Variable varX = makeVariable<double>({Dim::X, 2}, {2, 3});
  const Variable varZX =
      makeVariable<double>({{Dim::Y, 2}, {Dim::X, 2}}, {4, 5, 6, 7});
};

TEST_F(AttributesTest, dataset_attrs) {
  Dataset d;
  d.setAttr("scalar", scalar);
  d.setAttr("x", varX);
  ASSERT_EQ(d.attrs().size(), 2);
  ASSERT_TRUE(d.attrs().contains("scalar"));
  ASSERT_TRUE(d.attrs().contains("x"));
  ASSERT_EQ(d.dimensions(),
            (std::unordered_map<Dim, scipp::index>{{Dim::X, 2}}));
  d.eraseAttr("scalar");
  d.eraseAttr("x");
  ASSERT_EQ(d.attrs().size(), 0);
  ASSERT_EQ(d.dimensions(), (std::unordered_map<Dim, scipp::index>{}));
}

TEST_F(AttributesTest, dataset_item_attrs) {
  Dataset d;
  d.setData("a", varX);
  d["a"].attrs().set("scalar", scalar);
  d["a"].attrs().set("x", varX);
  d.attrs().set("dataset_attr", scalar);

  ASSERT_FALSE(d.attrs().contains("scalar"));
  ASSERT_FALSE(d.attrs().contains("x"));

  ASSERT_EQ(d["a"].attrs().size(), 2);
  ASSERT_TRUE(d["a"].attrs().contains("scalar"));
  ASSERT_TRUE(d["a"].attrs().contains("x"));
  ASSERT_FALSE(d["a"].attrs().contains("dataset_attr"));

  d["a"].attrs().erase("scalar");
  d["a"].attrs().erase("x");
  ASSERT_EQ(d["a"].attrs().size(), 0);
}

TEST_F(AttributesTest, dataset_item_attrs_dimensions_exceeding_data) {
  Dataset d;
  d.setData("scalar", scalar);
  EXPECT_THROW(d["scalar"].attrs().set("x", varX), except::DimensionError);
}

TEST_F(AttributesTest, slice_dataset_item_attrs) {
  Dataset d;
  d.setData("a", varZX);
  d["a"].attrs().set("scalar", scalar);
  d["a"].attrs().set("x", varX);

  // Same behavior as coord slicing:
  // - Lower-dimensional attrs are not hidden by slicing.
  // - Non-range slice hides attribute.
  // The alternative would be to handle attributes like data, but at least for
  // now coord-like handling appears to make more sense.
  ASSERT_TRUE(d["a"].slice({Dim::X, 0}).attrs().contains("scalar"));
  ASSERT_FALSE(d["a"].slice({Dim::X, 0}).attrs().contains("x"));
  ASSERT_TRUE(d["a"].slice({Dim::X, 0, 1}).attrs().contains("scalar"));
  ASSERT_TRUE(d["a"].slice({Dim::X, 0, 1}).attrs().contains("x"));
  ASSERT_TRUE(d["a"].slice({Dim::Y, 0}).attrs().contains("scalar"));
  ASSERT_TRUE(d["a"].slice({Dim::Y, 0}).attrs().contains("x"));
  ASSERT_TRUE(d["a"].slice({Dim::Y, 0, 1}).attrs().contains("scalar"));
  ASSERT_TRUE(d["a"].slice({Dim::Y, 0, 1}).attrs().contains("x"));
}

TEST_F(AttributesTest, binary_ops_in_place) {
  Dataset d1;
  d1.setData("a", varX);
  d1["a"].attrs().set("a_attr", scalar);
  d1.attrs().set("dataset_attr", scalar);

  Dataset d2;
  d2.setData("a", varX);
  d2["a"].attrs().set("a_attr", varX);
  d2.attrs().set("dataset_attr", varX);

  auto result(d1);
  result += d2;
  ASSERT_TRUE(result.attrs().contains("dataset_attr"));
  EXPECT_EQ(result.attrs()["dataset_attr"], scalar);
  ASSERT_TRUE(result["a"].attrs().contains("a_attr"));
  EXPECT_EQ(result["a"].attrs()["a_attr"], scalar);
}

TEST_F(AttributesTest, reduction_ops) {
  Dataset d;
  d.setData("a", varX);
  d["a"].attrs().set("a_attr", scalar);
  d["a"].attrs().set("a_attr_x", varX);
  d.attrs().set("dataset_attr", scalar);
  d.attrs().set("dataset_attr_x", varX);

  const auto result = sum(d, Dim::X);
  ASSERT_TRUE(result.attrs().contains("dataset_attr"));
  ASSERT_FALSE(result.attrs().contains("dataset_attr_x"));
  EXPECT_EQ(result.attrs()["dataset_attr"], scalar);
  ASSERT_TRUE(result["a"].attrs().contains("a_attr"));
  ASSERT_FALSE(result["a"].attrs().contains("a_attr_x"));
  EXPECT_EQ(result["a"].attrs()["a_attr"], scalar);
}
