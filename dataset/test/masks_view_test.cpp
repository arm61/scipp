// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/dimensions.h"
#include "scipp/dataset/dataset.h"
#include "test_macros.h"

using namespace scipp;

TEST(MasksViewTest, irreducible_mask) {
  DataArray a(
      makeVariable<double>(Dims{Dim::X, Dim::Y, Dim::Z}, Shape{2, 3, 4}));
  const auto x =
      makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{true, false});
  const auto y =
      makeVariable<bool>(Dims{Dim::Y}, Shape{3}, Values{true, false, false});
  a.masks().set("x", x);
  a.masks().set("y", y);
  EXPECT_EQ(irreducible_mask(a.masks(), Dim::X), x);
  EXPECT_EQ(irreducible_mask(a.masks(), Dim::Y), y);
  a.masks().set("xy", makeVariable<bool>(
                          Dims{Dim::X, Dim::Y}, Shape{2, 3},
                          Values{false, false, false, false, true, false}));
  EXPECT_EQ(irreducible_mask(a.masks(), Dim::X),
            makeVariable<bool>(Dims{Dim::X, Dim::Y}, Shape{2, 3},
                               Values{true, true, true, false, true, false}));
  EXPECT_EQ(irreducible_mask(a.masks(), Dim::Y),
            makeVariable<bool>(Dims{Dim::X, Dim::Y}, Shape{2, 3},
                               Values{true, false, false, true, true, false}));
  EXPECT_EQ(irreducible_mask(a.masks(), Dim::Z), Variable{});
}