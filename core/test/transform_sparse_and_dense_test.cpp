// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>
#include <vector>

#include "scipp/core/transform.h"
#include "scipp/core/variable.h"

#include "../operators.h"

using namespace scipp;
using namespace scipp::core;

TEST(TransformSparseAndDenseTest, two_args) {
  auto var = makeVariable<double>({Dim::Y, Dim::X}, {2, Dimensions::Sparse});
  auto vals = var.sparseValues<double>();
  vals[0] = {1, 2, 3};
  vals[1] = {4};

  auto dense =
      makeVariable<double>({{Dim::Y, 2}, {Dim::X, 2}}, {1.1, 2.2, 3.3, 4.4});
  std::vector<span<double>> spans;
  for (scipp::index i = 0; i < 2; ++i)
    spans.emplace_back(dense.values<double>().subspan(2 * i, 2));
  auto dense_view = makeVariable<span<double>>({Dim::Y, 2}, spans);

  const auto result = transform<
      pair_custom_t<std::pair<sparse_container<double>, span<double>>>>(
      var, dense_view,
      overloaded{[](const auto &a, const auto &b) {
                   fprintf(stderr, "%ld %ld\n", a.size(), b.size());
                   for (auto &a_ : a)
                     fprintf(stderr, "a: %lf\n", a_);
                   for (auto &b_ : b)
                     fprintf(stderr, "b: %lf\n", b_);
                   return a;
                 },
                 [](const units::Unit &a, const units::Unit &) { return a; },
                 transform_flags::expect_no_variance_arg<0>,
                 transform_flags::expect_no_variance_arg<1>});
}

TEST(TransformSparseAndDenseTest, three_args) {
  auto var = makeVariable<double>({Dim::Y, Dim::X}, {2, Dimensions::Sparse});
  auto vals = var.sparseValues<double>();
  vals[0] = {1, 2, 3};
  vals[1] = {4};

  auto dense =
      makeVariable<double>({{Dim::Y, 2}, {Dim::X, 2}}, {1.1, 2.2, 3.3, 4.4});
  std::vector<span<double>> spans;
  for (scipp::index i = 0; i < 2; ++i)
    spans.emplace_back(dense.values<double>().subspan(2 * i, 2));
  auto dense_view = makeVariable<span<double>>({Dim::Y, 2}, spans);

  const auto result = transform<std::tuple<
      std::tuple<sparse_container<double>, span<double>, span<double>>>>(
      var, dense_view, dense_view,
      overloaded{[](const auto &a, const auto &b, const auto &c) {
                   fprintf(stderr, "%ld %ld %ld\n", a.size(), b.size(),
                           c.size());
                   for (auto &a_ : a)
                     fprintf(stderr, "a: %lf\n", a_);
                   for (auto &b_ : b)
                     fprintf(stderr, "b: %lf\n", b_);
                   return a;
                 },
                 [](const units::Unit &a, const units::Unit &,
                    const units::Unit &) { return a; },
                 transform_flags::expect_no_variance_arg<0>,
                 transform_flags::expect_no_variance_arg<1>,
                 transform_flags::expect_no_variance_arg<2>});
}
