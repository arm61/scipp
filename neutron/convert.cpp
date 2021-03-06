// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <set>

#include "scipp/core/element/arg_list.h"

#include "scipp/variable/bucket_model.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/util.h"

#include "scipp/dataset/bins.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/dataset_util.h"

#include "scipp/neutron/constants.h"
#include "scipp/neutron/conversions.h"
#include "scipp/neutron/convert.h"

using namespace scipp::variable;
using namespace scipp::dataset;

namespace scipp::neutron {

template <class T, class Op>
T convert_generic(T &&d, const Dim from, const Dim to, Op op,
                  const VariableConstView &arg) {
  using core::element::arg_list;
  const auto op_ = overloaded{arg_list<double, std::tuple<float, double>>, op};
  const auto items = iter(d);
  // 1. Transform coordinate
  if (d.coords().contains(from)) {
    const auto coord = d.coords()[from];
    if (!coord.dims().contains(arg.dims()))
      d.coords().set(from, broadcast(coord, merge(arg.dims(), coord.dims())));
    transform_in_place(d.coords()[from], arg, op_);
  }
  // 2. Transform coordinates in bucket variables
  for (const auto &item : iter(d)) {
    if (item.dtype() != dtype<bucket<DataArray>>)
      continue;
    const auto &[indices, dim, buffer] =
        item.data().template constituents<bucket<DataArray>>();
    if (!buffer.coords().contains(from))
      continue;
    auto coord =
        make_bins(Variable(indices), dim, buffer.coords().extract(from));
    transform_in_place(coord, arg, op_);
    buffer.coords().set(
        to, std::get<2>(coord.template to_constituents<bucket<Variable>>()));
  }

  // 3. Rename dims
  d.rename(from, to);
  return std::move(d);
}

template <class T>
static T convert_with_factor(T &&d, const Dim from, const Dim to,
                             const Variable &factor) {
  return convert_generic(
      std::forward<T>(d), from, to,
      [](auto &coord, const auto &c) { coord *= c; }, factor);
}

/*
Dataset tofToDeltaE(const Dataset &d) {
  // There are two cases, direct inelastic and indirect inelastic. We can
  // distinguish them by the content of d.
  if (d.contains(Coord::Ei) && d.contains(Coord::Ef))
    throw std::runtime_error("Dataset contains Coord::Ei as well as Coord::Ef, "
                             "cannot have both for inelastic scattering.");

  // 1. Compute conversion factors
  const auto &compPos = d.get(Coord::ComponentInfo)[0](Coord::Position);
  const auto &sourcePos = compPos(Dim::Component, 0);
  const auto &samplePos = compPos(Dim::Component, 1);
  auto l1_square = norm(sourcePos - samplePos);
  l1_square *= l1_square;
  l1_square *= tofToEnergyPhysicalConstants;
  const auto specPos = getSpecPos(d);
  auto l2_square = norm(specPos - samplePos);
  l2_square *= l2_square;
  l2_square *= tofToEnergyPhysicalConstants;

  auto tofShift = makeVariable<double>({});
  auto scale = makeVariable<double>({});

  if (d.contains(Coord::Ei)) {
    // Direct-inelastic.
    // This is how we support multi-Ei data!
    tofShift = sqrt(l1_square / d(Coord::Ei));
    scale = std::move(l2_square);
  } else if (d.contains(Coord::Ef)) {
    // Indirect-inelastic.
    // Ef can be different for every spectrum.
    tofShift = sqrt(std::move(l2_square) / d(Coord::Ef));
    scale = std::move(l1_square);
  } else {
    throw std::runtime_error("Dataset contains neither Coord::Ei nor "
                             "Coord::Ef, this does not look like "
                             "inelastic-scattering data.");
  }

  // 2. Transform variables
  Dataset converted;
  for (const auto & [ name, tag, var ] : d) {
    auto varDims = var.dimensions();
    if (varDims.contains(Dim::Tof))
      varDims.relabel(varDims.index(Dim::Tof), Dim::DeltaE);
    if (tag == Coord::Tof) {
      Variable inv_tof = 1.0 / (var.reshape(varDims) - tofShift);
      Variable E = inv_tof * inv_tof * scale;
      if (d.contains(Coord::Ei)) {
        converted.insert(Coord::DeltaE, -(std::move(E) - d(Coord::Ei)));
      } else {
        converted.insert(Coord::DeltaE, std::move(E) - d(Coord::Ef));
      }
    } else if (tag == Data::Events) {
      throw std::runtime_error(
          "TODO Converting units of event data not implemented yet.");
    } else {
      if (counts::isDensity(var))
        throw std::runtime_error("TODO Converting units of count-density data "
                                 "not implemented yet for this case.");
      converted.insert(tag, name, var.reshape(varDims));
    }
  }

  // TODO Do we always require reversing for inelastic?
  // TODO Is is debatable whether this should revert automatically... probably
  // not, but we need to put a check in place for `rebin` to fail if the axis is
  // reversed.
  return reverse(converted, Dim::DeltaE);
}
*/

namespace {

template <class T> T coords_to_attrs(T &&x, const Dim from, const Dim to) {
  const auto to_attr = [&](const Dim field) {
    if (!x.coords().contains(field))
      return;
    Variable coord(x.coords()[field]);
    if constexpr (std::is_same_v<std::decay_t<T>, Dataset>) {
      x.coords().erase(field);
      for (const auto &item : iter(x))
        item.attrs().set(field, coord);
    } else {
      x.coords().erase(field);
      x.attrs().set(field, coord);
    }
  };
  // Will be replaced by explicit flag
  bool scatter = x.coords().contains(Dim("sample-position"));
  if (scatter) {
    std::set<Dim> pos_invariant{Dim::DSpacing, Dim::Q};
    if (pos_invariant.count(to))
      to_attr(Dim::Position);
  } else if (from == Dim::Tof) {
    to_attr(Dim::Position);
  }
  return std::move(x);
}

template <class T> T attrs_to_coords(T &&x, const Dim from, const Dim to) {
  const auto to_coord = [&](const Dim field) {
    auto &&range = iter(x);
    if (!range.begin()->attrs().contains(field))
      return;
    Variable attr(range.begin()->attrs()[field]);
    if constexpr (std::is_same_v<std::decay_t<T>, Dataset>) {
      for (const auto &item : range) {
        core::expect::equals(item.attrs()[field], attr);
        item.attrs().erase(field);
      }
      x.coords().set(field, attr);
    } else {
      x.attrs().erase(field);
      x.coords().set(field, attr);
    }
  };
  // Will be replaced by explicit flag
  bool scatter = x.coords().contains(Dim("sample-position"));
  if (scatter) {
    std::set<Dim> pos_invariant{Dim::DSpacing, Dim::Q};
    if (pos_invariant.count(from))
      to_coord(Dim::Position);
  } else if (to == Dim::Tof) {
    to_coord(Dim::Position);
  }
  return std::move(x);
}

} // namespace

template <class T> T convert_impl(T d, const Dim from, const Dim to) {
  for (const auto &item : iter(d))
    core::expect::notCountDensity(item.unit());
  d = attrs_to_coords(std::move(d), from, to);
  // This will need to be cleanup up in the future, but it is unclear how to do
  // so in a future-proof way. Some sort of double-dynamic dispatch based on
  // `from` and `to` will likely be required (with conversions helpers created
  // by a dynamic factory based on `Dim`). Conceptually we are dealing with a
  // bidirectional graph, and we would like to be able to find the shortest
  // paths between any two points, without defining all-to-all connections.
  // Approaches based on, e.g., a map of conversions and constants is also
  // tricky, since in particular the conversions are generic lambdas (passable
  // to `transform`) and are not readily stored as function pointers or
  // std::function.
  if ((from == Dim::Tof) && (to == Dim::DSpacing))
    return convert_with_factor(std::move(d), from, to,
                               constants::tof_to_dspacing(d));
  if ((from == Dim::DSpacing) && (to == Dim::Tof))
    return convert_with_factor(std::move(d), from, to,
                               reciprocal(constants::tof_to_dspacing(d)));

  if ((from == Dim::Tof) && (to == Dim::Wavelength))
    return convert_with_factor(std::move(d), from, to,
                               constants::tof_to_wavelength(d));
  if ((from == Dim::Wavelength) && (to == Dim::Tof))
    return convert_with_factor(std::move(d), from, to,
                               reciprocal(constants::tof_to_wavelength(d)));

  if ((from == Dim::Tof) && (to == Dim::Energy))
    return convert_generic(std::move(d), from, to, conversions::tof_to_energy,
                           constants::tof_to_energy(d));
  if ((from == Dim::Energy) && (to == Dim::Tof))
    return convert_generic(std::move(d), from, to, conversions::energy_to_tof,
                           constants::tof_to_energy(d));

  // lambda <-> Q conversion is symmetric
  if (((from == Dim::Wavelength) && (to == Dim::Q)) ||
      ((from == Dim::Q) && (to == Dim::Wavelength)))
    return convert_generic(std::move(d), from, to, conversions::wavelength_to_q,
                           constants::wavelength_to_q(d));

  throw except::UnitError(
      "Conversion between requested dimensions not implemented yet.");
}

DataArray convert(DataArray d, const Dim from, const Dim to) {
  return coords_to_attrs(convert_impl(std::move(d), from, to), from, to);
}

DataArray convert(const DataArrayConstView &d, const Dim from, const Dim to) {
  return convert(DataArray(d), from, to);
}

Dataset convert(Dataset d, const Dim from, const Dim to) {
  return coords_to_attrs(convert_impl(std::move(d), from, to), from, to);
}

Dataset convert(const DatasetConstView &d, const Dim from, const Dim to) {
  return convert(Dataset(d), from, to);
}

} // namespace scipp::neutron
