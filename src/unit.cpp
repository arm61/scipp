/// @file
/// SPDX-License-Identifier: GPL-3.0-or-later
/// @author Simon Heybrock
/// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
/// National Laboratory, and European Spallation Source ERIC.
#include <stdexcept>

#include <boost/units/systems/si/amount.hpp>
#include <boost/units/systems/si/dimensionless.hpp>
#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/area.hpp>

#include "unit.h"

Unit makeUnit(const boost::units::si::dimensionless &) {
  return {Unit::Id::Dimensionless};
}
Unit makeUnit(const boost::units::si::length &) { return {Unit::Id::Length}; }
Unit makeUnit(const boost::units::si::area &) { return {Unit::Id::Area}; }
Unit makeUnit(const decltype(std::declval<boost::units::si::amount>() *
                             std::declval<boost::units::si::amount>()) &) {
  return {Unit::Id::CountsVariance};
}
Unit makeUnit(const decltype(std::declval<boost::units::si::dimensionless>() /
                             std::declval<boost::units::si::length>()) &) {
  return {Unit::Id::InverseLength};
}
template <class T> Unit makeUnit(const T &) {
  throw std::runtime_error("Unsupported unit combination");
}

Unit operator+(const Unit &a, const Unit &b) {
  if (a != b)
    throw std::runtime_error("Cannot add different units");
  return a;
}

template <class A> Unit multiply(const A &a, const Unit &b) {
  if (b == Unit{Unit::Id::Dimensionless})
    return makeUnit(a);
  if (b == Unit{Unit::Id::Length})
    return makeUnit(a * boost::units::si::length{});
  if (b == Unit{Unit::Id::Area})
    return makeUnit(a * boost::units::si::area{});
  if (b == Unit{Unit::Id::Counts})
    return makeUnit(a * boost::units::si::amount{});
  throw std::runtime_error("Unsupported unit on RHS");
}

template <class A> Unit divide(const A &a, const Unit &b) {
  if (b == Unit{Unit::Id::Dimensionless})
    return makeUnit(a);
  if (b == Unit{Unit::Id::Length})
    return makeUnit(a / boost::units::si::length{});
  if (b == Unit{Unit::Id::Area})
    return makeUnit(a / boost::units::si::area{});
  if (b == Unit{Unit::Id::Counts})
    return makeUnit(a / boost::units::si::amount{});
  throw std::runtime_error("Unsupported unit on RHS");
}

template <>
Unit multiply<boost::units::si::area>(const boost::units::si::area &,
                                      const Unit &b) {
  if (b == Unit{Unit::Id::Area})
    return Unit::Id::AreaVariance;
  throw std::runtime_error("Unsupported unit on RHS");
}

Unit operator*(const Unit &a, const Unit &b) {
  if (a == Unit{Unit::Id::Dimensionless})
    return b;
  if (b == Unit{Unit::Id::Dimensionless})
    return a;
  if (a == Unit{Unit::Id::Length})
    return multiply(boost::units::si::length{}, b);
  if (a == Unit{Unit::Id::Area})
    return multiply(boost::units::si::area{}, b);
  // TODO Abusing another boost unit for now, need to define our own.
  if (a == Unit{Unit::Id::Counts})
    return multiply(boost::units::si::amount{}, b);
  throw std::runtime_error("Unsupported unit on LHS");
}

Unit operator/(const Unit &a, const Unit &b) {
  if (a == Unit{Unit::Id::Dimensionless})
    return divide(boost::units::si::dimensionless{}, b);
  if (b == Unit{Unit::Id::Dimensionless})
    return a;
  if (a == Unit{Unit::Id::Length})
    return divide(boost::units::si::length{}, b);
  if (a == Unit{Unit::Id::Area})
    return divide(boost::units::si::area{}, b);
  // TODO Abusing another boost unit for now, need to define our own.
  if (a == Unit{Unit::Id::Counts})
    return divide(boost::units::si::amount{}, b);
  throw std::runtime_error("Unsupported unit on LHS");
}
