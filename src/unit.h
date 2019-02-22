/// @file
/// SPDX-License-Identifier: GPL-3.0-or-later
/// @author Simon Heybrock
/// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
/// National Laboratory, and European Spallation Source ERIC.
#ifndef UNIT_H
#define UNIT_H

#include <boost/units/base_dimension.hpp>
#include <boost/units/make_system.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/unit.hpp>
#include <variant>

namespace neutron {
namespace tof {

// Base dimension of counts
struct counts_base_dimension
    : boost::units::base_dimension<counts_base_dimension, 1> {};
typedef counts_base_dimension::dimension_type counts_dimension;

// TODO: do we need to add some derived units here?

// Base units from the base dimensions
struct counts_base_unit
    : boost::units::base_unit<counts_base_unit, counts_dimension, 1> {};
struct wavelength_base_unit
    : boost::units::base_unit<wavelength_base_unit,
                              boost::units::length_dimension, 2> {};
struct energy_base_unit
    : boost::units::base_unit<energy_base_unit, boost::units::energy_dimension,
                              3> {};
struct tof_base_unit
    : boost::units::base_unit<tof_base_unit, boost::units::time_dimension, 4> {
};

// Create the units system using the make_system utility
typedef boost::units::make_system<wavelength_base_unit, counts_base_unit,
                                  energy_base_unit, tof_base_unit>::type
    units_system;

typedef boost::units::unit<counts_dimension, units_system> counts;
typedef boost::units::unit<boost::units::length_dimension, units_system>
    wavelength;
typedef boost::units::unit<boost::units::energy_dimension, units_system> energy;
typedef boost::units::unit<boost::units::time_dimension, units_system> tof;

/// unit constants
BOOST_UNITS_STATIC_CONSTANT(angstrom, wavelength);
BOOST_UNITS_STATIC_CONSTANT(angstroms, wavelength);
BOOST_UNITS_STATIC_CONSTANT(meV, energy);
BOOST_UNITS_STATIC_CONSTANT(meVs, energy);
BOOST_UNITS_STATIC_CONSTANT(microsecond, tof);
BOOST_UNITS_STATIC_CONSTANT(microseconds, tof);

} // namespace tof
} // namespace neutron

// Define conversion factors: the conversion will work both ways with a single
// macro

// Convert angstroms to SI meters
BOOST_UNITS_DEFINE_CONVERSION_FACTOR(neutron::tof::wavelength_base_unit,
                                     boost::units::si::length, double, 1.0e-10);
// Convert meV to SI Joule
BOOST_UNITS_DEFINE_CONVERSION_FACTOR(
    neutron::tof::energy_base_unit, boost::units::si::energy, double,
    1.0e-3 * boost::units::si::constants::codata::e.value().value());
// Convert tof microseconds to SI seconds
BOOST_UNITS_DEFINE_CONVERSION_FACTOR(neutron::tof::tof_base_unit,
                                     boost::units::si::time, double, 1.0e-6);

// Define full and short names for the new units
template <>
struct boost::units::base_unit_info<neutron::tof::wavelength_base_unit> {
  static std::string name() { return "angstroms"; }
  static std::string symbol() { return "AA"; }
};

template <>
struct boost::units::base_unit_info<neutron::tof::counts_base_unit> {
  static std::string name() { return "counts"; }
  static std::string symbol() { return "counts"; }
};

template <>
struct boost::units::base_unit_info<neutron::tof::energy_base_unit> {
  static std::string name() { return "milli-electronvolt"; }
  static std::string symbol() { return "meV"; }
};

template <> struct boost::units::base_unit_info<neutron::tof::tof_base_unit> {
  static std::string name() { return "microseconds"; }
  static std::string symbol() { return "us"; }
};

// Define some helper variables to make the declaration of the set of allowed
// units more succinct
namespace units {
static boost::units::si::dimensionless dimensionless;
static boost::units::si::length m;
static boost::units::si::time s;
static boost::units::si::mass kg;
static neutron::tof::counts counts;
static neutron::tof::wavelength angstrom;
static neutron::tof::energy meV;
static neutron::tof::tof us;

// TODO counts/m is mainly for testing and should maybe be removed.
// Note the factor `dimensionless` in units that otherwise contain only non-SI
// factors. This is a trick to overcome some subtleties of working with
// heterogeneous unit systems in boost::units: We are combing SI units with our
// own, and the two are considered independent unless you convert explicitly.
// Therefore, in operations like (counts * m) / m, boosts is not cancelling the
// m as expected
// --- you get counts * dimensionless. Explicitly putting a factor dimensionless
// (dimensionless) into all our non-SI units avoids special-case handling in all
// operations (which would attempt to remove the dimensionless factor manually).
using type = std::variant<
    decltype(dimensionless), decltype(m), decltype(m * m), decltype(s),
    decltype(kg), decltype(counts * dimensionless), decltype(dimensionless / m),
    decltype(angstrom * dimensionless), decltype(meV * dimensionless),
    decltype(us * dimensionless), decltype(us * us * dimensionless),
    decltype(dimensionless / us), decltype(dimensionless / (us * us)),
    decltype(dimensionless / s), decltype(m * m * m * m),
    decltype(counts * counts * dimensionless), decltype(counts / m),
    decltype(meV * us * us / (m * m)), decltype(meV * us * us * dimensionless)>;
} // namespace units

class Unit {
public:
  // Define a std::variant which will hold the set of allowed units.
  // Any unit that does not exist in the variant will either fail to compile or
  // throw a std::runtime_error during operations such as multiplication or
  // division.
  //
  // TODO: maybe it is possible to create a helper that will automatically
  // generate the squares for variance?
  // The following was attempted but did not succeed:
  //  template <class... Ts> struct unit_and_variance {
  //  using type =
  //    std::variant<Ts..., decltype(std::declval<Ts>()*std::declval<Ts>())...>;
  //  };
  using unit_t = units::type;

  enum class Id : uint16_t {
    Dimensionless,
    Length,
    Area,
    AreaVariance,
    Counts,
    CountsVariance,
    CountsPerMeter,
    InverseLength,
    InverseTime,
    Energy,
    Wavelength,
    Time,
    Tof,
    Mass
  };
  // TODO should this be explicit?
  Unit() = default;
  // TODO: should we have a templated constructor here?
  // e.g.: template <class T> Unit(T &&unit) : m_unit(std::forward<T>(unit)) {}
  Unit(const Unit::Id id);
  Unit(const unit_t unit) : m_unit(unit) {}

  Unit::Id id() const;
  const Unit::unit_t &getUnit() const { return m_unit; }

private:
  unit_t m_unit;
  // TODO need to support scale
};

inline bool operator==(const Unit &a, const Unit &b) {
  return a.getUnit() == b.getUnit();
}
inline bool operator!=(const Unit &a, const Unit &b) { return !(a == b); }

Unit operator+(const Unit &a, const Unit &b);
Unit operator*(const Unit &a, const Unit &b);
Unit operator/(const Unit &a, const Unit &b);
Unit sqrt(const Unit &a);

namespace units {
inline bool containsCounts(const Unit &unit) {
  if (unit == Unit::Id::Counts || unit == Unit::Id::CountsPerMeter)
    return true;
  return false;
}
inline bool containsCountsVariance(const Unit &unit) {
  if (unit == Unit::Id::CountsVariance)
    return true;
  return false;
}
} // namespace units

#endif // UNIT_H
