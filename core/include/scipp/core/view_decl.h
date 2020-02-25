// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Dimitar Tasev
#ifndef SCIPP_CORE_VIEW_DECL_H
#define SCIPP_CORE_VIEW_DECL_H

#include <boost/iterator/transform_iterator.hpp>

#include "scipp/core/dataset_access.h"
#include "scipp/core/except.h"
#include "scipp/core/slice.h"
#include "scipp/core/variable.h"
#include "scipp/units/unit.h"

namespace scipp::core {

namespace detail {
using slice_list =
    boost::container::small_vector<std::pair<Slice, scipp::index>, 2>;

template <class Var> auto makeSlice(Var &var, const slice_list &slices) {
  std::conditional_t<std::is_const_v<Var>, VariableConstView, VariableView>
      slice(var);
  for (const auto [params, extent] : slices) {
    if (slice.dims().contains(params.dim())) {
      const auto new_end = params.end() + slice.dims()[params.dim()] - extent;
      const auto pointSlice = (new_end == -1);
      if (pointSlice) {
        slice = slice.slice(Slice{params.dim(), params.begin()});
      } else {
        slice = slice.slice(Slice{params.dim(), params.begin(), new_end});
      }
    }
  }
  return slice;
}

static constexpr auto make_key_value = [](auto &&view) {
  using In = decltype(view);
  using View =
      std::conditional_t<std::is_rvalue_reference_v<In>, std::decay_t<In>, In>;
  return std::pair<const std::string &, View>(
      view.name(), std::forward<decltype(view)>(view));
};

static constexpr auto make_key = [](auto &&view) -> decltype(auto) {
  using T = std::decay_t<decltype(view)>;
  if constexpr (std::is_base_of_v<DataArrayConstView, T>)
    return view.name();
  else
    return view.first;
};

static constexpr auto make_value = [](auto &&view) -> decltype(auto) {
  return view.second;
};

} // namespace detail

namespace ViewId {
class Attrs;
class Coords;
class Labels;
class Masks;
} // namespace ViewId
template <class Id, class Key, class Value> class ConstView;
template <class Base, class Access> class MutableView;

/// View for accessing coordinates of const Dataset and DataArrayConstView.
using CoordsConstView = ConstView<ViewId::Coords, Dim, Variable>;
/// View for accessing coordinates of Dataset and DataArrayView.
using CoordsView = MutableView<CoordsConstView, CoordAccess>;
/// View for accessing attributes of const Dataset and DataArrayConstView.
using AttrsConstView = ConstView<ViewId::Attrs, std::string, Variable>;
/// View for accessing attributes of Dataset and DataArrayView.
using AttrsView = MutableView<AttrsConstView, AttrAccess>;
/// View for accessing masks of const Dataset and DataArrayConstView
using MasksConstView = ConstView<ViewId::Masks, std::string, Variable>;
/// View for accessing masks of Dataset and DataArrayView
using MasksView = MutableView<MasksConstView, MaskAccess>;

/// Return the dimension for given coord.
/// @param var Coordinate variable
/// @param key Key of the coordinate in a coord dict
///
/// For dimension-coords, this is the same as the key, for non-dimension-coords
/// (labels) we adopt the convention that they are "label" their inner
/// dimension.
template <class T, class Key> Dim dim_of_coord(const T &var, const Key &key) {
  if constexpr (std::is_same_v<Key, Dim>) {
    const bool is_dimension_coord = var.dims().contains(key);
    return is_dimension_coord ? key : var.dims().inner();
  } else
    return var.dims().inner();
}

/// Common functionality for other const-view classes.
template <class Id, class Key, class Value> class ConstView {
private:
  struct make_item {
    const ConstView *view;
    template <class T> auto operator()(const T &item) const {
      return std::pair<Key, typename ConstView::mapped_type::const_view_type>(
          item.first, detail::makeSlice(*item.second.first, view->slices()));
    }
  };

public:
  using key_type = Key;
  using mapped_type = Value;
  using holder_type =
      std::unordered_map<key_type,
                         std::pair<const mapped_type *, mapped_type *>>;

  ConstView(holder_type &&items, const detail::slice_list &slices = {})
      : m_items(std::move(items)), m_slices(slices) {
    // TODO This is very similar to the code in makeViewItems(), provided that
    // we can give a good definion of the `dims` argument (roughly the space
    // spanned by all coords, excluding the dimensions that are sliced away).
    // Remove any items for a non-range sliced dimension. Identified via the
    // item in case of coords, or via the inner dimension in case of labels and
    // attributes.
    for (const auto &s : m_slices) {
      const auto slice = s.first;
      if (!slice.isRange()) { // The slice represents a point not a range.
                              // Dimension removed.
        for (auto it = m_items.begin(); it != m_items.end();) {
          auto erase = [slice](const auto &it2) {
            if constexpr (std::is_same_v<Key, Dim>) {
              // Remove dimension-coords for given dim, or non-dimension coords
              // if their inner dim is the given dim.
              constexpr auto is_dimension_coord = [](const auto &_) {
                return _.second.first->dims().contains(_.first);
              };
              return is_dimension_coord(it2)
                         ? it2.first == slice.dim()
                         : (!it2.second.first->dims().empty() &&
                            (it2.second.first->dims().inner() == slice.dim()));
            } else {
              return !it2.second.first->dims().empty() &&
                     (it2.second.first->dims().inner() == slice.dim());
            }
          };
          if (erase(*it))
            it = m_items.erase(it);
          else
            ++it;
        }
      }
    }
  }

  /// Return the number of coordinates in the view.
  index size() const noexcept { return scipp::size(m_items); }
  /// Return true if there are 0 coordinates in the view.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  /// Returns whether a given key is present in the view.
  bool contains(const Key &k) const {
    return m_items.find(k) != m_items.cend();
  }

  /// Return a const view to the coordinate for given dimension.
  typename mapped_type::const_view_type operator[](const Key key) const {
    expect::contains(*this, key);
    return detail::makeSlice(*m_items.at(key).first, m_slices);
  }

  auto find(const Key k) const && = delete;
  auto find(const Key k) const &noexcept {
    return boost::make_transform_iterator(m_items.find(k), make_item{this});
  }

  auto begin() const && = delete;
  /// Return const iterator to the beginning of all items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(m_items.begin(), make_item{this});
  }
  auto end() const && = delete;
  /// Return const iterator to the end of all items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(m_items.end(), make_item{this});
  }

  auto items_begin() const && = delete;
  /// Return const iterator to the beginning of all items.
  auto items_begin() const &noexcept { return begin(); }
  auto items_end() const && = delete;
  /// Return const iterator to the end of all items.
  auto items_end() const &noexcept { return end(); }

  auto keys_begin() const && = delete;
  /// Return const iterator to the beginning of all keys.
  auto keys_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_key);
  }
  auto keys_end() const && = delete;
  /// Return const iterator to the end of all keys.
  auto keys_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_key);
  }

  auto values_begin() const && = delete;
  /// Return const iterator to the beginning of all values.
  auto values_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_value);
  }
  auto values_end() const && = delete;
  /// Return const iterator to the end of all values.
  auto values_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_value);
  }

  ConstView slice(const Slice slice1) const {
    auto slices = m_slices;
    if constexpr (std::is_same_v<Key, Dim>) {
      const auto &coord = *m_items.at(slice1.dim()).first;
      slices.emplace_back(slice1, coord.dims()[slice1.dim()]);
    } else {
      throw std::runtime_error("TODO");
    }
    auto items = m_items;
    return ConstView(std::move(items), slices);
  }

  ConstView slice(const Slice slice1, const Slice slice2) const {
    return slice(slice1).slice(slice2);
  }
  ConstView slice(const Slice slice1, const Slice slice2,
                  const Slice slice3) const {
    return slice(slice1, slice2).slice(slice3);
  }

  bool operator==(const ConstView &other) const {
    if (size() != other.size())
      return false;
    for (const auto &[name, data] : *this) {
      try {
        if (data != other[name])
          return false;
      } catch (except::NotFoundError &) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const ConstView &other) const { return !operator==(other); }

  const auto &items() const noexcept { return m_items; }
  const auto &slices() const noexcept { return m_slices; }

protected:
  holder_type m_items;
  detail::slice_list m_slices;
};

/// Common functionality for other view classes.
template <class Base, class Access> class MutableView : public Base {
private:
  struct make_item {
    const MutableView<Base, Access> *view;
    template <class T> auto operator()(const T &item) const {
      return std::pair<typename Base::key_type,
                       typename Base::mapped_type::view_type>(
          item.first, detail::makeSlice(*item.second.second, view->slices()));
    }
  };

  MutableView(const Access &access, Base &&base)
      : Base(std::move(base)), m_access(access) {}

  Access m_access;

public:
  MutableView(const Access &access, typename Base::holder_type &&items,
              const detail::slice_list &slices = {})
      : Base(std::move(items), slices), m_access(access) {}

  /// Return a view to the coordinate for given dimension.
  typename Base::mapped_type::view_type
  operator[](const typename Base::key_type key) const {
    expect::contains(*this, key);
    return detail::makeSlice(*Base::items().at(key).second, Base::slices());
  }

  template <class T> auto find(const T k) const && = delete;
  template <class T> auto find(const T k) const &noexcept {
    return boost::make_transform_iterator(Base::items().find(k),
                                          make_item{this});
  }

  auto begin() const && = delete;
  /// Return iterator to the beginning of all items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(Base::items().begin(),
                                          make_item{this});
  }
  auto end() const && = delete;
  /// Return iterator to the end of all items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(Base::items().end(), make_item{this});
  }

  auto items_begin() const && = delete;
  /// Return iterator to the beginning of all items.
  auto items_begin() const &noexcept { return begin(); }
  auto items_end() const && = delete;
  /// Return iterator to the end of all items.
  auto items_end() const &noexcept { return end(); }

  auto values_begin() const && = delete;
  /// Return iterator to the beginning of all values.
  auto values_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_value);
  }
  auto values_end() const && = delete;
  /// Return iterator to the end of all values.
  auto values_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_value);
  }

  MutableView slice(const Slice slice1) const {
    // parent = nullptr since adding coords via slice is not supported.
    // TODO
    return MutableView(m_access, Base::slice(slice1));
  }

  MutableView slice(const Slice slice1, const Slice slice2) const {
    return slice(slice1).slice(slice2);
  }

  MutableView slice(const Slice slice1, const Slice slice2,
                    const Slice slice3) const {
    return slice(slice1, slice2).slice(slice3);
  }

  template <class VarOrView>
  void set(const typename Base::key_type key, VarOrView var) const {
    m_access.set(key, typename Base::mapped_type(std::move(var)));
  }

  void erase(const typename Base::key_type key) { m_access.erase(key); }
};

SCIPP_CORE_EXPORT Variable masks_merge_if_contains(const MasksConstView &masks,
                                                   const Dim dim);

SCIPP_CORE_EXPORT Variable masks_merge_if_contained(const MasksConstView &masks,
                                                    const Dimensions &dims);

} // namespace scipp::core
#endif // SCIPP_CORE_VIEW_DECL_H
