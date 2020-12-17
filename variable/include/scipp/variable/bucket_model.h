// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once
#include <algorithm>

#include "scipp/core/bucket_array_view.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/cumulative.h"
#include "scipp/variable/data_model.h"
#include "scipp/variable/except.h"
#include "scipp/variable/reduction.h"
#include "scipp/variable/util.h"

namespace scipp::variable {

/// Specialization of DataModel for "bucketed" data. T could be Variable,
/// DataArray, or Dataset.
///
/// A bucket in this context is defined as an element of a variable mapping to a
/// range of data, such as a slice of a DataArray.
template <class T> class DataModel<bucket<T>> : public VariableConcept {
  using Indices = std::conditional_t<is_view_v<T>, VariableConstView, Variable>;

public:
  using value_type = bucket<T>;
  using range_type = typename bucket<T>::range_type;

  DataModel(const VariableConstView &indices, const Dim dim, T buffer)
      : VariableConcept(indices.dims()),
        m_indices(validated_indices(indices, dim, buffer)), m_dim(dim),
        m_buffer(std::move(buffer)) {}

  VariableConceptHandle clone() const override {
    return std::make_unique<DataModel>(*this);
  }

  bool operator==(const DataModel &other) const noexcept {
    return dims() == other.dims() && m_indices == other.m_indices &&
           m_dim == other.m_dim && m_buffer == other.m_buffer;
  }
  bool operator!=(const DataModel &other) const noexcept {
    return !(*this == other);
  }

  VariableConceptHandle
  makeDefaultFromParent(const Dimensions &dims) const override {
    return std::make_unique<DataModel>(makeVariable<range_type>(dims), m_dim,
                                       T{m_buffer.slice({m_dim, 0, 0})});
  }

  VariableConceptHandle
  makeDefaultFromParent(const VariableConstView &shape) const override {
    const auto end = cumsum(shape);
    const auto begin = end - shape;
    const auto size = end.values<scipp::index>().as_span().back();
    if constexpr (is_view_v<T>) {
      // converting, e.g., bucket<VariableView> to bucket<Variable>
      return std::make_unique<DataModel<bucket<typename T::value_type>>>(
          zip(begin, begin), m_dim, resize_default_init(m_buffer, m_dim, size));
    } else {
      return std::make_unique<DataModel>(
          zip(begin, begin), m_dim, resize_default_init(m_buffer, m_dim, size));
    }
  }

  static DType static_dtype() noexcept { return scipp::dtype<bucket<T>>; }
  DType dtype() const noexcept override { return scipp::dtype<bucket<T>>; }

  bool hasVariances() const noexcept override { return false; }
  void setVariances(Variable &&) override {
    if (!core::canHaveVariances<T>())
      throw except::VariancesError("This data type cannot have variances.");
  }

  bool equals(const VariableConstView &a,
              const VariableConstView &b) const override;
  void copy(const VariableConstView &src,
            const VariableView &dest) const override;
  void assign(const VariableConcept &other) override;

  Dim dim() const noexcept { return m_dim; }
  // TODO Should the mutable version return a view to prevent risk of clients
  // breaking invariants of variable?
  const T &buffer() const noexcept { return m_buffer; }
  T &buffer() noexcept { return m_buffer; }
  const auto &indices() const { return m_indices; }
  auto &indices() { return m_indices; }

  ElementArrayView<bucket<T>> values(const core::ElementArrayViewParams &base) {
    return {index_values(base), m_dim, m_buffer};
  }
  ElementArrayView<const bucket<T>>
  values(const core::ElementArrayViewParams &base) const {
    return {index_values(base), m_dim, m_buffer};
  }

  scipp::index dtype_size() const override { return sizeof(range_type); }

private:
  static auto validated_indices(const VariableConstView &indices,
                                [[maybe_unused]] const Dim dim,
                                [[maybe_unused]] const T &buffer) {
    if constexpr (is_view_v<T>) {
      // Assume validation happened outside when constructing owning DataModel.
      return indices;
    } else {
      Variable copy(indices);
      const auto vals =
          scipp::span(copy.values<range_type>().data(),
                      copy.values<range_type>().data() + copy.dims().volume());
      std::sort(vals.begin(), vals.end());
      if ((!vals.empty() && (vals.begin()->first < 0)) ||
          (!vals.empty() && ((vals.end() - 1)->second > buffer.dims()[dim])))
        throw except::SliceError("Bucket indices out of range");
      if (std::adjacent_find(vals.begin(), vals.end(),
                             [](const auto a, const auto b) {
                               return a.second > b.first;
                             }) != vals.end())
        throw except::SliceError(
            "Bucket begin index must be less or equal to its end index.");
      if (std::find_if(vals.begin(), vals.end(), [](const auto x) {
            return x.second >= 0 && x.first > x.second;
          }) != vals.end())
        throw except::SliceError("Overlapping bucket indices are not allowed.");
      if (buffer.dims().ndim() > 1 && buffer.dims().index(dim) != 0) {
        throw except::BucketError(
            "The bin dimension must be the outermost (first) dimension of "
            "the buffer. This is a temporary workaround for issue #1479 and "
            "will be fixed in the future.");
      }
      // Copy to avoid a second memory allocation
      const auto &i = indices.values<range_type>();
      std::copy(i.begin(), i.end(), vals.begin());
      return copy;
    }
  }
  auto index_values(const core::ElementArrayViewParams &base) const {
    if constexpr (is_view_v<T>) {
      // m_indices is a VariableConstView and may thus contain slicing
      const auto params = m_indices.array_params();
      const auto offset = params.offset() + base.offset();
      // `base` comes from the variable (view) holding this model, so it
      // contains slicing applied after the one that may be part of m_indices.
      // Dimensions are thus given by `base`:
      const auto dims = base.dims();
      const auto dataDims = params.dataDims();
      return cast<range_type>(m_indices.underlying())
          .values(core::ElementArrayViewParams(offset, dims, dataDims, {}));
    } else {
      return cast<range_type>(m_indices).values(base);
    }
  }
  Indices m_indices;
  Dim m_dim;
  T m_buffer;
};

template <class T>
bool DataModel<bucket<T>>::equals(const VariableConstView &a,
                                  const VariableConstView &b) const {
  if (a.unit() != b.unit())
    return false;
  if (a.dims() != b.dims())
    return false;
  if (a.dtype() != b.dtype())
    return false;
  if (a.hasVariances() != b.hasVariances())
    return false;
  if (a.dims().volume() == 0 && a.dims() == b.dims())
    return true;
  // TODO This implementation is slow since it creates a view for every bucket.
  return equals_impl(a.values<bucket<T>>(), b.values<bucket<T>>());
}

template <class T>
void DataModel<bucket<T>>::copy(const VariableConstView &src,
                                const VariableView &dst) const {
  const auto &[indices0, dim0, buffer0] = src.constituents<bucket<T>>();
  const auto [begin0, end0] = unzip(indices0);
  const auto sizes1 = end0 - begin0;
  const auto end1 = cumsum(sizes1);
  const auto size1 = end1.template values<scipp::index>().as_span().back();
  auto indices1 = zip(end1 - sizes1, end1);
  auto buffer1 = resize_default_init(buffer0, dim0, size1);
  copy_slices(buffer0, buffer1, dim0, indices0, indices1);
  if constexpr (is_view_v<T>) {
    throw std::runtime_error(
        "Copying a non-owning binned view is not supported.");
  } else {
    dst.replace_model(
        DataModel<bucket<T>>{std::move(indices1), dim0, std::move(buffer1)});
  }
}

template <class T>
void DataModel<bucket<T>>::assign(const VariableConcept &other) {
  *this = requireT<const DataModel<bucket<T>>>(other);
}

} // namespace scipp::variable
