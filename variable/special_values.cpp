// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include "scipp/core/element/special_values.h"
#include "scipp/variable/misc_operations.h"
#include "scipp/variable/transform.h"

using namespace scipp::core;

namespace scipp::variable {

Variable isnan(const VariableConstView &var) {
  return transform(var, element::isnan);
}

Variable isinf(const VariableConstView &var) {
  return transform(var, element::isinf);
}

Variable isfinite(const VariableConstView &var) {
  return transform(var, element::isfinite);
}

Variable isposinf(const VariableConstView &var) {
  return transform(var, element::isposinf);
}

Variable isneginf(const VariableConstView &var) {
  return transform(var, element::isneginf);
}

VariableView nan_to_num(const VariableConstView &var,
                        const VariableConstView &replacement,
                        const VariableView &out) {
  transform_in_place(out, var, replacement, element::nan_to_num_out_arg);
  return out;
}

VariableView positive_inf_to_num(const VariableConstView &var,
                                 const VariableConstView &replacement,
                                 const VariableView &out) {
  transform_in_place(out, var, replacement,
                     element::positive_inf_to_num_out_arg);
  return out;
}
VariableView negative_inf_to_num(const VariableConstView &var,
                                 const VariableConstView &replacement,
                                 const VariableView &out) {
  transform_in_place(out, var, replacement,
                     element::negative_inf_to_num_out_arg);
  return out;
}

Variable nan_to_num(const VariableConstView &var,
                    const VariableConstView &replacement) {
  return transform(var, replacement, element::nan_to_num);
}

Variable pos_inf_to_num(const VariableConstView &var,
                        const VariableConstView &replacement) {
  return transform(var, replacement, element::positive_inf_to_num);
}

Variable neg_inf_to_num(const VariableConstView &var,
                        const VariableConstView &replacement) {
  return transform(var, replacement, element::negative_inf_to_num);
}

} // namespace scipp::variable
