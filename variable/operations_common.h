// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/variable/variable.h"

namespace scipp::variable {

// Helpers for in-place reductions and reductions with groupby.
SCIPP_VARIABLE_EXPORT void sum_impl(const VariableView &summed,
                                    const VariableConstView &var);
SCIPP_VARIABLE_EXPORT void all_impl(const VariableView &out,
                                    const VariableConstView &var);
SCIPP_VARIABLE_EXPORT void any_impl(const VariableView &out,
                                    const VariableConstView &var);
SCIPP_VARIABLE_EXPORT void max_impl(const VariableView &out,
                                    const VariableConstView &var);
SCIPP_VARIABLE_EXPORT void min_impl(const VariableView &out,
                                    const VariableConstView &var);
SCIPP_VARIABLE_EXPORT Variable mean_impl(const VariableConstView &var,
                                         const Dim dim,
                                         const VariableConstView &masks_sum);
SCIPP_VARIABLE_EXPORT VariableView mean_impl(const VariableConstView &var,
                                             const Dim dim,
                                             const VariableConstView &masks_sum,
                                             const VariableView &out);
SCIPP_VARIABLE_EXPORT Variable nanmean_impl(const VariableConstView &var,
                                            const Dim dim,
                                            const VariableConstView &masks_sum);
SCIPP_VARIABLE_EXPORT VariableView
nanmean_impl(const VariableConstView &var, const Dim dim,
             const VariableConstView &masks_sum, const VariableView &out);

} // namespace scipp::variable
