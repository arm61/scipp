# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the CovarianceVariable prototype.

Results are checked against an independent reference implementation of linear
error propagation, ``J C J^T`` with a numerically evaluated Jacobian.
"""

import numpy as np
import pytest
from covariance_variable import (
    CovarianceError,
    CovarianceVariable,
    concat,
    covariance_array,
    covariance_scalar,
)

import scipp as sc


def reference_propagate(f, x, cov, eps=1e-6):
    """Reference ``J C J^T`` with a numerically differentiated Jacobian."""
    x = np.asarray(x, dtype=float).ravel()
    y0 = np.asarray(f(x), dtype=float).ravel()
    jac = np.empty((y0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        jac[:, i] = (np.asarray(f(x + dx)).ravel() - np.asarray(f(x - dx)).ravel()) / (
            2 * eps
        )
    return jac @ np.asarray(cov) @ jac.T


COV2 = np.array([[0.04, 0.02], [0.02, 0.09]])
COV3 = np.array([[0.04, 0.02, 0.00], [0.02, 0.09, -0.01], [0.00, -0.01, 0.16]])


@pytest.fixture
def a():
    return covariance_array(
        dims=['x'], values=[1.0, 2.0, 3.0], covariance=COV3, unit='m'
    )


def cov_matrix(var):
    n = int(np.prod(var.shape, dtype=int))
    return np.asarray(var.covariance.values).reshape(n, n)


# -- construction and invariants -------------------------------------------


def test_is_a_scipp_variable(a):
    assert isinstance(a, sc.Variable)
    assert a.dims == ('x',)
    assert a.unit == sc.Unit('m')


def test_variances_are_the_diagonal_of_the_covariance(a):
    np.testing.assert_allclose(a.variances, np.diag(COV3))


def test_covariance_carries_squared_unit(a):
    assert a.covariance.unit == sc.Unit('m**2')
    assert a.covariance.dims == ('x', "x'")


def test_setting_variances_is_rejected(a):
    with pytest.raises(CovarianceError):
        a.variances = [1.0, 1.0, 1.0]


def test_setting_covariance_updates_variances(a):
    a.covariance = np.diag([1.0, 4.0, 9.0])
    np.testing.assert_allclose(a.variances, [1.0, 4.0, 9.0])


def test_from_variable_assumes_no_correlation():
    v = sc.array(dims=['x'], values=[1.0, 2.0], variances=[0.1, 0.2], unit='s')
    c = CovarianceVariable.from_variable(v)
    np.testing.assert_allclose(cov_matrix(c), np.diag([0.1, 0.2]))


def test_to_variable_preserves_marginal_variances(a):
    plain = a.to_variable()
    assert type(plain) is sc.Variable
    np.testing.assert_allclose(plain.variances, np.diag(COV3))


def test_reserved_dimension_label_is_rejected():
    with pytest.raises(CovarianceError):
        covariance_array(dims=["x'"], values=[1.0], covariance=[[1.0]])


# -- arithmetic against the reference ---------------------------------------


def test_addition_of_independent_variables():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    b = covariance_array(
        dims=['x'], values=[3.0, 4.0], covariance=np.diag([0.01, 0.02]), unit='m'
    )
    result = a + b
    np.testing.assert_allclose(result.values, [4.0, 6.0])
    np.testing.assert_allclose(cov_matrix(result), COV2 + np.diag([0.01, 0.02]))


def test_multiplication_matches_reference():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    b = covariance_array(
        dims=['x'], values=[3.0, 4.0], covariance=np.diag([0.01, 0.02]), unit='s'
    )
    result = a * b
    expected = reference_propagate(
        lambda v: v[:2] * v[2:],
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.block(
            [
                [COV2, np.zeros((2, 2))],
                [np.zeros((2, 2)), np.diag([0.01, 0.02])],
            ]
        ),
    )
    np.testing.assert_allclose(cov_matrix(result), expected, rtol=1e-6)
    assert result.unit == sc.Unit('m*s')


def test_division_matches_reference():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    b = covariance_array(
        dims=['x'], values=[3.0, 4.0], covariance=np.diag([0.01, 0.02]), unit='s'
    )
    result = a / b
    expected = reference_propagate(
        lambda v: v[:2] / v[2:],
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.block(
            [
                [COV2, np.zeros((2, 2))],
                [np.zeros((2, 2)), np.diag([0.01, 0.02])],
            ]
        ),
    )
    np.testing.assert_allclose(cov_matrix(result), expected, rtol=1e-5)


def test_self_addition_keeps_perfect_correlation(a):
    # a + a == 2a, so the covariance quadruples. Plain scipp special-cases this
    # too (see lib/variable/arithmetic.cpp), and we must agree.
    np.testing.assert_allclose(cov_matrix(a + a), 4 * COV3)
    np.testing.assert_allclose((a + a).values, [2.0, 4.0, 6.0])


def test_self_subtraction_is_exactly_zero(a):
    result = a - a
    np.testing.assert_allclose(result.values, 0.0)
    np.testing.assert_allclose(cov_matrix(result), 0.0, atol=1e-15)


def test_scalar_operand():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    np.testing.assert_allclose(cov_matrix(a * 3.0), 9 * COV2)
    np.testing.assert_allclose(cov_matrix(3.0 * a), 9 * COV2)
    np.testing.assert_allclose(cov_matrix(a + sc.scalar(1.0, unit='m')), COV2)


def test_reflected_operators_take_priority():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    plain = sc.array(dims=['x'], values=[1.0, 1.0], unit='m')
    assert isinstance(plain + a, CovarianceVariable)
    assert isinstance(plain * a, CovarianceVariable)
    result = plain - a
    assert isinstance(result, CovarianceVariable)
    np.testing.assert_allclose(result.values, [0.0, -1.0])
    np.testing.assert_allclose(cov_matrix(result), COV2)


def test_negation(a):
    np.testing.assert_allclose(cov_matrix(-a), COV3)
    np.testing.assert_allclose((-a).values, [-1.0, -2.0, -3.0])


def test_power_matches_reference():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2)
    expected = reference_propagate(lambda v: v**3, [1.0, 2.0], COV2)
    np.testing.assert_allclose(cov_matrix(a**3), expected, rtol=1e-5)


@pytest.mark.parametrize(
    ('name', 'func', 'unit'),
    [
        ('sqrt', np.sqrt, 'dimensionless'),
        ('exp', np.exp, 'dimensionless'),
        ('log', np.log, 'dimensionless'),
        ('sin', np.sin, 'rad'),
        ('cos', np.cos, 'rad'),
    ],
)
def test_unary_functions_match_reference(name, func, unit):
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit=unit)
    result = getattr(a, name)()
    expected = reference_propagate(func, [1.0, 2.0], COV2)
    np.testing.assert_allclose(cov_matrix(result), expected, rtol=1e-5)


# -- the payoff: correlations that plain scipp cannot represent -------------


def test_sum_accounts_for_off_diagonal_terms():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    total = a.sum('x')
    assert float(total.variance) == pytest.approx(COV2.sum())
    # Plain scipp would give only the sum of the diagonal, underestimating:
    assert float(sc.sum(a.to_variable()).variance) == pytest.approx(np.trace(COV2))
    assert float(total.variance) > float(sc.sum(a.to_variable()).variance)


def test_mean_scales_covariance_by_n_squared():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    assert float(a.mean('x').variance) == pytest.approx(COV2.sum() / 4)


def test_broadcast_is_allowed_and_records_correlations():
    a = covariance_scalar(2.0, variance=0.25, unit='m')
    wide = a.broadcast(sizes={'x': 3})
    assert wide.dims == ('x',)
    # Every element is the same measurement: perfectly correlated.
    np.testing.assert_allclose(cov_matrix(wide), np.full((3, 3), 0.25))
    np.testing.assert_allclose(wide.correlation.values, np.ones((3, 3)))


def test_broadcast_of_variances_is_forbidden_in_plain_scipp():
    """This is what ADR 0015 protects against, and what a covariance lifts."""
    data = sc.array(dims=['x'], values=[10.0, 20.0], variances=[10.0, 20.0])
    norm = sc.scalar(2.0, variance=0.04)
    with pytest.raises(sc.VariancesError):
        data / norm
    # The same expression is well defined once correlations are representable.
    result = CovarianceVariable.from_variable(data) / CovarianceVariable.from_variable(
        norm
    )
    assert isinstance(result, CovarianceVariable)
    assert cov_matrix(result)[0, 1] > 0.0


def test_normalisation_by_a_broadcast_denominator():
    """The neutron-normalisation case from ADR 0015 / doi:10.3233/JNR-220049."""
    data = covariance_array(
        dims=['x'], values=[10.0, 20.0], covariance=np.diag([10.0, 20.0]), unit='counts'
    )
    norm = covariance_scalar(2.0, variance=0.04)
    result = data / norm
    expected = reference_propagate(
        lambda v: v[:2] / v[2],
        np.array([10.0, 20.0, 2.0]),
        np.block(
            [
                [np.diag([10.0, 20.0]), np.zeros((2, 1))],
                [np.zeros((1, 2)), np.array([[0.04]])],
            ]
        ),
    )
    np.testing.assert_allclose(cov_matrix(result), expected, rtol=1e-5)
    # The shared denominator correlates the two outputs; scipp cannot say this.
    assert cov_matrix(result)[0, 1] > 0.0
    # And the total is more uncertain than an uncorrelated treatment implies.
    naive = np.trace(cov_matrix(result))
    assert float(result.sum('x').variance) > naive


# -- shape operations -------------------------------------------------------


def test_slicing_selects_both_index_axes(a):
    s = a['x', 0:2]
    np.testing.assert_allclose(s.values, [1.0, 2.0])
    np.testing.assert_allclose(cov_matrix(s), COV3[:2, :2])


def test_integer_indexing_gives_a_scalar(a):
    s = a['x', 1]
    assert s.dims == ()
    assert float(s.variance) == pytest.approx(COV3[1, 1])


def test_transpose_reorders_both_index_axes():
    values = np.arange(6.0).reshape(2, 3)
    n = 6
    rng = np.random.default_rng(1)
    m = rng.normal(size=(n, n))
    cov = (m @ m.T).reshape(2, 3, 2, 3)
    v = covariance_array(dims=['x', 'y'], values=values, covariance=cov, unit='m')
    t = v.transpose(['y', 'x'])
    assert t.dims == ('y', 'x')
    assert t.covariance.dims == ('y', 'x', "y'", "x'")
    np.testing.assert_allclose(t.values, values.T)
    np.testing.assert_allclose(cov_matrix(t), cov.transpose(1, 0, 3, 2).reshape(n, n))


def test_unit_conversion_scales_covariance_quadratically():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    cm = a.to(unit='cm')
    np.testing.assert_allclose(cm.values, [100.0, 200.0])
    np.testing.assert_allclose(cov_matrix(cm), COV2 * 1e4)


def test_concat_is_block_diagonal():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    b = covariance_array(dims=['x'], values=[3.0], covariance=[[0.01]], unit='m')
    c = concat([a, b], 'x')
    expected = np.zeros((3, 3))
    expected[:2, :2] = COV2
    expected[2, 2] = 0.01
    np.testing.assert_allclose(cov_matrix(c), expected)


def test_copy_is_independent(a):
    c = a.copy()
    assert isinstance(c, CovarianceVariable)
    c.covariance = np.zeros((3, 3))
    np.testing.assert_allclose(cov_matrix(a), COV3)


def test_correlation_matrix(a):
    sigma = np.sqrt(np.diag(COV3))
    np.testing.assert_allclose(a.correlation.values, COV3 / np.outer(sigma, sigma))


# -- degradation is explicit, never silent ---------------------------------


@pytest.mark.parametrize('name', ['flatten', 'fold', 'hist', 'cumsum', 'max'])
def test_unsupported_operations_raise(a, name):
    with pytest.raises(CovarianceError):
        getattr(a, name)()


def test_free_functions_silently_degrade_to_plain_variable(a):
    """Documents the known escape hatch: C++ free functions bypass the subclass."""
    result = sc.sum(a, 'x')
    assert type(result) is sc.Variable
    # Marginals stay correct because of the diag invariant, correlations are lost.
    assert float(result.variance) == pytest.approx(np.trace(COV3))
    assert float(result.variance) != pytest.approx(COV3.sum())


def test_storing_in_a_data_array_strips_the_covariance(a):
    """Documents the other known escape hatch: C++ containers copy-construct."""
    da = sc.DataArray(data=a)
    assert type(da.data) is sc.Variable
    np.testing.assert_allclose(da.data.variances, np.diag(COV3))
