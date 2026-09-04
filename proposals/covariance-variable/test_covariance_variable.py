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
    install_dispatch,
    uninstall_dispatch,
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


def test_html_repr_is_not_prefixed_with_scipp(a):
    html = a._repr_html_()
    assert "sc-obj-type'>CovarianceVariable " in html
    assert 'scipp.CovarianceVariable' not in html


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


@pytest.mark.parametrize(
    'name',
    ['flatten', 'fold', 'hist', 'cumsum', 'max', 'squeeze', 'astype', 'round'],
)
def test_unsupported_operations_raise(a, name):
    with pytest.raises(CovarianceError):
        getattr(a, name)()


def test_every_inherited_variable_method_is_accounted_for():
    """No Variable method may degrade silently.

    A hand-written denylist let `squeeze`, `astype`, `round`, `rename_dims`,
    `all` and `any` fall through to C++ and quietly drop the covariance. The
    stubs are generated from Variable's API instead, so every public method is
    either implemented here or replaced by a raising stub -- and that stays
    true when scipp adds methods.
    """
    import inspect

    missing = [
        name
        for name in dir(sc.Variable)
        if not name.startswith('_')
        and name not in ('plot', 'underlying_size')
        and not isinstance(inspect.getattr_static(sc.Variable, name), property)
        and callable(getattr(sc.Variable, name, None))
        and name not in vars(CovarianceVariable)
    ]
    assert missing == []


# -- DataGroup is the one container that preserves the subclass ------------


def test_data_group_preserves_the_subclass(a):
    dg = sc.DataGroup({'a': a})
    assert dg['a'] is a
    # DataGroup methods dispatch to the item's method, so overrides are used.
    assert isinstance(dg.sum('x')['a'], CovarianceVariable)
    assert isinstance(dg.mean('x')['a'], CovarianceVariable)
    assert isinstance(dg['x', 0:2]['a'], CovarianceVariable)
    assert isinstance(dg.copy()['a'], CovarianceVariable)
    assert isinstance((dg + dg)['a'], CovarianceVariable)


def test_free_functions_on_a_data_group_still_strip(a):
    """data_group_nary calls the scipp *free* function on each item."""
    dg = sc.DataGroup({'a': a})
    assert type(sc.sum(dg)['a']) is sc.Variable


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


def test_a_data_array_inside_a_data_group_still_strips(a):
    """The DataGroup is innocent; wrapping in a DataArray first is not.

    `sc.DataGroup({'x': sc.DataArray(data=cv)})` looks like it should work,
    because DataGroup preserves the subclass -- but the DataArray has already
    dropped it by then.
    """
    dg = sc.DataGroup({'x': sc.DataArray(data=a)})
    assert type(dg['x'].data) is sc.Variable
    assert isinstance(sc.DataGroup({'x': a})['x'], CovarianceVariable)


def test_covariance_cannot_ride_along_as_a_coord(a):
    """A coord carrying the mirror dimension is rejected by DataArray.

    There is therefore no way to keep a covariance inside a DataArray at all,
    neither as data nor as metadata.
    """
    with pytest.raises(sc.DimensionError):
        sc.DataArray(data=a.to_variable(), coords={'cov': a.covariance})


def test_nested_data_group_is_the_working_container(a):
    """The DataArray-shaped alternative that does preserve the covariance."""
    coord = sc.arange('x', 3.0)
    dg = sc.DataGroup({'item': sc.DataGroup({'data': a, 'x': coord})})
    assert isinstance(dg['item']['data'], CovarianceVariable)
    # and it still slices as a unit
    assert isinstance(dg['x', 0:2]['item']['data'], CovarianceVariable)


# -- in-place operators must not leave a stale covariance ------------------


def test_in_place_operators_keep_the_invariant(a):
    """pybind11's in-place slots mutate the buffer and return the same object.

    That left `_covariance` stale -- `a -= a` gave zero values but non-zero
    variances. The overrides rebind instead.
    """
    for op in ('__iadd__', '__isub__', '__imul__', '__itruediv__'):
        x = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
        rhs = (
            sc.scalar(2.0)
            if op in ('__imul__', '__itruediv__')
            else covariance_array(
                dims=['x'], values=[1.0, 1.0], covariance=COV2, unit='m'
            )
        )
        result = getattr(x, op)(rhs)
        assert isinstance(result, CovarianceVariable), op
        np.testing.assert_allclose(
            result.variances, np.diag(cov_matrix(result)), err_msg=op
        )


def test_self_subtraction_in_place_is_exactly_zero(a):
    a -= a
    np.testing.assert_allclose(a.values, 0.0)
    np.testing.assert_allclose(a.variances, 0.0, atol=1e-15)


def test_abs_propagates_the_covariance():
    a = covariance_array(dims=['x'], values=[-1.0, 2.0], covariance=COV2, unit='m')
    result = abs(a)
    np.testing.assert_allclose(result.values, [1.0, 2.0])
    signs = np.array([-1.0, 1.0])
    np.testing.assert_allclose(cov_matrix(result), np.outer(signs, signs) * COV2)


@pytest.mark.parametrize('op', ['__floordiv__', '__mod__', '__invert__'])
def test_operators_without_covariance_meaning_raise(a, op):
    with pytest.raises(CovarianceError):
        getattr(a, op)(a)


def test_comparisons_still_return_plain_variables(a):
    """Boolean results carry no uncertainty, so degrading there is correct."""
    assert type(a == a) is sc.Variable
    assert type(a < a) is sc.Variable


# -- free-function dispatch -------------------------------------------------


@pytest.fixture
def dispatch():
    install_dispatch()
    yield
    uninstall_dispatch()


def test_free_functions_strip_without_dispatch(a):
    dg = sc.DataGroup({'a': a})
    assert type(sc.sum(dg, 'x')['a']) is sc.Variable


@pytest.mark.usefixtures('dispatch')
def test_free_functions_dispatch_inside_a_data_group():
    a = covariance_array(dims=['x'], values=[1.0, 2.0], covariance=COV2, unit='m')
    dg = sc.DataGroup({'a': a})
    total = sc.sum(dg, 'x')['a']
    assert isinstance(total, CovarianceVariable)
    assert float(total.variance) == pytest.approx(COV2.sum())
    assert isinstance(sc.mean(dg, 'x')['a'], CovarianceVariable)
    assert isinstance(sc.abs(dg)['a'], CovarianceVariable)
    assert isinstance(sc.concat([dg, dg], 'x')['a'], CovarianceVariable)


@pytest.mark.usefixtures('dispatch')
def test_free_functions_dispatch_directly(a):
    assert isinstance(sc.sum(a, 'x'), CovarianceVariable)
    assert float(sc.sum(a, 'x').variance) == pytest.approx(COV3.sum())


@pytest.mark.usefixtures('dispatch')
def test_lossy_free_functions_raise(a):
    with pytest.raises(CovarianceError):
        sc.max(a)
    with pytest.raises(CovarianceError):
        sc.max(sc.DataGroup({'a': a}))


@pytest.mark.usefixtures('dispatch')
def test_dispatch_leaves_plain_variables_alone():
    v = sc.array(dims=['x'], values=[1.0, 2.0, 3.0])
    assert float(sc.sum(v).value) == pytest.approx(6.0)
    assert type(sc.sum(v)) is sc.Variable
    assert float(sc.max(v).value) == pytest.approx(3.0)


def test_uninstall_dispatch_restores_scipp(a):
    install_dispatch()
    uninstall_dispatch()
    assert type(sc.sum(sc.DataGroup({'a': a}), 'x')['a']) is sc.Variable
