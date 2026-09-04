# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the CovarianceDataArray prototype."""

import numpy as np
import pytest
from covariance_data_array import CovarianceDataArray
from covariance_variable import (
    CovarianceError,
    CovarianceVariable,
    covariance_array,
    install_dispatch,
    uninstall_dispatch,
)

import scipp as sc

COV3 = np.array([[0.04, 0.02, 0.00], [0.02, 0.09, -0.01], [0.00, -0.01, 0.16]])


@pytest.fixture
def da():
    cv = covariance_array(dims=['x'], values=[1.0, 2.0, 3.0], covariance=COV3, unit='m')
    return CovarianceDataArray(data=cv, coords={'x': sc.arange('x', 3.0)})


def cov_of(obj):
    n = int(np.prod(obj.shape, dtype=int))
    return np.asarray(obj.covariance.values).reshape(n, n)


# -- construction and invariants -------------------------------------------


def test_is_a_scipp_data_array(da):
    assert isinstance(da, sc.DataArray)
    assert da.dims == ('x',)
    assert list(da.coords.keys()) == ['x']


def test_data_comes_back_as_a_covariance_variable(da):
    assert isinstance(da.data, CovarianceVariable)
    np.testing.assert_allclose(cov_of(da), COV3)


def test_variances_are_the_diagonal(da):
    np.testing.assert_allclose(da.data.variances, np.diag(COV3))
    # The C++ side sees correct marginals even without the subclass.
    np.testing.assert_allclose(sc.DataArray.data.fget(da).variances, np.diag(COV3))


def test_from_plain_data_array_assumes_no_correlation():
    plain = sc.DataArray(
        data=sc.array(dims=['x'], values=[1.0, 2.0], variances=[0.1, 0.2]),
        coords={'x': sc.arange('x', 2.0)},
    )
    result = CovarianceDataArray.from_data_array(plain)
    np.testing.assert_allclose(cov_of(result), np.diag([0.1, 0.2]))


def test_to_data_array_preserves_marginals_and_coords(da):
    plain = da.to_data_array()
    assert type(plain) is sc.DataArray
    np.testing.assert_allclose(plain.data.variances, np.diag(COV3))
    assert list(plain.coords.keys()) == ['x']


def test_covariance_matches_data_covariance(da):
    assert sc.identical(da.covariance, da.data.covariance)


def test_covariance_accessor_does_not_expose_internal_state(da):
    """Writing through the accessor would otherwise leave the C++ side stale."""
    da.covariance.values[0, 0] = 999.0
    np.testing.assert_allclose(cov_of(da), COV3)
    np.testing.assert_allclose(sc.DataArray.data.fget(da).variances, np.diag(COV3))


def test_setting_data_updates_the_covariance(da):
    da.data = covariance_array(
        dims=['x'], values=[1.0, 1.0, 1.0], covariance=np.eye(3), unit='m'
    )
    np.testing.assert_allclose(cov_of(da), np.eye(3))


def test_html_repr_is_not_prefixed_with_scipp(da):
    assert 'scipp.CovarianceDataArray' not in da._repr_html_()


# -- arithmetic -------------------------------------------------------------


def test_addition_of_independent_operands(da):
    other = CovarianceDataArray(
        data=covariance_array(
            dims=['x'], values=[1.0, 1.0, 1.0], covariance=np.eye(3) * 0.01, unit='m'
        ),
        coords={'x': sc.arange('x', 3.0)},
    )
    result = da + other
    np.testing.assert_allclose(result.data.values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(cov_of(result), COV3 + np.eye(3) * 0.01)


def test_self_addition_is_perfectly_correlated(da):
    """`.data` rebuilds a new object each access, so aliasing needs care."""
    np.testing.assert_allclose(cov_of(da + da), 4 * COV3)


def test_self_subtraction_is_exactly_zero(da):
    result = da - da
    np.testing.assert_allclose(result.data.values, 0.0)
    np.testing.assert_allclose(cov_of(result), 0.0, atol=1e-15)


def test_scalar_multiplication(da):
    np.testing.assert_allclose(cov_of(da * 3.0), 9 * COV3)
    np.testing.assert_allclose(cov_of(3.0 * da), 9 * COV3)


def test_negation_and_abs(da):
    np.testing.assert_allclose(cov_of(-da), COV3)
    np.testing.assert_allclose(abs(-da).data.values, [1.0, 2.0, 3.0])


def test_in_place_keeps_the_invariant(da):
    da += da
    assert isinstance(da, CovarianceDataArray)
    np.testing.assert_allclose(da.data.variances, np.diag(cov_of(da)))


def test_coords_are_preserved_through_operations(da):
    assert list((da + da).coords.keys()) == ['x']
    assert list((da * 2.0).coords.keys()) == ['x']


# -- reductions -------------------------------------------------------------


def test_sum_accounts_for_off_diagonal_terms(da):
    total = da.sum('x')
    assert isinstance(total, CovarianceDataArray)
    assert float(total.data.variance) == pytest.approx(COV3.sum())
    # Plain scipp underestimates by dropping the off-diagonals.
    assert float(da.to_data_array().sum('x').data.variance) == pytest.approx(
        np.trace(COV3)
    )


def test_mean_scales_by_n_squared(da):
    assert float(da.mean('x').data.variance) == pytest.approx(COV3.sum() / 9)


def test_reduction_drops_the_reduced_coord(da):
    assert list(da.sum('x').coords.keys()) == []


# -- indexing ---------------------------------------------------------------


def test_positional_slice_selects_both_axes(da):
    s = da['x', 0:2]
    assert isinstance(s, CovarianceDataArray)
    np.testing.assert_allclose(cov_of(s), COV3[:2, :2])
    np.testing.assert_allclose(s.coords['x'].values, [0.0, 1.0])


def test_integer_index_gives_a_scalar(da):
    s = da['x', 1]
    assert s.data.dims == ()
    assert float(s.data.variance) == pytest.approx(COV3[1, 1])


def test_label_based_slice_stays_consistent(da):
    """The probe-array trick keeps label indexing in step with the base class."""
    s = da['x', sc.scalar(1.0) : sc.scalar(3.0)]
    np.testing.assert_allclose(cov_of(s), COV3[1:, 1:])


def test_transpose_reorders_both_axes():
    values = np.arange(6.0).reshape(2, 3)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(6, 6))
    cov = (m @ m.T).reshape(2, 3, 2, 3)
    cv = covariance_array(dims=['x', 'y'], values=values, covariance=cov, unit='m')
    da = CovarianceDataArray(data=cv)
    t = da.transpose(['y', 'x'])
    assert t.dims == ('y', 'x')
    np.testing.assert_allclose(cov_of(t), cov.transpose(1, 0, 3, 2).reshape(6, 6))


def test_copy_is_independent(da):
    c = da.copy()
    c.data = covariance_array(
        dims=['x'], values=[0.0, 0.0, 0.0], covariance=np.zeros((3, 3)), unit='m'
    )
    np.testing.assert_allclose(cov_of(da), COV3)


# -- containers -------------------------------------------------------------


def test_data_group_preserves_it(da):
    dg = sc.DataGroup({'a': da})
    assert dg['a'] is da
    assert isinstance(dg['a'].data, CovarianceVariable)


def test_dataset_still_strips_it(da):
    """Dataset stores C++ DataArrays, so the subclass cannot survive."""
    assert type(sc.Dataset({'a': da})['a']) is sc.DataArray


@pytest.mark.usefixtures('_dispatch')
def test_free_functions_dispatch(da):
    dg = sc.DataGroup({'a': da})
    total = sc.sum(dg, 'x')['a']
    assert isinstance(total, CovarianceDataArray)
    assert float(total.data.variance) == pytest.approx(COV3.sum())
    assert isinstance(sc.sum(da, 'x'), CovarianceDataArray)


@pytest.fixture
def _dispatch():
    install_dispatch()
    yield
    uninstall_dispatch()


# -- refusals ---------------------------------------------------------------


@pytest.mark.parametrize('name', ['flatten', 'hist', 'squeeze', 'bin'])
def test_unsupported_operations_raise(da, name):
    with pytest.raises(CovarianceError):
        getattr(da, name)()


def test_every_inherited_data_array_method_is_accounted_for():
    import inspect

    from covariance_data_array import _SAFE_INHERITED

    missing = [
        name
        for name in dir(sc.DataArray)
        if not name.startswith('_')
        and name not in _SAFE_INHERITED
        and not isinstance(inspect.getattr_static(sc.DataArray, name), property)
        and callable(getattr(sc.DataArray, name, None))
        and name not in vars(CovarianceDataArray)
    ]
    assert missing == []
