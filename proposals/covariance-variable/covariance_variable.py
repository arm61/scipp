# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Reference prototype of a ``scipp.Variable`` subclass carrying a full covariance.

This module is a *proposal prototype*, not part of the public scipp API.  See the
accompanying ``README.md`` for the design rationale and for the limits of the
approach.

The central invariant is::

    self.variances == diag(self.covariance)

The subclass keeps the inherited C++ ``variances`` buffer populated with the
diagonal of the covariance matrix at all times.  Any code path that falls
through to C++ therefore still sees *correct marginal variances*; it only loses
the off-diagonal correlations.  Operations that cannot propagate a covariance
raise instead of silently degrading, in the spirit of ADR 0015.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

import scipp as sc

__all__ = [
    'CovarianceError',
    'CovarianceVariable',
    'concat',
    'covariance_array',
    'covariance_scalar',
]

#: Suffix used to build the "mirror" dimension labels of the covariance matrix.
#: A variable with dims ``('x', 'y')`` has a covariance with dims
#: ``('x', 'y', "x'", "y'")``.  Scipp forbids repeated dimension labels, so the
#: two index axes of the matrix must be named differently.
MIRROR = "'"

_RAD = sc.scalar(1.0, unit='rad')


class CovarianceError(RuntimeError):
    """Raised when an operation cannot propagate a full covariance matrix."""


# --------------------------------------------------------------------------
# Dimension bookkeeping
# --------------------------------------------------------------------------


def _mirror(dims: Sequence[str]) -> tuple[str, ...]:
    return tuple(d + MIRROR for d in dims)


def _cov_dims(dims: Sequence[str]) -> tuple[str, ...]:
    return (*dims, *_mirror(dims))


def _check_dim_labels(dims: Iterable[str]) -> None:
    for d in dims:
        if d.endswith(MIRROR):
            raise CovarianceError(
                f"Dimension label {d!r} ends in {MIRROR!r}, which is reserved for "
                f"the mirror axes of the covariance matrix."
            )


def _canonical(
    cov: sc.Variable, dims: Sequence[str], shape: Sequence[int]
) -> sc.Variable:
    """Broadcast/transpose ``cov`` into the canonical layout for ``dims``."""
    full = _cov_dims(dims)
    sizes = dict(zip(full, (*shape, *shape), strict=True))
    order = [d for d in full if d in cov.dims]
    if tuple(cov.dims) != tuple(order):
        cov = cov.transpose(order)
    if tuple(cov.dims) != full:
        cov = sc.broadcast(cov, sizes=sizes).copy()
    return cov


def _matrix(cov: sc.Variable, n: int) -> np.ndarray:
    """View the canonical covariance as a plain ``(n, n)`` matrix."""
    return np.asarray(cov.values).reshape(n, n)


def _shares_buffer(a: sc.Variable, b: sc.Variable) -> bool:
    """Whether two variables alias the same data.

    Scipp's own C++ ``Variable::is_same`` -- which drives its correlation
    special-cases in ``lib/variable/arithmetic.cpp`` -- is not exposed to
    Python, so a Python-level subclass has to fall back on NumPy.
    """
    if a is b:
        return True
    try:
        return bool(np.shares_memory(a.values, b.values))
    except (TypeError, ValueError):
        return False


def _as_unit(unit: Any) -> sc.Unit:
    return unit if isinstance(unit, sc.Unit) else sc.Unit(unit)


def _values_only(var: sc.Variable) -> sc.Variable:
    """Strip variances, so derivative factors do not double-count uncertainty."""
    if isinstance(var, CovarianceVariable):
        var = var.to_variable()
    if var.variances is None:
        return var
    out = var.copy()
    sc.Variable.variances.fset(out, None)  # type: ignore[union-attr]
    return out


# --------------------------------------------------------------------------
# The subclass
# --------------------------------------------------------------------------


class CovarianceVariable(sc.Variable):
    """A ``scipp.Variable`` that stores and propagates a full covariance matrix.

    Instances *are* ``scipp.Variable`` objects as far as C++ is concerned: they
    carry values and variances, so ``isinstance(x, sc.Variable)`` holds and
    every existing scipp function accepts them.  In addition they carry a
    ``covariance``: a ``Variable`` with dims ``(*dims, *mirror(dims))`` and unit
    ``unit**2``, whose diagonal is kept equal to the inherited ``variances``.

    Parameters
    ----------
    dims:
        Dimension labels.
    values:
        Array of values.
    covariance:
        Covariance matrix, as a ``scipp.Variable`` with dims
        ``(*dims, *mirror(dims))``, an array-like of shape ``(*shape, *shape)``,
        or ``None`` for a zero covariance.
    unit:
        Physical unit of the values; the covariance carries ``unit**2``.
    dtype:
        Element type, deduced from ``values`` if omitted.

    Examples
    --------
    Two 50% correlated measurements:

      >>> a = covariance_array(
      ...     dims=['x'],
      ...     values=[1.0, 2.0],
      ...     covariance=[[0.04, 0.02], [0.02, 0.04]],
      ...     unit='m',
      ... )
      >>> a.variances
      array([0.04, 0.04])

    Summing correlated elements accounts for the off-diagonal terms, which
    plain scipp cannot do:

      >>> round(float(a.sum('x').variance), 12)
      0.12
    """

    def __init__(
        self,
        *,
        dims: Sequence[str],
        values: Any,
        covariance: Any = None,
        unit: Any = sc.units.dimensionless,
        dtype: Any = None,
    ) -> None:
        dims = tuple(dims)
        _check_dim_labels(dims)
        values = np.asarray(values, dtype=np.float64)
        cov = self._coerce_covariance(covariance, dims, values.shape, unit)
        n = int(np.prod(values.shape, dtype=int))
        variances = np.diag(_matrix(cov, n)).reshape(values.shape).copy()
        super().__init__(
            dims=dims,
            values=values,
            variances=variances,
            unit=unit,
            **({} if dtype is None else {'dtype': dtype}),
        )
        self._covariance = cov

    # -- construction --------------------------------------------------------

    @staticmethod
    def _coerce_covariance(
        covariance: Any, dims: tuple[str, ...], shape: tuple[int, ...], unit: Any
    ) -> sc.Variable:
        cov_dims = _cov_dims(dims)
        cov_shape = (*shape, *shape)
        cov_unit = _as_unit(unit) ** 2
        if covariance is None:
            return sc.zeros(dims=list(cov_dims), shape=list(cov_shape), unit=cov_unit)
        if isinstance(covariance, sc.Variable):
            cov = _values_only(covariance)
            if tuple(cov.dims) != cov_dims:
                cov = cov.transpose(list(cov_dims))
            if cov.unit != cov_unit:
                cov = cov.to(unit=cov_unit)
            return cov.copy()
        return sc.array(
            dims=list(cov_dims),
            values=np.asarray(covariance, dtype=np.float64).reshape(cov_shape),
            unit=cov_unit,
        )

    @classmethod
    def from_variable(
        cls, var: sc.Variable, covariance: Any = None
    ) -> CovarianceVariable:
        """Build a :class:`CovarianceVariable` from a plain ``Variable``.

        With ``covariance=None`` the variable's own ``variances`` become the
        diagonal, i.e. its elements are assumed uncorrelated -- exactly the
        assumption plain scipp makes implicitly.
        """
        if covariance is None and var.variances is not None:
            n = int(np.prod(var.shape, dtype=int))
            covariance = np.diag(np.asarray(var.variances).reshape(n)).reshape(
                (*var.shape, *var.shape)
            )
        return cls(
            dims=var.dims, values=var.values, covariance=covariance, unit=var.unit
        )

    @classmethod
    def _wrap(cls, base: sc.Variable, cov: sc.Variable) -> CovarianceVariable:
        """Attach ``cov`` to ``base``, the plain result of an operation."""
        cov = _canonical(cov, base.dims, base.shape)
        return cls(dims=base.dims, values=base.values, covariance=cov, unit=base.unit)

    # -- properties ----------------------------------------------------------

    @property
    def covariance(self) -> sc.Variable:
        """The full covariance matrix, with dims ``(*dims, *mirror(dims))``."""
        return self._covariance

    @covariance.setter
    def covariance(self, cov: Any) -> None:
        cov = self._coerce_covariance(
            cov, tuple(self.dims), tuple(self.shape), self.unit
        )
        self._covariance = cov
        sc.Variable.variances.fset(  # type: ignore[union-attr]
            self, np.diag(_matrix(cov, self._n)).reshape(self.shape).copy()
        )

    @property
    def correlation(self) -> sc.Variable:
        """The correlation matrix (dimensionless), with the covariance's dims."""
        n = self._n
        cov = _matrix(self._covariance, n)
        sigma = np.sqrt(np.diag(cov))
        outer = np.outer(sigma, sigma)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.where(outer == 0.0, 0.0, cov / outer)
        return sc.array(
            dims=list(self._covariance.dims),
            values=corr.reshape(self._covariance.shape),
            unit='dimensionless',
        )

    @property
    def _n(self) -> int:
        return int(np.prod(self.shape, dtype=int))

    @sc.Variable.variances.setter  # type: ignore[misc]
    def variances(self, _: Any) -> None:
        raise CovarianceError(
            "Cannot set 'variances' on a CovarianceVariable; they are the diagonal "
            "of 'covariance'. Assign to 'covariance' instead."
        )

    def to_variable(self) -> sc.Variable:
        """Explicitly drop the correlations, returning a plain ``Variable``.

        The variances of the result are the diagonal of the covariance, so
        marginal uncertainties are preserved.
        """
        return sc.array(
            dims=list(self.dims),
            values=self.values,
            variances=self.variances,
            unit=self.unit,
        )

    def __repr__(self) -> str:
        base = sc.Variable.__repr__(self).split('> ', 1)[1]
        corr = np.array2string(_matrix(self.correlation, self._n), precision=3)
        return f"<CovarianceVariable> {base}\n  correlation:\n{corr}"

    def _repr_html_(self) -> str:
        # scipp's variable_repr (src/scipp/visualization/formatting_html.py)
        # hard-codes a 'scipp.' prefix on the class name, which is wrong for a
        # class defined outside the scipp namespace.
        name = type(self).__name__
        return sc.Variable._repr_html_(self).replace(
            f"<div class='sc-obj-type'>scipp.{name} ",
            f"<div class='sc-obj-type'>{name} ",
            1,
        )

    # -- covariance algebra --------------------------------------------------

    @staticmethod
    def _sandwich(jacobian: sc.Variable, cov: sc.Variable) -> sc.Variable:
        """``J C J^T`` for a *diagonal* Jacobian whose diagonal is ``jacobian``.

        Because ``J`` is diagonal this reduces to an outer-product scaling.
        Broadcasting falls out for free: if ``jacobian`` has dims the covariance
        lacks, the result gains those dims on both index axes, which is exactly
        the correlation structure a broadcast introduces.
        """
        jacobian = _values_only(jacobian)
        rename = dict(zip(jacobian.dims, _mirror(jacobian.dims), strict=True))
        return jacobian * cov * jacobian.rename_dims(rename)

    @staticmethod
    def _covariance_of(other: Any) -> sc.Variable | None:
        """Covariance of an operand, or ``None`` if it is exact."""
        if isinstance(other, CovarianceVariable):
            return other.covariance
        if isinstance(other, sc.Variable):
            if other.variances is None:
                return None
            # A plain Variable asserts that its own elements are uncorrelated.
            n = int(np.prod(other.shape, dtype=int))
            return sc.array(
                dims=list(_cov_dims(other.dims)),
                values=np.diag(np.asarray(other.variances).reshape(n)).reshape(
                    (*other.shape, *other.shape)
                ),
                unit=other.unit**2,
            )
        return None

    def _binary(
        self,
        other: Any,
        base: sc.Variable,
        d_self: sc.Variable,
        d_other: sc.Variable | None,
    ) -> CovarianceVariable:
        """Assemble ``J C J^T`` for a binary op from its partial derivatives."""
        cov_other = self._covariance_of(other)
        if isinstance(other, sc.Variable) and _shares_buffer(other, self):
            # Same buffer on both sides: the total derivative is the sum of the
            # partials, so the operands stay perfectly correlated.
            if d_other is None:
                raise CovarianceError("Self-referencing operand without a derivative.")
            return self._wrap(base, self._sandwich(d_self + d_other, self.covariance))
        cov = self._sandwich(d_self, self.covariance)
        if cov_other is not None and d_other is not None:
            cov = cov + self._sandwich(d_other, cov_other)
        return self._wrap(base, cov)

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> CovarianceVariable:
        rhs = _as_operand(other)
        one = _unit_derivative(self, rhs)
        return self._binary(other, _values_only(self) + _values_only(rhs), one, one)

    __radd__ = __add__

    def __sub__(self, other: Any) -> CovarianceVariable:
        rhs = _as_operand(other)
        one = _unit_derivative(self, rhs)
        return self._binary(other, _values_only(self) - _values_only(rhs), one, -one)

    def __rsub__(self, other: Any) -> CovarianceVariable:
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> CovarianceVariable:
        rhs = _values_only(_as_operand(other))
        lhs = _values_only(self)
        return self._binary(other, lhs * rhs, rhs, lhs)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> CovarianceVariable:
        rhs = _values_only(_as_operand(other))
        lhs = _values_only(self)
        return self._binary(other, lhs / rhs, sc.reciprocal(rhs), -lhs / (rhs * rhs))

    def __rtruediv__(self, other: Any) -> CovarianceVariable:
        lhs = _values_only(_as_operand(other))
        rhs = _values_only(self)
        base = lhs / rhs
        # d(a/b)/db = -a/b**2 for self==b, d(a/b)/da = 1/b for the other operand.
        cov = self._sandwich(-lhs / (rhs * rhs), self.covariance)
        cov_other = self._covariance_of(other)
        if cov_other is not None:
            cov = cov + self._sandwich(sc.reciprocal(rhs), cov_other)
        return self._wrap(base, cov)

    def __neg__(self) -> CovarianceVariable:
        return self._wrap(-_values_only(self), self.covariance)

    def __pow__(self, exponent: Any) -> CovarianceVariable:
        if self._covariance_of(exponent) is not None:
            raise CovarianceError("Exponents with uncertainties are not supported.")
        e = float(exponent) if not isinstance(exponent, sc.Variable) else exponent.value
        values = _values_only(self)
        return self._chain(values**e, e * values ** (e - 1.0))

    # -- elementwise unary functions ----------------------------------------

    def _chain(self, base: sc.Variable, derivative: sc.Variable) -> CovarianceVariable:
        return self._wrap(base, self._sandwich(derivative, self.covariance))

    def sqrt(self) -> CovarianceVariable:
        root = sc.sqrt(_values_only(self))
        return self._chain(root, sc.reciprocal(2.0 * root))

    def exp(self) -> CovarianceVariable:
        e = sc.exp(_values_only(self))
        return self._chain(e, e)

    def log(self) -> CovarianceVariable:
        values = _values_only(self)
        return self._chain(sc.log(values), sc.reciprocal(values))

    def sin(self) -> CovarianceVariable:
        # d(sin x)/dx carries 1/rad: scipp's cos() returns a dimensionless
        # value, but the Jacobian must undo the rad of the covariance.
        values = _values_only(self)
        return self._chain(sc.sin(values), sc.cos(values) / _RAD)

    def cos(self) -> CovarianceVariable:
        values = _values_only(self)
        return self._chain(sc.cos(values), -sc.sin(values) / _RAD)

    # -- reductions ----------------------------------------------------------

    def sum(self, dim: str | Iterable[str] | None = None) -> CovarianceVariable:
        """Sum over ``dim``, accounting for correlations between the summands."""
        dims = self._reduction_dims(dim)
        cov = self.covariance
        base = _values_only(self)
        for d in dims:
            cov = cov.sum(d).sum(d + MIRROR)
            base = base.sum(d)
        return self._wrap(base, cov)

    def mean(self, dim: str | Iterable[str] | None = None) -> CovarianceVariable:
        dims = self._reduction_dims(dim)
        n = float(np.prod([self.sizes[d] for d in dims], dtype=int))
        total = self.sum(dims)
        return self._wrap(_values_only(total) / n, total.covariance / (n * n))

    def _reduction_dims(self, dim: str | Iterable[str] | None) -> list[str]:
        if dim is None:
            return list(self.dims)
        return [dim] if isinstance(dim, str) else list(dim)

    # -- shape and indexing --------------------------------------------------

    def __getitem__(self, key: Any) -> CovarianceVariable:
        dim, index = _normalize_index(self, key)
        base = _values_only(self)[dim, index]
        cov = self.covariance[dim, index][dim + MIRROR, index]
        return self._wrap(base, cov)

    def transpose(self, dims: Sequence[str] | None = None) -> CovarianceVariable:
        base = _values_only(self).transpose(None if dims is None else list(dims))
        return self._wrap(base, self.covariance)

    def broadcast(
        self,
        dims: Sequence[str] | None = None,
        shape: Sequence[int] | None = None,
        sizes: dict[str, int] | None = None,
    ) -> CovarianceVariable:
        """Broadcast, recording the correlations the broadcast introduces.

        Plain scipp raises ``VariancesError`` here (ADR 0015) because a
        broadcast duplicates an uncertain value, perfectly correlating the
        copies.  With a full covariance matrix those correlations are
        representable, so the operation becomes well defined.
        """
        if sizes is None:
            if dims is None or shape is None:
                raise TypeError("Provide either 'sizes' or both 'dims' and 'shape'.")
            sizes = dict(zip(dims, shape, strict=True))
        _check_dim_labels(sizes)
        base = sc.broadcast(_values_only(self), sizes=sizes).copy()
        # The Jacobian of a broadcast is a 0/1 selection matrix; through the
        # sandwich it is just a broadcast of ones.
        ones = sc.ones(
            dims=list(sizes), shape=list(sizes.values()), unit='dimensionless'
        )
        return self._wrap(base, self._sandwich(ones, self.covariance))

    def to(
        self, *, unit: Any = None, dtype: Any = None, copy: bool = True
    ) -> CovarianceVariable:
        base = _values_only(self).to(unit=unit, dtype=dtype, copy=copy)
        # _wrap re-coerces the covariance to ``base.unit**2``, which performs
        # the quadratic scaling exactly once.
        return self._wrap(base, self.covariance)

    def copy(self, deep: bool = True) -> CovarianceVariable:
        return self._wrap(self.to_variable().copy(deep=deep), self.covariance)

    def __copy__(self) -> CovarianceVariable:
        return self.copy(deep=False)

    def __deepcopy__(self, _: Any) -> CovarianceVariable:
        return self.copy(deep=True)


# --------------------------------------------------------------------------
# Operations that cannot preserve a covariance
# --------------------------------------------------------------------------

#: Inherited methods safe to leave alone: they do not produce a derived
#: variable whose correlations could be lost.
_SAFE_INHERITED = frozenset({'plot', 'underlying_size'})


def _unsupported_method(name: str) -> Any:
    def fail(self: Any, *_: Any, **__: Any) -> Any:
        raise CovarianceError(
            f"'{name}' does not propagate a covariance matrix. Call "
            f"'.to_variable()' first if dropping the correlations is intended."
        )

    fail.__name__ = name
    fail.__qualname__ = f'CovarianceVariable.{name}'
    return fail


def _install_unsupported_stubs() -> None:
    """Make every un-overridden ``Variable`` method fail loudly.

    Listing the unsupported operations by hand is a denylist, and anything left
    off it degrades *silently* to a plain ``Variable`` -- the exact failure mode
    this design exists to prevent. ``sc.DataGroup`` makes that easy to hit:
    its methods dispatch to the item's method (``operator.methodcaller`` in
    ``src/scipp/core/data_group.py``), so ``dg.squeeze()`` reaches
    ``Variable.squeeze`` and quietly drops the covariance.

    So the list is derived rather than written: every public ``sc.Variable``
    method this class does not explicitly implement gets a raising stub.
    Properties are left alone -- they return metadata, and scipp reads some of
    them (``bins``, ``dims``) while dispatching.
    """
    for name in dir(sc.Variable):
        if name.startswith('_') or name in _SAFE_INHERITED:
            continue
        if name in vars(CovarianceVariable):
            continue
        if isinstance(inspect.getattr_static(sc.Variable, name), property):
            continue
        if not callable(getattr(sc.Variable, name, None)):
            continue
        setattr(CovarianceVariable, name, _unsupported_method(name))


_install_unsupported_stubs()


# --------------------------------------------------------------------------
# Free helpers
# --------------------------------------------------------------------------


def _as_operand(other: Any) -> sc.Variable:
    if isinstance(other, CovarianceVariable):
        return other.to_variable()
    if isinstance(other, sc.Variable):
        return other
    return sc.scalar(float(other), unit='dimensionless')


def _unit_derivative(lhs: sc.Variable, rhs: sc.Variable) -> sc.Variable:
    """A derivative of 1, shaped so the sandwich reaches all output dims."""
    sizes = {**dict(lhs.sizes), **dict(rhs.sizes)}
    _check_dim_labels(sizes)
    return sc.ones(dims=list(sizes), shape=list(sizes.values()), unit='dimensionless')


def _normalize_index(var: sc.Variable, key: Any) -> tuple[str, Any]:
    if isinstance(key, tuple):
        dim, index = key
        return str(dim), index
    if var.ndim != 1:
        raise CovarianceError(
            "Implicit-dimension indexing requires a 1-D variable; "
            "use `var['dim', index]`."
        )
    return var.dims[0], key


def covariance_array(
    *,
    dims: Sequence[str],
    values: Any,
    covariance: Any = None,
    unit: Any = sc.units.dimensionless,
    dtype: Any = None,
) -> CovarianceVariable:
    """Create a :class:`CovarianceVariable`, mirroring :func:`scipp.array`."""
    return CovarianceVariable(
        dims=dims, values=values, covariance=covariance, unit=unit, dtype=dtype
    )


def covariance_scalar(
    value: float, *, variance: float = 0.0, unit: Any = sc.units.dimensionless
) -> CovarianceVariable:
    """Create a 0-D :class:`CovarianceVariable`, mirroring :func:`scipp.scalar`."""
    return CovarianceVariable(
        dims=(), values=np.asarray(value), covariance=np.asarray(variance), unit=unit
    )


def concat(variables: Sequence[CovarianceVariable], dim: str) -> CovarianceVariable:
    """Concatenate along ``dim``, producing a block-diagonal covariance.

    The inputs are assumed mutually independent, so the cross-blocks are zero.
    """
    variables = list(variables)
    base = sc.concat([_values_only(v) for v in variables], dim)
    n = int(np.prod(base.shape, dtype=int))
    out = np.zeros((n, n))
    offset = 0
    for v in variables:
        k = v._n
        out[offset : offset + k, offset : offset + k] = _matrix(v.covariance, k)
        offset += k
    cov = sc.array(
        dims=list(_cov_dims(base.dims)),
        values=out.reshape((*base.shape, *base.shape)),
        unit=base.unit**2,
    )
    return CovarianceVariable._wrap(base, cov)
