# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""A ``scipp.DataArray`` subclass that carries a :class:`CovarianceVariable`.

``sc.DataArray`` stores its data as a C++ ``std::shared_ptr<Variable>``
(``lib/dataset/include/scipp/dataset/data_array.h:105``), so a
``CovarianceVariable`` put into one is stripped on insertion. This subclass
works around that the same way :class:`CovarianceVariable` works around
``Variable``: the C++ side keeps values and variances, the covariance lives in
the Python instance, and ``.data`` reassembles the two on access.

Coords and masks are ordinary variables and are handled by the base class
unchanged.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from covariance_variable import (
    CovarianceError,
    CovarianceVariable,
    _cov_dims,
    _matrix,
    _values_only,
)

import scipp as sc

__all__ = ['CovarianceDataArray', 'covariance_data_array']


def _plain_data(obj: Any) -> Any:
    """The C++-side data of an operand, with no covariance attached."""
    if isinstance(obj, CovarianceDataArray):
        return obj.to_data_array()
    if isinstance(obj, CovarianceVariable):
        return obj.to_variable()
    return obj


def _covariance_data(obj: Any) -> Any:
    """The covariance-carrying data of an operand, for the algebra."""
    if isinstance(obj, CovarianceDataArray):
        return obj.data
    if isinstance(obj, sc.DataArray):
        return obj.data
    return obj


def _same_data(a: Any, b: Any) -> bool:
    """Whether two data arrays are backed by the same buffer.

    ``.data`` rebuilds a new ``CovarianceVariable`` on every access, so the
    aliasing check inside :class:`CovarianceVariable` cannot see that ``da + da``
    has the same operand twice. Compare the underlying C++ variables instead.
    """
    if a is b:
        return True
    if not isinstance(a, sc.DataArray) or not isinstance(b, sc.DataArray):
        return False
    try:
        return bool(
            np.shares_memory(
                sc.DataArray.data.fget(a).values,  # type: ignore[union-attr]
                sc.DataArray.data.fget(b).values,  # type: ignore[union-attr]
            )
        )
    except (TypeError, ValueError):
        return False


class CovarianceDataArray(sc.DataArray):
    """A ``scipp.DataArray`` whose data carries a full covariance matrix.

    Parameters
    ----------
    data:
        The data. A :class:`CovarianceVariable` supplies its own covariance; a
        plain ``Variable`` is taken as uncorrelated unless ``covariance`` is
        given.
    coords, masks, name:
        As for ``scipp.DataArray``.
    covariance:
        Covariance matrix, overriding one carried by ``data``.

    Examples
    --------
      >>> import numpy as np
      >>> from covariance_variable import covariance_array
      >>> cv = covariance_array(
      ...     dims=['x'], values=[1.0, 2.0], covariance=[[0.04, 0.02], [0.02, 0.09]]
      ... )
      >>> da = CovarianceDataArray(data=cv, coords={'x': sc.arange('x', 2.0)})
      >>> type(da.data).__name__
      'CovarianceVariable'
      >>> round(float(da.sum('x').data.variance), 12)
      0.17

    Notes
    -----
    ``.data`` rebuilds the ``CovarianceVariable`` on each access, which copies
    the ``N x N`` covariance. Bind it to a local name in hot loops.
    """

    def __init__(
        self,
        data: sc.Variable,
        coords: Mapping[str, sc.Variable] | None = None,
        masks: Mapping[str, sc.Variable] | None = None,
        name: str = '',
        covariance: Any = None,
    ) -> None:
        if covariance is not None:
            cv = CovarianceVariable.from_variable(_plain_data(data), covariance)
        elif isinstance(data, CovarianceVariable):
            cv = data
        else:
            cv = CovarianceVariable.from_variable(data)
        super().__init__(
            data=cv.to_variable(),
            coords=dict(coords or {}),
            masks=dict(masks or {}),
            name=name,
        )
        self._covariance = cv.covariance

    # -- construction --------------------------------------------------------

    @classmethod
    def _from_base(cls, base: sc.DataArray, cov: sc.Variable) -> CovarianceDataArray:
        """Attach ``cov`` to ``base``, the plain result of an operation."""
        return cls(
            data=_values_only(base.data),
            coords=dict(base.coords),
            masks=dict(base.masks),
            name=base.name,
            covariance=cov,
        )

    @classmethod
    def from_data_array(
        cls, da: sc.DataArray, covariance: Any = None
    ) -> CovarianceDataArray:
        """Build from a plain ``DataArray``, assuming no correlation by default."""
        return cls(
            data=da.data,
            coords=dict(da.coords),
            masks=dict(da.masks),
            name=da.name,
            covariance=covariance,
        )

    def to_data_array(self) -> sc.DataArray:
        """Drop the correlations, returning a plain ``DataArray``.

        The variances of the result are the diagonal of the covariance, so
        marginal uncertainties are preserved.
        """
        return sc.DataArray(
            data=sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            coords=dict(self.coords),
            masks=dict(self.masks),
            name=self.name,
        )

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> CovarianceVariable:
        """The data, as a :class:`CovarianceVariable`."""
        return CovarianceVariable._wrap(
            sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            self._covariance,
        )

    @data.setter
    def data(self, value: sc.Variable) -> None:
        cv = (
            value
            if isinstance(value, CovarianceVariable)
            else CovarianceVariable.from_variable(value)
        )
        sc.DataArray.data.fset(self, cv.to_variable())  # type: ignore[union-attr]
        self._covariance = cv.covariance

    @property
    def covariance(self) -> sc.Variable:
        """The full covariance matrix of the data."""
        return self._covariance

    @covariance.setter
    def covariance(self, cov: Any) -> None:
        self.data = CovarianceVariable.from_variable(
            sc.DataArray.data.fget(self),  # type: ignore[union-attr]
            cov,
        )

    @property
    def correlation(self) -> sc.Variable:
        """The correlation matrix of the data."""
        return self.data.correlation

    def __repr__(self) -> str:
        return f"<CovarianceDataArray>\n{sc.DataArray.__repr__(self)}"

    def _repr_html_(self) -> str:
        name = type(self).__name__
        return sc.DataArray._repr_html_(self).replace(
            f"<div class='sc-obj-type'>scipp.{name} ",
            f"<div class='sc-obj-type'>{name} ",
            1,
        )

    # -- arithmetic ----------------------------------------------------------

    def _binary(self, other: Any, name: str) -> CovarianceDataArray:
        # The base class handles coord alignment and mask merging.
        base = getattr(sc.DataArray, name)(self.to_data_array(), _plain_data(other))
        left = self.data
        # Pass the *same* object on both sides when the operands alias, so the
        # covariance algebra sees them as perfectly correlated rather than
        # independent.
        right = left if _same_data(other, self) else _covariance_data(other)
        data = getattr(left, name)(right)
        return type(self)._from_base(base, data.covariance)

    def __add__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__add__')

    def __radd__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__add__')

    def __sub__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__sub__')

    def __rsub__(self, other: Any) -> CovarianceDataArray:
        return (-self).__add__(other)

    def __mul__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__mul__')

    def __rmul__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__mul__')

    def __truediv__(self, other: Any) -> CovarianceDataArray:
        return self._binary(other, '__truediv__')

    def __neg__(self) -> CovarianceDataArray:
        return type(self)._from_base(-self.to_data_array(), self.covariance)

    def __abs__(self) -> CovarianceDataArray:
        data = abs(self.data)
        return type(self)._from_base(abs(self.to_data_array()), data.covariance)

    def __pow__(self, other: Any) -> CovarianceDataArray:
        data = self.data**other
        return type(self)._from_base(self.to_data_array() ** other, data.covariance)

    def __iadd__(self, other: Any) -> CovarianceDataArray:  # noqa: PYI034
        return self.__add__(other)

    def __isub__(self, other: Any) -> CovarianceDataArray:  # noqa: PYI034
        return self.__sub__(other)

    def __imul__(self, other: Any) -> CovarianceDataArray:  # noqa: PYI034
        return self.__mul__(other)

    def __itruediv__(self, other: Any) -> CovarianceDataArray:  # noqa: PYI034
        return self.__truediv__(other)

    # -- reductions ----------------------------------------------------------

    def sum(self, dim: Any = None) -> CovarianceDataArray:
        return self._reduce('sum', dim)

    def mean(self, dim: Any = None) -> CovarianceDataArray:
        return self._reduce('mean', dim)

    def _reduce(self, name: str, dim: Any) -> CovarianceDataArray:
        plain = self.to_data_array()
        base = getattr(plain, name)() if dim is None else getattr(plain, name)(dim)
        data = getattr(self.data, name)(dim)
        return type(self)._from_base(base, data.covariance)

    # -- shape and indexing --------------------------------------------------

    def __getitem__(self, key: Any) -> CovarianceDataArray:
        """Slice, selecting the same elements on both covariance axes.

        The selected positions are found by slicing a probe array of flat
        indices with the *same* key, so positional, label-based and boolean
        indexing all stay consistent with the base class.
        """
        base = self.to_data_array()[key]
        n = int(np.prod(self.shape, dtype=int))
        probe = sc.DataArray(
            data=sc.array(
                dims=list(self.dims),
                values=np.arange(n, dtype=np.float64).reshape(self.shape),
            ),
            coords=dict(self.coords),
            masks=dict(self.masks),
        )
        index = np.asarray(probe[key].data.values).astype(int).ravel()
        selected = _matrix(self._covariance, n)[np.ix_(index, index)]
        cov = sc.array(
            dims=list(_cov_dims(base.dims)),
            values=selected.reshape((*base.shape, *base.shape)),
            unit=self.unit**2,
        )
        return type(self)._from_base(base, cov)

    def transpose(self, dims: Sequence[str] | None = None) -> CovarianceDataArray:
        base = self.to_data_array().transpose(None if dims is None else list(dims))
        return type(self)._from_base(base, self.data.transpose(dims).covariance)

    def to(self, **kwargs: Any) -> CovarianceDataArray:
        base = self.to_data_array().to(**kwargs)
        return type(self)._from_base(base, self.data.to(**kwargs).covariance)

    def copy(self, deep: bool = True) -> CovarianceDataArray:
        return type(self)._from_base(
            self.to_data_array().copy(deep=deep), self._covariance.copy(deep=deep)
        )

    def __copy__(self) -> CovarianceDataArray:
        return self.copy(deep=False)

    def __deepcopy__(self, _: Any) -> CovarianceDataArray:
        return self.copy(deep=True)


# --------------------------------------------------------------------------
# Operations that cannot preserve a covariance
# --------------------------------------------------------------------------

#: Inherited members safe to leave alone: they return metadata, a plot, or a
#: plain view that carries no correlations by design.
_SAFE_INHERITED = frozenset(
    {
        'plot',
        'underlying_size',
        'drop_coords',
        'drop_masks',
        'assign_coords',
        'assign_masks',
        'rename',
        'rename_dims',
    }
)


def _unsupported_method(name: str) -> Any:
    def fail(self: Any, *_: Any, **__: Any) -> Any:
        raise CovarianceError(
            f"'{name}' does not propagate a covariance matrix. Call "
            f"'.to_data_array()' first if dropping the correlations is intended."
        )

    fail.__name__ = name
    fail.__qualname__ = f'CovarianceDataArray.{name}'
    return fail


def _install_unsupported_stubs() -> None:
    """Make every un-overridden ``DataArray`` method fail loudly.

    Same reasoning as in ``covariance_variable``: a hand-written denylist lets
    anything omitted degrade silently to a plain ``DataArray``.
    """
    for name in dir(sc.DataArray):
        if name.startswith('_') or name in _SAFE_INHERITED:
            continue
        if name in vars(CovarianceDataArray):
            continue
        if isinstance(inspect.getattr_static(sc.DataArray, name), property):
            continue
        if not callable(getattr(sc.DataArray, name, None)):
            continue
        setattr(CovarianceDataArray, name, _unsupported_method(name))


_install_unsupported_stubs()

for _op in ('__floordiv__', '__mod__', '__and__', '__or__', '__xor__'):
    setattr(CovarianceDataArray, _op, _unsupported_method(_op))
del _op


@functools.wraps(CovarianceDataArray.__init__)
def covariance_data_array(*args: Any, **kwargs: Any) -> CovarianceDataArray:
    """Create a :class:`CovarianceDataArray`."""
    return CovarianceDataArray(*args, **kwargs)
