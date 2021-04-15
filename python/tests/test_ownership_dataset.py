# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
from copy import copy, deepcopy

import numpy as np
import scipp as sc


def test_own_darr_set():
    # Data and metadata are shared
    v = sc.array(dims=['x'], values=[10, 20], unit='m')
    c = sc.array(dims=['x'], values=[1, 2], unit='s')
    a = sc.array(dims=['x'], values=[100, 200])
    m = sc.array(dims=['x'], values=[True, False])
    da = sc.DataArray(v, coords={'x': c}, attrs={'a': a}, masks={'m': m})
    da['x', 0] = -10
    da.data['x', 1] = -20
    da.coords['x']['x', 0] = -1
    da.attrs['a']['x', 0] = -100
    da.masks['m']['x', 0] = False
    c['x', 1] = -2
    a['x', 1] = -200
    m['x', 1] = True
    da.unit = 'kg'
    da.coords['x'].unit = 'J'
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[-10, -20], unit='kg'),
                                         coords={'x': sc.array(dims=['x'], values=[-1, -2], unit='J')},
                                         attrs={'a': sc.array(dims=['x'], values=[-100, -200])},
                                         masks={'m': sc.array(dims=['x'], values=[False, True])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[-10, -20], unit='kg'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))
    assert sc.identical(m, sc.array(dims=['x'], values=[False, True]))

    # TODO intentional?
    # Assignments overwrite data but not metadata.
    da.data = sc.array(dims=['x'], values=[11, 22], unit='m')
    da.coords['x'] = sc.array(dims=['x'], values=[3, 4], unit='s')
    da.attrs['a'] = sc.array(dims=['x'], values=[300, 400])
    da.masks['m'] = sc.array(dims=['x'], values=[True, True])
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[11, 22], unit='m'),
                                         coords={'x': sc.array(dims=['x'], values=[3, 4], unit='s')},
                                         attrs={'a': sc.array(dims=['x'], values=[300, 400])},
                                         masks={'m': sc.array(dims=['x'], values=[True, True])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[11, 22], unit='m'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))
    assert sc.identical(m, sc.array(dims=['x'], values=[False, True]))


def test_own_darr_get():
    # Data and metadata are shared.
    da = sc.DataArray(sc.array(dims=['x'], values=[10, 20], unit='m'),
                      coords={'x': sc.array(dims=['x'], values=[1, 2], unit='s')},
                      attrs={'a': sc.array(dims=['x'], values=[100, 200])},
                      masks={'m': sc.array(dims=['x'], values=[True, False])})
    v = da.data
    c = da.coords['x']
    a = da.attrs['a']
    m = da.masks['m']
    da['x', 0] = -10
    da.data['x', 1] = -20
    da.coords['x']['x', 0] = -1
    da.attrs['a']['x', 0] = -100
    da.masks['m']['x', 0] = False
    c['x', 1] = -2
    a['x', 1] = -200
    m['x', 1] = True
    da.unit = 'kg'
    da.coords['x'].unit = 'J'
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[-10, -20], unit='kg'),
                                         coords={'x': sc.array(dims=['x'], values=[-1, -2], unit='J')},
                                         attrs={'a': sc.array(dims=['x'], values=[-100, -200])},
                                         masks={'m': sc.array(dims=['x'], values=[False, True])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[-10, -20], unit='kg'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))
    assert sc.identical(m, sc.array(dims=['x'], values=[False, True]))

    # TODO intentional?
    # Assignments overwrite data but not coords.
    da.data = sc.array(dims=['x'], values=[11, 22], unit='m')
    da.coords['x'] = sc.array(dims=['x'], values=[3, 4], unit='s')
    da.attrs['a'] = sc.array(dims=['x'], values=[300, 400])
    da.masks['m'] = sc.array(dims=['x'], values=[True, True])
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[11, 22], unit='m'),
                                         coords={'x': sc.array(dims=['x'], values=[3, 4], unit='s')},
                                         attrs={'a': sc.array(dims=['x'], values=[300, 400])},
                                         masks={'m': sc.array(dims=['x'], values=[True, True])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[11, 22], unit='m'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))
    assert sc.identical(m, sc.array(dims=['x'], values=[False, True]))


def test_own_darr_get_meta():
    # Data and metadata are shared.
    da = sc.DataArray(sc.array(dims=['x'], values=[10, 20], unit='m'),
                      coords={'x': sc.array(dims=['x'], values=[1, 2], unit='s')},
                      attrs={'a': sc.array(dims=['x'], values=[100, 200])})
    v = da.data
    c = da.meta['x']
    a = da.meta['a']
    da['x', 0] = -10
    da.data['x', 1] = -20
    da.coords['x']['x', 0] = -1
    da.attrs['a']['x', 0] = -100
    c['x', 1] = -2
    a['x', 1] = -200
    da.unit = 'kg'
    da.coords['x'].unit = 'J'
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[-10, -20], unit='kg'),
                                         coords={'x': sc.array(dims=['x'], values=[-1, -2], unit='J')},
                                         attrs={'a': sc.array(dims=['x'], values=[-100, -200])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[-10, -20], unit='kg'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))

    # TODO intentional?
    # Assignments overwrite data but not coords.
    da.data = sc.array(dims=['x'], values=[11, 22], unit='m')
    da.coords['x'] = sc.array(dims=['x'], values=[3, 4], unit='s')
    da.attrs['a'] = sc.array(dims=['x'], values=[300, 400])
    assert sc.identical(da, sc.DataArray(sc.array(dims=['x'], values=[11, 22], unit='m'),
                                         coords={'x': sc.array(dims=['x'], values=[3, 4], unit='s')},
                                         attrs={'a': sc.array(dims=['x'], values=[300, 400])}))
    assert sc.identical(v, sc.array(dims=['x'], values=[11, 22], unit='m'))
    assert sc.identical(c, sc.array(dims=['x'], values=[-1, -2], unit='J'))
    assert sc.identical(a, sc.array(dims=['x'], values=[-100, -200]))
