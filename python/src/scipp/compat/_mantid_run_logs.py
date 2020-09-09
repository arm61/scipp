from copy import deepcopy
import warnings
import numpy as np

from .._scipp import core as sc

_additional_unit_mapping = {
    "Kelvin": sc.units.K,
    "microsecond": sc.units.us,
    "nanosecond": sc.units.ns,
    "second": sc.units.s,
    "Angstrom": sc.units.angstrom,
    "Hz": sc.units.one / sc.units.s,
    "degree": sc.units.deg,
}


def make_variables_from_run_logs(ws):
    lookup_units = dict(
        zip([str(unit) for unit in sc.units.supported_units()],
            sc.units.supported_units()))
    lookup_units.update(_additional_unit_mapping)
    for property_name in ws.run().keys():
        units_string = ws.run()[property_name].units
        unit = lookup_units.get(units_string, None)
        values = deepcopy(ws.run()[property_name].value)

        if units_string and unit is None:
            warnings.warn(f"Workspace run log '{property_name}' "
                          f"has unrecognised units: '{units_string}'")
        if unit is None:
            unit = sc.units.one

        try:
            times = deepcopy(ws.run()[property_name].times)
            is_time_series = True
            dimension_label = "time"
        except AttributeError:
            times = None
            is_time_series = False
            dimension_label = property_name

        if np.isscalar(values):
            property_data = sc.Variable(value=values, unit=unit)
        else:
            property_data = sc.Variable(values=values,
                                        unit=unit,
                                        dims=[dimension_label])

        if is_time_series:
            # If property has timestamps, create a DataArray
            data_array = sc.DataArray(data=property_data,
                                      coords={
                                          dimension_label:
                                          sc.Variable([dimension_label],
                                                      values=times,
                                                      dtype=sc.dtype.int64,
                                                      unit=sc.units.ns)
                                      })
            yield property_name, sc.Variable(data_array)
        else:
            yield property_name, property_data
