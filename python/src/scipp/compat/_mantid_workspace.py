import numpy as np
import re
from copy import deepcopy
from .._scipp import core as sc
from ._mantid_run_logs import make_variables_from_run_logs
from ._mantid_detectors import get_detector_properties, make_detector_info
from ._mantid_algorithm import run_mantid_alg


def _get_dtype_from_values(values, coerce_floats_to_ints):
    if coerce_floats_to_ints and np.all(np.mod(values, 1.0) == 0.0):
        dtype = sc.dtype.int32
    elif hasattr(values, 'dtype'):
        dtype = values.dtype
    else:
        if len(values) > 0:
            dtype = type(values[0])
            if dtype is str:
                dtype = sc.dtype.string
            elif dtype is int:
                dtype = sc.dtype.int64
            elif dtype is float:
                dtype = sc.dtype.float64
            else:
                raise RuntimeError("Cannot handle the dtype that this "
                                   "workspace has on Axis 1.")
        else:
            raise RuntimeError("Axis 1 of this workspace has no values. "
                               "Cannot determine dtype.")
    return dtype


def init_spec_axis(ws):
    axis = ws.getAxis(1)
    dim, unit = validate_and_get_unit(axis.getUnit().unitID())
    values = axis.extractValues()
    dtype = _get_dtype_from_values(values, dim == 'spectrum')
    return dim, sc.Variable([dim], values=values, unit=unit, dtype=dtype)


def set_bin_masks(bin_masks, dim, index, masked_bins):
    for masked_bin in masked_bins:
        bin_masks['spectrum', index][dim, masked_bin].value = True


def validate_and_get_unit(unit, allow_empty=False):
    known_units = {
        "DeltaE": ['Delta-E', sc.units.meV],
        "TOF": ['tof', sc.units.us],
        "Wavelength": ['wavelength', sc.units.angstrom],
        "Energy": ['energy', sc.units.meV],
        "dSpacing": ['d-spacing', sc.units.angstrom],
        "MomentumTransfer": ['Q', sc.units.dimensionless / sc.units.angstrom],
        "QSquared": [
            'Q^2',
            sc.units.dimensionless / (sc.units.angstrom * sc.units.angstrom)
        ],
        "Label": ['spectrum', sc.units.dimensionless],
        "Empty": ['empty', sc.units.dimensionless],
        "Counts": ['counts', sc.units.counts]
    }

    if unit not in known_units.keys():
        if allow_empty:
            return ['unknown', sc.units.dimensionless]
        else:
            raise RuntimeError("Unit not currently supported."
                               "Possible values are: {}, "
                               "got '{}'. ".format(
                                   [k for k in known_units.keys()], unit))
    else:
        return known_units[unit]


def _get_pos(pos):
    return [pos.X(), pos.Y(), pos.Z()]


def make_sample(ws):
    return sc.Variable(value=deepcopy(ws.sample()))


def make_component_info(ws):
    component_info = ws.componentInfo()

    if component_info.hasSource():
        source_pos = component_info.sourcePosition()
    else:
        source_pos = None

    if component_info.hasSample():
        sample_pos = component_info.samplePosition()
    else:
        sample_pos = None

    def as_var(pos):
        if pos is None:
            return pos
        return sc.Variable(value=np.array(_get_pos(pos)),
                           dtype=sc.dtype.vector_3_float64,
                           unit=sc.units.m)

    return as_var(source_pos), as_var(sample_pos)


def md_dimension(mantid_dim):
    # Look for q dimensions
    patterns = ["^q.*{0}$".format(coord) for coord in ['x', 'y', 'z']]
    q_dims = ['Q_x', 'Q_y', 'Q_z']
    pattern_result = zip(patterns, q_dims)
    if mantid_dim.getMDFrame().isQ():
        for pattern, result in pattern_result:
            if re.search(pattern, mantid_dim.name, re.IGNORECASE):
                return result

    # Look for common/known mantid dimensions
    patterns = ["DeltaE", "T"]
    dims = ['Delta-E', 'temperature']
    pattern_result = zip(patterns, dims)
    for pattern, result in pattern_result:
        if re.search(pattern, mantid_dim.name, re.IGNORECASE):
            return result

    # Look for common spacial dimensions
    patterns = ["^{0}$".format(coord) for coord in ['x', 'y', 'z']]
    dims = ['x', 'y', 'z']
    pattern_result = zip(patterns, dims)
    for pattern, result in pattern_result:
        if re.search(pattern, mantid_dim.name, re.IGNORECASE):
            return result

    raise ValueError(
        "Cannot infer scipp dimension from input mantid dimension {}".format(
            mantid_dim.name()))


def md_unit(frame):
    known_md_units = {
        "Angstrom^-1": sc.units.dimensionless / sc.units.angstrom,
        "r.l.u": sc.units.dimensionless,
        "T": sc.units.K,
        "DeltaE": sc.units.meV
    }
    if frame.getUnitLabel().ascii() in known_md_units:
        return known_md_units[frame.getUnitLabel().ascii()]
    else:
        return sc.units.dimensionless


def _convert_MatrixWorkspace_info(ws,
                                  advanced_geometry=False,
                                  load_run_logs=True):
    common_bins = ws.isCommonBins()
    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit().unitID())
    source_pos, sample_pos = make_component_info(ws)
    pos, rot, shp = get_detector_properties(
        ws, source_pos, sample_pos, advanced_geometry=advanced_geometry)
    spec_dim, spec_coord = init_spec_axis(ws)

    if common_bins:
        coord = sc.Variable([dim], values=ws.readX(0), unit=unit)
    else:
        coord = sc.Variable([spec_dim, dim], values=ws.extractX(), unit=unit)

    info = {
        "coords": {
            dim: coord,
            spec_dim: spec_coord
        },
        "masks": {},
        "unaligned_coords": {
            "sample":
            make_sample(ws),
            "instrument-name":
            sc.Variable(
                value=ws.componentInfo().name(ws.componentInfo().root()))
        },
    }

    if load_run_logs:
        for log_name, log_variable in make_variables_from_run_logs(ws):
            info["unaligned_coords"][log_name] = log_variable

    if advanced_geometry:
        info["coords"]["detector-info"] = make_detector_info(ws)

    if not np.all(np.isnan(pos.values)):
        info["coords"].update({"position": pos})

    if rot is not None and shp is not None and not np.all(np.isnan(
            pos.values)):
        info["unaligned_coords"].update({"rotation": rot, "shape": shp})

    if source_pos is not None:
        info["coords"]["source-position"] = source_pos

    if sample_pos is not None:
        info["coords"]["sample-position"] = sample_pos

    if ws.detectorInfo().hasMaskedDetectors():
        spectrum_info = ws.spectrumInfo()
        mask = np.array([
            spectrum_info.isMasked(i) for i in range(ws.getNumberHistograms())
        ])
        info["masks"]["spectrum"] = sc.Variable([spec_dim], values=mask)
    return info


def convert_monitors_ws(ws, converter, **ignored):
    validate_and_get_unit(ws.getAxis(0).getUnit().unitID())
    spec_dim, spec_coord = init_spec_axis(ws)
    spec_info = ws.spectrumInfo()
    comp_info = ws.componentInfo()
    monitors = []
    indexes = (ws.getIndexFromSpectrumNumber(int(i))
               for i in spec_coord.values)
    for index in indexes:
        definition = spec_info.getSpectrumDefinition(index)
        if not definition.size() == 1:
            raise RuntimeError("Cannot deal with grouped monitor detectors")
        det_index = definition[0][0]  # Ignore time index
        # We only ExtractSpectra for compatibility with
        # existing convert_Workspace2D_to_dataarray. This could instead be
        # refactored if found to be slow
        with run_mantid_alg('ExtractSpectra',
                            InputWorkspace=ws,
                            WorkspaceIndexList=[index]) as monitor_ws:
            # Run logs are already loaded in the data workspace
            single_monitor = converter(monitor_ws, load_run_logs=False)
        # Remove redundant information that is duplicated from workspace
        # We get this extra information from the generic converter reuse
        del single_monitor.coords['sample-position']
        if 'detector-info' in single_monitor.coords:
            del single_monitor.coords['detector-info']
        del single_monitor.unaligned_coords['sample']
        monitors.append((comp_info.name(det_index), single_monitor))
    return monitors


def load_component_info(ds, file, advanced_geometry=False):
    """
    Adds the component info coord into the dataset. The following are added:

    - source-position
    - sample-position
    - detector positions
    - detector rotations
    - detector shapes

    :param ds: Dataset on which the component info will be added as coords.
    :param file: File from which the IDF will be loaded.
                 This can be anything that mantid.Load can load.
    :param bool advanced_geometry: If True, load the full detector geometry
                                   including shapes and rotations. The
                                   positions of grouped detectors are
                                   spherically averaged. If False,
                                   load only the detector position, and return
                                   the cartesian average of the grouped
                                   detector positions.
    """
    with run_mantid_alg('Load', file) as ws:
        source_pos, sample_pos = make_component_info(ws)

        ds.coords["source-position"] = source_pos
        ds.coords["sample-position"] = sample_pos
        pos, rot, shp = get_detector_properties(
            ws, source_pos, sample_pos, advanced_geometry=advanced_geometry)
        ds.coords["position"] = pos
        if rot is not None:
            ds.unaligned_coords["rotation"] = rot
        if shp is not None:
            ds.unaligned_coords["shape"] = shp


def validate_dim_and_get_mantid_string(unit_dim):
    known_units = {
        'Delta-E': "DeltaE",
        'tof': "TOF",
        'wavelength': "Wavelength",
        'E': "Energy",
        'd-spacing': "dSpacing",
        'Q': "MomentumTransfer",
        'Q^2': "QSquared",
    }

    user_k = str(unit_dim).casefold()
    known_units = {k.casefold(): v for k, v in known_units.items()}

    if user_k not in known_units:
        raise RuntimeError("Axis unit not currently supported."
                           "Possible values are: {}, "
                           "got '{}'. ".format([k for k in known_units.keys()],
                                               unit_dim))
    else:
        return known_units[user_k]


def _table_to_data_array(table, key, value, stddev):
    stddevs = table[stddev].values
    dim = 'parameter'
    coord = table[key].data.copy()
    coord.rename_dims({'row': dim})
    return sc.DataArray(data=sc.Variable(dims=[dim],
                                         values=table[value].values,
                                         variances=stddevs * stddevs),
                        coords={dim: coord})
