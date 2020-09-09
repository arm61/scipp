# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock, Neil Vaytet

import numpy as np

from .. import detail
from .._utils import is_data_array
from .._scipp import core as sc
from .._scipp.core import contains_events
from ._mantid_algorithm import run_mantid_alg
from ._mantid_workspace import (_table_to_data_array, validate_and_get_unit,
                                validate_dim_and_get_mantid_string,
                                init_spec_axis, _convert_MatrixWorkspace_info,
                                set_bin_masks, md_dimension, md_unit,
                                convert_monitors_ws)


def _fit_workspace(ws, mantid_args):
    """
    Performs a fit on the workspace.

    :param ws: The workspace on which the fit will be performed
    :returns: Dataset containing all of Fit's outputs
    """
    with run_mantid_alg('Fit',
                        InputWorkspace=ws,
                        **mantid_args,
                        CreateOutput=True) as fit_algorithm:
        # This is assuming that all parameters are dimensionless. If this is
        # not the case we should use a dataset with a scalar variable per
        # parameter instead. Or better, a dict of scalar variables?
        parameters = convert_TableWorkspace_to_dataset(
            fit_algorithm.OutputParameters)
        parameters = _table_to_data_array(parameters,
                                          key='Name',
                                          value='Value',
                                          stddev='Error')
        out = convert_Workspace2D_to_data_array(fit_algorithm.OutputWorkspace)
        data = sc.Dataset()
        data['data'] = out['empty', 0]
        data['calculated'] = out['empty', 1]
        data['diff'] = out['empty', 2]
        parameters.coords['status'] = sc.Variable(fit_algorithm.OutputStatus)
        parameters.coords['chi^2/d.o.f.'] = sc.Variable(
            fit_algorithm.OutputChi2overDoF)
        parameters.coords['function'] = sc.Variable(str(
            fit_algorithm.Function))
        parameters.coords['cost-function'] = sc.Variable(
            fit_algorithm.CostFunction)
        return parameters, data


def fit(data, mantid_args):
    if len(data.dims) != 1 or 'WorkspaceIndex' in mantid_args:
        raise RuntimeError(
            "Only 1D fitting is supported. Use scipp slicing and do not"
            "provide a WorkspaceIndex.")
    dim = data.dims[0]
    ws = to_mantid(data, dim)
    mantid_args['workspace_index'] = 0
    return _fit_workspace(ws, mantid_args)


def to_mantid(data, dim, instrument_file=None):
    """
    Convert data to a Mantid workspace.

    The Mantid layout expect the spectra to be the Outer-most dimension,
    i.e. y.shape[0]. If that is not the case you might have to transpose
    your data to fit that, otherwise it will not be aligned correctly in the
    Mantid workspace.

    :param data: Data to be converted.
    :param dim: Coord to use for Mantid's first axis (X).
    :param instrument_file: Instrument file that will be
                            loaded into the workspace
    :returns: Workspace containing converted data. The concrete workspace type
              may differ depending on the content of `data`.
    """
    if not is_data_array(data):
        raise RuntimeError(
            "Currently only data arrays can be converted to a Mantid workspace"
        )
    if data.data is None or contains_events(data):
        raise RuntimeError(
            "Currently only histogrammed data can be converted.")
    try:
        import mantid.simpleapi as mantid
    except ImportError:
        raise ImportError(
            "Mantid Python API was not found, please install Mantid framework "
            "as detailed in the installation instructions (https://scipp."
            "github.io/getting-started/installation.html)")
    x = data.coords[dim].values
    y = data.values
    e = data.variances

    assert (len(y.shape) == 2 or len(y.shape) == 1), \
        "Currently can only handle 2D data."

    e = np.sqrt(e) if e is not None else np.sqrt(y)

    # Convert a single array (e.g. single spectra) into 2d format
    if len(y.shape) == 1:
        y = np.array([y])

    if len(e.shape) == 1:
        e = np.array([e])

    unit_x = validate_dim_and_get_mantid_string(dim)

    nspec = y.shape[0]
    if len(x.shape) == 1:
        # SCIPP is using a  1:n spectra coord mapping, Mantid needs
        # a 1:1 mapping so expand this out
        x = np.broadcast_to(x, shape=(nspec, len(x)))

    nbins = x.shape[1]
    nitems = y.shape[1]

    ws = mantid.WorkspaceFactory.create("Workspace2D",
                                        NVectors=nspec,
                                        XLength=nbins,
                                        YLength=nitems)
    if data.unit != sc.units.counts:
        ws.setDistribution(True)

    for i in range(nspec):
        ws.setX(i, x[i])
        ws.setY(i, y[i])
        ws.setE(i, e[i])

    # Set X-Axis unit
    ws.getAxis(0).setUnit(unit_x)

    if instrument_file is not None:
        mantid.LoadInstrument(ws,
                              FileName=instrument_file,
                              RewriteSpectraMap=True)

    return ws


def convert_Workspace2D_to_data_array(ws,
                                      load_run_logs=True,
                                      advanced_geometry=False,
                                      **ignored):

    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit().unitID())
    spec_dim, spec_coord = init_spec_axis(ws)

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs)
    _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)
    if ws.id() == 'MaskWorkspace':
        coords_labs_data["data"] = sc.Variable([spec_dim],
                                               unit=data_unit,
                                               values=ws.extractY().flatten(),
                                               dtype=sc.dtype.bool)
    else:
        stddev2 = ws.extractE()
        np.multiply(stddev2, stddev2, out=stddev2)  # much faster than np.power
        coords_labs_data["data"] = sc.Variable([spec_dim, dim],
                                               unit=data_unit,
                                               values=ws.extractY(),
                                               variances=stddev2)
    array = detail.move_to_data_array(**coords_labs_data)

    if ws.hasAnyMaskedBins():
        bin_mask = sc.Variable(dims=array.dims,
                               shape=array.shape,
                               dtype=sc.dtype.bool)
        for i in range(ws.getNumberHistograms()):
            # maskedBinsIndices throws instead of returning empty list
            if ws.hasMaskedBins(i):
                set_bin_masks(bin_mask, dim, i, ws.maskedBinsIndices(i))
        common_mask = sc.all(bin_mask, 'spectrum')
        if sc.is_equal(common_mask, sc.any(bin_mask, 'spectrum')):
            array.masks["bin"] = detail.move(common_mask)
        else:
            array.masks["bin"] = detail.move(bin_mask)

    # Avoid creating dimensions that are not required since this mostly an
    # artifact of inflexible data structures and gets in the way when working
    # with scipp.
    if len(spec_coord.values) == 1:
        if 'position' in array.coords:
            array.coords['position'] = array.coords['position'][spec_dim, 0]
        array = array[spec_dim, 0].copy()
    return array


def convert_EventWorkspace_to_data_array(ws,
                                         load_pulse_times=True,
                                         realign_events=False,
                                         advanced_geometry=False,
                                         load_run_logs=True,
                                         **ignored):
    from mantid.api import EventType

    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit().unitID())
    spec_dim, spec_coord = init_spec_axis(ws)
    n_hist = ws.getNumberHistograms()

    coord = sc.Variable([spec_dim],
                        shape=[n_hist],
                        unit=unit,
                        dtype=sc.dtype.event_list_float64)
    if load_pulse_times:
        labs = sc.Variable([spec_dim],
                           shape=[n_hist],
                           dtype=sc.dtype.event_list_int64)

    # Check for weighted events
    evtp = ws.getSpectrum(0).getEventType()
    contains_weighted_events = ((evtp == EventType.WEIGHTED)
                                or (evtp == EventType.WEIGHTED_NOTIME))
    if contains_weighted_events:
        weights = sc.Variable([spec_dim],
                              shape=[n_hist],
                              dtype=sc.dtype.event_list_float32,
                              variances=True)

    for i in range(n_hist):
        sp = ws.getSpectrum(i)
        coord[spec_dim, i].values = sp.getTofs()
        if load_pulse_times:
            labs[spec_dim, i].values = sp.getPulseTimesAsNumpy()
        if contains_weighted_events:
            weights[spec_dim, i].values = sp.getWeights()
            weights[spec_dim, i].variances = sp.getWeightErrors()

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs)
    bin_edges = coords_labs_data["coords"][dim]
    coords_labs_data["coords"][dim] = coord

    if load_pulse_times:
        coords_labs_data["coords"]["pulse-time"] = labs
    if contains_weighted_events:
        coords_labs_data["data"] = weights
    else:
        _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)
        coords_labs_data["data"] = sc.Variable(dims=[spec_dim],
                                               values=np.ones(n_hist),
                                               variances=np.ones(n_hist),
                                               unit=data_unit,
                                               dtype=sc.dtype.float32)
    array = detail.move_to_data_array(**coords_labs_data)
    if realign_events:
        # Event data is stored as unaligned content, with realigned wrapper
        # based on Mantid's bin edges.
        array.realign({dim: bin_edges})
    return array


def convert_MDHistoWorkspace_to_data_array(md_histo, **ignored):
    ndims = md_histo.getNumDims()
    coords = dict()
    dims_used = []
    for i in range(ndims):
        dim = md_histo.getDimension(i)
        frame = dim.getMDFrame()
        sc_dim = md_dimension(dim)
        coords[sc_dim] = sc.Variable(dims=[sc_dim],
                                     values=np.linspace(
                                         dim.getMinimum(), dim.getMaximum(),
                                         dim.getNBins()),
                                     unit=md_unit(frame))
        dims_used.append(sc_dim)
    data = sc.Variable(dims=dims_used,
                       values=md_histo.getSignalArray(),
                       variances=md_histo.getErrorSquaredArray(),
                       unit=sc.units.counts)
    nevents = sc.Variable(dims=dims_used, values=md_histo.getNumEventsArray())
    return detail.move_to_data_array(coords=coords,
                                     data=data,
                                     unaligned_coords={'nevents': nevents})


def convert_TableWorkspace_to_dataset(ws, error_connection=None, **ignored):
    """
    Converts from a Mantid TableWorkspace to a scipp dataset. It is possible
    to assign a column as the error for another column, in which case a
    the data from the two columns will be represented by a single scipp
    variable with variance. This is done using the error_connection Keyword
    argument. The error is transformed to variance in this converter.

    Parameters
    ----------
        :param ws: Mantid TableWorkspace
            Mantid TableWorkspace to be converted into scipp dataset

    Keyword arguments
    -----------------
        :param error_connection: Dict with data column names as keys
                                 to names of their error column
    """

    # Extract information from workspace
    n_columns = ws.columnCount()
    column_names = ws.getColumnNames()  # list of names matching each column
    column_types = ws.columnTypes()  # list of types matching each column

    # Types available in TableWorkspace that can not be loaded into scipp
    blacklist_types = []
    # Types for which the transformation from error to variance will fail
    blacklist_variance_types = ["str"]

    dataset = sc.Dataset()
    for i in range(n_columns):
        if column_types[i] in blacklist_types:
            continue  # skips loading data of this type

        data_name = column_names[i]
        if error_connection is None:
            dataset[data_name] = detail.move(
                sc.Variable(['row'], values=ws.column(i)))
        elif data_name in error_connection:
            # This data has error availble
            error_name = error_connection[data_name]
            error_index = column_names.index(error_name)

            if column_types[error_index] in blacklist_variance_types:
                # Raise error to avoid numpy square error for strings
                raise RuntimeError("Variance can not have type string. \n" +
                                   "Data:     " + str(data_name) + "\n" +
                                   "Variance: " + str(error_name) + "\n")

            variance = np.array(ws.column(error_name))**2
            dataset[data_name] = detail.move(
                sc.Variable(['row'],
                            values=np.array(ws.column(i)),
                            variances=variance))
        elif data_name not in error_connection.values():
            # This data is not an error for another dataset, and has no error
            dataset[data_name] = detail.move(
                sc.Variable(['row'], values=ws.column(i)))

    return dataset


def from_mantid(workspace, **kwargs):
    """Convert Mantid workspace to a scipp data array or dataset
    :param workspace: Mantid workspace to convert.
    """
    scipp_obj = None  # This is either a Dataset or DataArray
    monitor_ws = None
    workspaces_to_delete = []
    w_id = workspace.id()
    if (w_id == 'Workspace2D' or w_id == 'RebinnedOutput'
            or w_id == 'MaskWorkspace'):
        n_monitor = 0
        spec_info = workspace.spectrumInfo()
        for i in range(len(spec_info)):
            if spec_info.hasDetectors(i) and spec_info.isMonitor(i):
                n_monitor += 1
        # If there are *only* monitors we do not move them to an attribute
        if 0 < n_monitor < len(spec_info):
            import mantid.simpleapi as mantid
            workspace, monitor_ws = mantid.ExtractMonitors(workspace)
            workspaces_to_delete.append(workspace)
            workspaces_to_delete.append(monitor_ws)
        scipp_obj = convert_Workspace2D_to_data_array(workspace, **kwargs)
    elif w_id == 'EventWorkspace':
        scipp_obj = convert_EventWorkspace_to_data_array(workspace, **kwargs)
    elif w_id == 'TableWorkspace':
        scipp_obj = convert_TableWorkspace_to_dataset(workspace, **kwargs)
    elif w_id == 'MDHistoWorkspace':
        scipp_obj = convert_MDHistoWorkspace_to_data_array(workspace, **kwargs)

    if scipp_obj is None:
        raise RuntimeError('Unsupported workspace type {}'.format(w_id))

    # TODO Is there ever a case where a Workspace2D has a separate monitor
    # workspace? This is not handled by ExtractMonitors above, I think.
    if monitor_ws is None:
        if hasattr(workspace, 'getMonitorWorkspace'):
            try:
                monitor_ws = workspace.getMonitorWorkspace()
            except RuntimeError:
                # Have to try/fail here. No inspect method on Mantid for this.
                pass

    if monitor_ws is not None:
        if monitor_ws.id() == 'MaskWorkspace' or monitor_ws.id(
        ) == 'Workspace2D':
            converter = convert_Workspace2D_to_data_array
        elif monitor_ws.id() == 'EventWorkspace':
            converter = convert_EventWorkspace_to_data_array
        else:
            raise RuntimeError(
                'Monitor workspace not of known type in from_mantid()')

        monitors = convert_monitors_ws(monitor_ws, converter, **kwargs)
        for name, monitor in monitors:
            scipp_obj.unaligned_coords[name] = detail.move(
                sc.Variable(value=monitor))
    for ws in workspaces_to_delete:
        mantid.DeleteWorkspace(ws)

    return scipp_obj


def load(filename="",
         load_pulse_times=True,
         realign_events=False,
         instrument_filename=None,
         error_connection=None,
         mantid_alg='Load',
         mantid_args=None,
         advanced_geometry=False):
    """
    Wrapper function to provide a load method for a Nexus file, hiding mantid
    specific code from the scipp interface. All other keyword arguments not
    specified in the parameters below are passed on to the mantid.Load
    function.

    Example of use:

    .. highlight:: python
    .. code-block:: python

        from scipp.neutron import load
        d = sc.Dataset()
        d["sample"] = load(filename='PG3_4844_event.nxs',
                           load_pulse_times=False,
                           mantid_args={'BankName': 'bank184',
                                        'LoadMonitors': True})

    See also the neutron-data tutorial.

    Note that this function requires mantid to be installed and available in
    the same Python environment as scipp.

    :param str filename: The name of the Nexus/HDF file to be loaded.
    :param bool load_pulse_times: Read the pulse times if True.
    :param bool realign_events: Realign event data according to "X" axis given
                                by file.
    :param str instrument_filename: If specified, over-write the instrument
                                    definition in the final Dataset with the
                                    geometry contained in the file.
    :param error_connection: Dict with data column names as keys to names of
                             their error column
    :param str mantid_alg: Mantid algorithm to use for loading. Default is
                           `Load`.
    :param dict mantid_args: Dict of keyword arguments to forward to Mantid.
    :param bool advanced_geometry: If True, load the full detector geometry
                                   including shapes and rotations. The
                                   positions of grouped detectors are
                                   spherically averaged. If False,
                                   load only the detector position, and return
                                   the cartesian average of the grouped
                                   detector positions.
    :raises: If the Mantid workspace type returned by the Mantid loader is not
             either EventWorkspace or Workspace2D.
    :return: A Dataset containing the neutron event/histogram data and the
             instrument geometry.
    :rtype: Dataset
    """

    if mantid_args is None:
        mantid_args = {}

    with run_mantid_alg(mantid_alg, filename, **mantid_args) as loaded:
        # Determine what Load has provided us
        from mantid.api import Workspace
        if isinstance(loaded, Workspace):
            # A single workspace
            data_ws = loaded
        else:
            # Seperate data and monitor workspaces
            data_ws = loaded.OutputWorkspace

        if instrument_filename is not None:
            import mantid.simpleapi as mantid
            mantid.LoadInstrument(data_ws,
                                  FileName=instrument_filename,
                                  RewriteSpectraMap=True)

        return from_mantid(data_ws,
                           load_pulse_times=load_pulse_times,
                           realign_events=realign_events,
                           error_connection=error_connection,
                           advanced_geometry=advanced_geometry)
