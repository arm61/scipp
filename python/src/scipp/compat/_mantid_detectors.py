import numpy as np

from .._scipp import core as sc


def make_detector_info(ws):
    det_info = ws.detectorInfo()
    # det -> spec mapping
    n_det = det_info.size()
    spectrum = sc.Variable(['detector'], shape=(n_det, ), dtype=sc.dtype.int32)
    has_spectrum = sc.Variable(['detector'], values=np.full((n_det, ), False))
    spectrum_ = spectrum.values
    has_spectrum_ = has_spectrum.values
    spec_info = ws.spectrumInfo()
    for i, spec in enumerate(spec_info):
        spec_def = spec.spectrumDefinition
        for j in range(len(spec_def)):
            det, time = spec_def[j]
            if time != 0:
                raise RuntimeError(
                    "Conversion of Mantid Workspace with scanning instrument "
                    "not supported yet.")
            spectrum_[det] = i
            has_spectrum_[det] = True
    detector = sc.Variable(['detector'], values=det_info.detectorIDs())

    # Remove any information about detectors without data (a spectrum). This
    # mostly just gets in the way and including it in the default converter
    # is probably not required.
    spectrum = sc.filter(spectrum, has_spectrum)
    detector = sc.filter(detector, has_spectrum)

    # May want to include more information here, such as detector positions,
    # but for now this is not necessary.
    return sc.Variable(value=sc.Dataset(coords={
        'detector': detector,
        'spectrum': spectrum
    }))


def get_detector_properties(ws,
                            source_pos,
                            sample_pos,
                            advanced_geometry=False):
    if not advanced_geometry:
        return _get_detector_pos(ws), None, None
    spec_info = ws.spectrumInfo()
    det_info = ws.detectorInfo()
    comp_info = ws.componentInfo()
    nspec = len(spec_info)
    det_rot = np.zeros([nspec, 3, 3])
    det_bbox = np.zeros([nspec, 3])

    if sample_pos is not None and source_pos is not None:
        total_detectors = spec_info.detectorCount()
        act_beam = (sample_pos - source_pos).values
        rot = _rot_from_vectors(act_beam, [0, 0, 1])
        inv_rot = _rot_from_vectors([0, 0, 1], act_beam)

        pos_d = sc.Dataset()
        # Create empty to hold position info for all spectra detectors
        pos_d["x"] = sc.Variable(["detector"],
                                 shape=[total_detectors],
                                 unit=sc.units.m)
        pos_d["y"] = pos_d["x"]
        pos_d["z"] = pos_d["x"]
        pos_d.coords["spectrum"] = sc.Variable(
            ["detector"], values=np.empty(total_detectors))
        spectrum_values = pos_d.coords["spectrum"].values

        x_values = pos_d["x"].values
        y_values = pos_d["y"].values
        z_values = pos_d["z"].values

        idx = 0
        for i, spec in enumerate(spec_info):
            if spec.hasDetectors:
                definition = spec_info.getSpectrumDefinition(i)
                n_dets = len(definition)
                quats = []
                bboxes = []
                for j in range(n_dets):
                    det_idx = definition[j][0]
                    p = det_info.position(det_idx)
                    r = det_info.rotation(det_idx)
                    s = comp_info.shape(det_idx)
                    spectrum_values[idx] = i
                    x_values[idx] = p.X()
                    y_values[idx] = p.Y()
                    z_values[idx] = p.Z()
                    idx += 1
                    quats.append(
                        np.array([r.imagI(),
                                  r.imagJ(),
                                  r.imagK(),
                                  r.real()]))
                    bboxes.append(s.getBoundingBox().width())
                det_rot[i, :] = sc.rotation_matrix_from_quaternion_coeffs(
                    np.mean(quats, axis=0))
                det_bbox[i, :] = np.sum(bboxes, axis=0)

        rot_pos = rot * sc.geometry.position(pos_d["x"].data, pos_d["y"].data,
                                             pos_d["z"].data)

        _to_spherical(rot_pos, pos_d)

        averaged = sc.groupby(pos_d,
                              "spectrum",
                              bins=sc.Variable(["spectrum"],
                                               values=np.arange(
                                                   -0.5,
                                                   len(spec_info) + 0.5,
                                                   1.0))).mean("detector")

        averaged["x"] = averaged["r"].data * sc.sin(
            averaged["t"].data) * sc.cos(averaged["p"].data)
        averaged["y"] = averaged["r"].data * sc.sin(
            averaged["t"].data) * sc.sin(averaged["p"].data)
        averaged["z"] = averaged["r"].data * sc.cos(averaged["t"].data)

        pos = sc.geometry.position(averaged["x"].data, averaged["y"].data,
                                   averaged["z"].data)
        return (inv_rot * pos,
                sc.Variable(['spectrum'],
                            values=det_rot,
                            dtype=sc.dtype.matrix_3_float64),
                sc.Variable(['spectrum'],
                            values=det_bbox,
                            unit=sc.units.m,
                            dtype=sc.dtype.vector_3_float64))
    else:
        pos = np.zeros([nspec, 3])

        for i, spec in enumerate(spec_info):
            if spec.hasDetectors:
                definition = spec_info.getSpectrumDefinition(i)
                n_dets = len(definition)
                vec3s = []
                quats = []
                bboxes = []
                for j in range(n_dets):
                    det_idx = definition[j][0]
                    p = det_info.position(det_idx)
                    r = det_info.rotation(det_idx)
                    s = comp_info.shape(det_idx)
                    vec3s.append([p.X(), p.Y(), p.Z()])
                    quats.append(
                        np.array([r.imagI(),
                                  r.imagJ(),
                                  r.imagK(),
                                  r.real()]))
                    bboxes.append(s.getBoundingBox().width())
                pos[i, :] = np.mean(vec3s, axis=0)
                det_rot[i, :] = sc.rotation_matrix_from_quaterion_cooffs(
                    np.mean(quats, axis=0))
                det_bbox[i, :] = np.sum(bboxes, axis=0)
            else:
                pos[i, :] = [np.nan, np.nan, np.nan]
                det_rot[i, :] = [np.nan, np.nan, np.nan, np.nan]
                det_bbox[i, :] = [np.nan, np.nan, np.nan]
        return (sc.Variable(['spectrum'],
                            values=pos,
                            unit=sc.units.m,
                            dtype=sc.dtype.vector_3_float64),
                sc.Variable(['spectrum'],
                            values=det_rot,
                            dtype=sc.dtype.matrix_3_float64),
                sc.Variable(['spectrum'],
                            values=det_bbox,
                            unit=sc.units.m,
                            dtype=sc.dtype.vector_3_float64))


def _get_detector_pos(ws):
    n_hist = ws.getNumberHistograms()
    pos = np.zeros([n_hist, 3])

    spec_info = ws.spectrumInfo()
    for i in range(n_hist):
        if spec_info.hasDetectors(i):
            p = spec_info.position(i)
            pos[i, 0] = p.X()
            pos[i, 1] = p.Y()
            pos[i, 2] = p.Z()
        else:
            pos[i, :] = [np.nan, np.nan, np.nan]
    return sc.Variable(['spectrum'],
                       values=pos,
                       unit=sc.units.m,
                       dtype=sc.dtype.vector_3_float64)


def _rot_from_vectors(vec1, vec2):
    a = sc.Variable(value=vec1 / np.linalg.norm(vec1),
                    dtype=sc.dtype.vector_3_float64)
    b = sc.Variable(value=vec2 / np.linalg.norm(vec2),
                    dtype=sc.dtype.vector_3_float64)
    c = sc.Variable(value=np.cross(a.value, b.value),
                    dtype=sc.dtype.vector_3_float64)
    angle = sc.acos(sc.dot(a, b)).value
    q = sc.rotation_matrix_from_quaternion_coeffs(
        list(c.value * np.sin(angle / 2)) + [np.cos(angle / 2)])
    return sc.Variable(value=q)


def _to_spherical(pos, output):
    output["r"] = sc.sqrt(sc.dot(pos, pos))
    output["t"] = sc.acos(sc.geometry.z(pos) / output["r"].data)
    output["p"] = output["t"].data.copy()
    sc.atan2(sc.geometry.y(pos), sc.geometry.x(pos), output["p"].data)
