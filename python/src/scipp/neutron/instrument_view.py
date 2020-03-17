# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from .. import config
from ..plot.render import render_plot
from ..plot.sciplot import SciPlot
from ..plot.sparse import histogram_sparse_data, make_bins
from ..plot.tools import parse_params
from ..utils import name_with_unit, value_to_string
from .._scipp import core as sc, neutron as sn

# Other imports
import numpy as np
import importlib


def instrument_view(scipp_obj=None,
                    bins=None,
                    masks=None,
                    filename=None,
                    figsize=None,
                    aspect="equal",
                    cmap=None,
                    log=False,
                    vmin=None,
                    vmax=None,
                    size=0.12,
                    projection="3D Z",
                    nan_color="#d3d3d3",
                    continuous_update=True,
                    dim="tof"):
    """
    Plot a 2D or 3D view of the instrument.
    A slider is also generated to navigate the dimension (dim) given as an
    input. Since this is designed for neutron science, the default dimension
    is neutron time-of-flight (sc.Dim.Tof), but this could be anything
    (wavelength, temperature...)

    Example:

    import scipp.neutron as sn
    sample = sn.load(filename="PG3_4844_event.nxs")
    sn.instrument_view(sample)
    """

    iv = InstrumentView(scipp_obj=scipp_obj,
                        bins=bins,
                        masks=masks,
                        cmap=cmap,
                        log=log,
                        vmin=vmin,
                        vmax=vmax,
                        aspect=aspect,
                        size=size,
                        projection=projection,
                        nan_color=nan_color,
                        continuous_update=continuous_update,
                        dim=dim)

    render_plot(widgets=iv.box, filename=filename)

    return SciPlot(iv.members)


class InstrumentView:
    def __init__(self,
                 scipp_obj=None,
                 bins=None,
                 masks=None,
                 cmap=None,
                 log=None,
                 vmin=None,
                 vmax=None,
                 aspect=None,
                 size=None,
                 projection=None,
                 nan_color=None,
                 continuous_update=None,
                 dim=None):

        # Delayed imports to avoid hard dependencies
        self.widgets = importlib.import_module("ipywidgets")
        self.mpl = importlib.import_module("matplotlib")
        self.mpl_cm = importlib.import_module("matplotlib.cm")
        self.mpl_plt = importlib.import_module("matplotlib.pyplot")
        self.mpl_figure = importlib.import_module("matplotlib.figure")
        self.mpl_colors = importlib.import_module("matplotlib.colors")
        self.mpl_backend_agg = importlib.import_module(
            "matplotlib.backends.backend_agg")
        self.pil_image = importlib.import_module("PIL.Image")
        self.p3 = importlib.import_module("pythreejs")

        self.fig = None
        self.aspect = aspect
        self.nan_color = nan_color
        self.log = log
        self.current_projection = None
        self.dim = dim
        self.cbar_image = self.widgets.Image()

        if len(np.shape(size)) == 0:
            self.size = [size, size]
        else:
            self.size = size

        self.data_arrays = {}
        masks_present = False
        tp = type(scipp_obj)
        if tp is sc.Dataset or tp is sc.DatasetView:
            for key in sorted(scipp_obj.keys()):
                var = scipp_obj[key]
                if self.dim in var.dims:
                    self.data_arrays[key] = var
                    if len(var.masks) > 0:
                        masks_present = True
        elif tp is sc.DataArray or tp is sc.DataArrayView:
            self.data_arrays[scipp_obj.name] = scipp_obj
            if len(scipp_obj.masks) > 0:
                masks_present = True
        else:
            raise RuntimeError("Unknown input type: {}. Allowed inputs "
                               "are a Dataset or a DataArray (and their "
                               "respective proxies).".format(tp))

        self.globs = {"cmap": cmap, "log": log, "vmin": vmin, "vmax": vmax}
        self.params = {}
        self.cmap = {}
        self.hist_data_array = {}
        self.scalar_map = {}
        self.minmax = {}
        self.masks = masks
        self.masks_variables = {}
        self.masks_params = {}
        self.masks_cmap = {}
        self.masks_scalar_map = {}

        # Find the min/max time-of-flight limits and store them
        self.minmax["tof"] = [np.Inf, np.NINF, 1]
        for key, data_array in self.data_arrays.items():
            bins_here = bins
            is_events = sc.is_events(data_array)
            if is_events and bins_here is None:
                bins_here = True
            if bins_here is not None:
                dim = None if is_events else self.dim
                spdim = None if not is_events else self.dim
                var = make_bins(data_array=data_array,
                                sparse_dim=spdim,
                                dim=dim,
                                bins=bins_here,
                                padding=is_events)
            else:
                var = data_array.coords[self.dim]
            self.minmax["tof"][0] = min(self.minmax["tof"][0], var.values[0])
            self.minmax["tof"][1] = max(self.minmax["tof"][1], var.values[-1])
            self.minmax["tof"][2] = var.shape[0]

        available_cmaps = sorted(m for m in self.mpl_plt.colormaps()
                                 if not m.endswith("_r"))

        # Store current active data entry (DataArray)
        keys = list(self.data_arrays.keys())
        self.key = keys[0]

        # This widget needs to be before the call to rebin_data
        self.masks_cmap_or_color = self.widgets.RadioButtons(
            options=['colormap', 'solid color'],
            description='Mask color:',
            layout={'width': "250px"})
        self.masks_cmap_or_color.observe(self.toggle_color_selection,
                                         names="value")

        # Rebin all DataArrays to common Tof axis
        self.rebin_data(np.linspace(*self.minmax["tof"]))

        # Create dropdown menu to select the DataArray
        self.dropdown = self.widgets.Dropdown(options=keys,
                                              description="Select entry:",
                                              layout={"width": "initial"})
        self.dropdown.observe(self.change_data_array, names="value")

        # Create a Tof slider and its label
        self.tof_dim_indx = self.hist_data_array[self.key].dims.index(self.dim)
        self.slider = self.widgets.IntSlider(
            value=0,
            min=0,
            step=1,
            description=str(self.dim).replace("Dim.", ""),
            max=self.hist_data_array[self.key].shape[self.tof_dim_indx] - 1,
            continuous_update=continuous_update,
            readout=False)
        self.slider.observe(self.update_colors, names="value")
        self.label = self.widgets.Label()

        # Add dropdown to change cmap
        self.select_colormap = self.widgets.Combobox(
            options=available_cmaps,
            value=self.params[self.key]["cmap"],
            description="Select colormap",
            ensure_option=True,
            layout={'width': "200px"},
            style={"description_width": "initial"})
        self.select_colormap.observe(self.update_colormap, names="value")

        # Add text boxes to change number of bins/bin size
        self.nbins = self.widgets.Text(value=str(
            self.hist_data_array[self.key].shape[self.tof_dim_indx]),
                                       description="Number of bins:",
                                       style={"description_width": "initial"})
        self.nbins.on_submit(self.update_nbins)

        tof_values = self.hist_data_array[self.key].coords[self.dim].values
        self.bin_size = self.widgets.Text(value=str(tof_values[1] -
                                                    tof_values[0]),
                                          description="Bin size:")
        self.bin_size.on_submit(self.update_bin_size)


        self.opacity_slider = self.widgets.FloatSlider(
            min=0, max=1.0, value=1.0, step=0.01, description="Opacity")
        self.opacity_slider.observe(self.update_opacity, names="value")

        projections = [
            "3D X", "3D Y", "3D Z",
            "Cylindrical X", "Cylindrical Y", "Cylindrical Z",
            "Spherical X", "Spherical Y", "Spherical Z"
        ]

        # Create toggle buttons to change projection
        self.buttons = {}
        for p in projections:
            self.buttons[p] = self.widgets.Button(
                description=p,
                disabled=False,
                button_style=("info" if (p == projection) else ""))
            self.buttons[p].on_click(self.change_projection)
        items = []
        for x in "XYZ":
            items.append(self.buttons["3D {}".format(x)])
            items.append(self.buttons["Cylindrical {}".format(x)])
            items.append(self.buttons["Spherical {}".format(x)])
            # if x != "Z":
            #     items.append(self.widgets.Label())

        self.togglebuttons = self.widgets.GridBox(
            items,
            layout=self.widgets.Layout(
                grid_template_columns="repeat(3, 150px)"))

        # Create control section for masks
        self.masks_showhide = self.widgets.ToggleButton(
            value=True, description="Hide masks", button_style="")
        self.masks_showhide.observe(self.toggle_masks, names="value")

        self.masks_colormap = self.widgets.Combobox(
            options=available_cmaps,
            value=self.masks_params[self.key]["cmap"],
            description="",
            ensure_option=True,
            disabled=not self.masks_cmap_or_color.value == "colormap",
            layout={'width': "100px"})
        self.masks_colormap.observe(self.update_masks_colormap, names="value")

        self.masks_solid_color = self.widgets.ColorPicker(
            concise=False,
            description='',
            value='#%02X%02X%02X' % (tuple(np.random.randint(0, 255, 3))),
            disabled=not self.masks_cmap_or_color.value == "solid color",
            layout={'width': "100px"})
        self.masks_solid_color.observe(self.update_masks_solid_color,
                                       names="value")

        # Place widgets in boxes
        box_list = [
            self.widgets.HBox(
                [self.dropdown, self.slider, self.label,
                 self.select_colormap]),
            self.widgets.HBox([self.nbins, self.bin_size]),
            self.opacity_slider, self.togglebuttons
        ]
        # Only show mask controls if masks are present
        if masks_present:
            box_list.append(
                self.widgets.HBox([
                    self.masks_showhide, self.masks_cmap_or_color,
                    self.masks_colormap, self.masks_solid_color
                ]))

        self.vbox = self.widgets.VBox(box_list)

        # Get detector positions
        self.det_pos = np.array(sn.position(
            self.hist_data_array[self.key]).values,
                                dtype=np.float32)
        # Find extents of the detectors
        self.camera_pos = np.NINF
        for i, x in enumerate("xyz"):
            self.minmax[x] = [
                np.amin(self.det_pos[:, i]),
                np.amax(self.det_pos[:, i])
            ]
            self.camera_pos = max(self.camera_pos,
                                  np.amax(np.abs(self.minmax[x])))

        # # Create texture for scatter points to represent detector shapes
        # nx = 32
        # det_aspect_ratio = int(round(nx * min(self.size) / max(self.size)))
        # if det_aspect_ratio % 2 == 0:
        #     half_width = det_aspect_ratio // 2
        #     istart = nx // 2 - half_width
        #     iend = nx // 2 + half_width
        # else:
        #     half_width = (det_aspect_ratio - 1) // 2
        #     istart = nx // 2 - 1 - half_width
        #     iend = nx // 2 + half_width
        #     nx -= 1
        # texture_array = np.zeros([nx, nx, 4], dtype=np.float32)
        # if np.argmin(self.size) == 0:
        #     texture_array[:, istart:iend, :] = 1.0
        # else:
        #     texture_array[istart:iend, :, :] = 1.0
        # texture = self.p3.DataTexture(data=texture_array,
        #                               format="RGBAFormat",
        #                               type="FloatType")


        size_x = 0.007
        size_y = 0.10
        size_z = 0.007
        # # size_x = 1.0
        # # size_y = 1.0
        # # size_z = 1.0
        # detector_shape = np.array([[ 0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y,  0.5*size_z],
        #                            [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y, -0.5*size_z],
        #                            [ 0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x, -0.5*size_y, -0.5*size_z],
        #                            [-0.5*size_x,  0.5*size_y, -0.5*size_z]], dtype=np.float32)


        detector_shape = np.array([[-0.5*size_x, -0.5*size_y, -0.5*size_z],
                                   [ 0.5*size_x, -0.5*size_y, -0.5*size_z],
                                   [ 0.5*size_x,  0.5*size_y, -0.5*size_z],
                                   [-0.5*size_x,  0.5*size_y, -0.5*size_z],
                                   [-0.5*size_x, -0.5*size_y,  0.5*size_z],
                                   [ 0.5*size_x, -0.5*size_y,  0.5*size_z],
                                   [ 0.5*size_x,  0.5*size_y,  0.5*size_z],
                                   [-0.5*size_x,  0.5*size_y,  0.5*size_z]], dtype=np.float32)

        detector_faces = np.array([[0,4,3],
                                   [3,4,7],
                                   [1,2,6],
                                   [1,6,5],
                                   [0,1,5],
                                   [0,5,4],
                                   [2,3,7],
                                   [2,7,6],
                                   [0,2,1],
                                   [0,3,2],
                                   [4,5,7],
                                   [5,6,7]], dtype=np.uint32)





        # N = 100000
        # M = len(detector_shape)
        # L = len(detector_faces)

        # offsets = (np.random.random([N, 3]).astype(np.float32) - 0.5) * 100.0

        # # vertices = np.tile(morph.attributes["position"].array, [N, 1]) + np.repeat(offsets, M, axis=0)
        # vertices = np.tile(detector_shape, [N, 1]) + np.repeat(offsets, M, axis=0)

        # # faces = np.arange(M * N, dtype=np.uint32)

        # # faces = np.tile(detector_faces, [N, 1]) + np.repeat(np.arange(N, dtype=np.uint32), L, axis=0)
        # faces = np.tile(detector_faces, [N, 1]) + np.repeat(
        #     np.arange(0, N*M, M, dtype=np.uint32), L*3, axis=0).reshape(L*N, 3)


        # vertexcolors = np.repeat(np.random.random([N, 3]), M, axis=0).astype(np.float32)











        self.nverts = len(detector_shape)
        self.nfaces = len(detector_faces)
        self.ndets = len(self.det_pos)

        print(self.nverts, self.ndets, self.nfaces)

        # offsets = (np.random.random([N, 3]).astype(np.float32) - 0.5) * 50.0

        # print(np.repeat(self.det_pos, self.nverts, axis=0)[:100])

        self.vertices = np.tile(detector_shape, [self.ndets, 1]) + np.repeat(self.det_pos, self.nverts, axis=0)

        # faces = np.arange(self.nverts * self.ndets, dtype=np.uint)
        faces = np.tile(detector_faces, [self.ndets, 1]) + np.repeat(
            np.arange(0, self.ndets*self.nverts, self.nverts, dtype=np.uint32), self.nfaces*3, axis=0).reshape(self.nfaces*self.ndets, 3)

        # vertexcolors = np.repeat(np.random.random([self.ndets, 3]), self.nverts, axis=0).astype(np.float32)
        vertexcolors = np.zeros([self.ndets*self.nverts, 3], dtype=np.float32)

        # print(vertices.shape)
        # print(faces.shape)
        # print(vertexcolors.shape)


        self.geometry = self.p3.BufferGeometry(attributes=dict(
            position=self.p3.BufferAttribute(self.vertices, normalized=False),
            index=self.p3.BufferAttribute(faces.ravel(), normalized=False),
            color=self.p3.BufferAttribute(vertexcolors),
        ))

        self.material = self.p3.MeshBasicMaterial(vertexColors='VertexColors',
                                                  transparent=True)

        self.mesh = self.p3.Mesh(
            geometry=self.geometry,
            material=self.material
            # position=[-0.5, -0.5, -0.5]   # Center the cube
        )

        # self.wireframe = self.p3.LineSegments(
        #     geometry=self.p3.WireframeGeometry(self.geometry),
        #     material=self.p3.LineBasicMaterial())



        # self.nverts = 1000
        # self.ndets = 36

        # offsets = (np.random.random([self.nverts, 3]).astype(np.float32) - 0.5) * 50.0

        # # vertices = np.tile(morph.attributes["position"].array, [N, 1]) + np.repeat(offsets, M, axis=0)
        # vertices = np.tile(detector_shape, [self.nverts, 1]) + np.repeat(offsets, self.ndets, axis=0)

        # faces = np.arange(self.ndets * self.nverts, dtype=np.uint)

        # vertexcolors = np.repeat(np.random.random([self.nverts, 3]), self.ndets, axis=0).astype(np.float32)

        # print(vertices.shape)
        # print(faces.shape)
        # print(vertexcolors.shape)

        # self.geometry = self.p3.BufferGeometry(attributes=dict(
        #     position=self.p3.BufferAttribute(vertices, normalized=False),
        #     index=self.p3.BufferAttribute(faces, normalized=False),
        #     color=self.p3.BufferAttribute(vertexcolors),
        # ))

        # self.mesh = self.p3.Mesh(
        #     geometry=self.geometry,
        #     material=self.p3.MeshBasicMaterial(vertexColors='VertexColors'),
        # #     position=[-0.5, -0.5, -0.5]   # Center the cube
        # )








        # # The point cloud and its properties
        # self.pts = self.p3.BufferAttribute(array=self.det_pos)
        # self.colors = self.p3.BufferAttribute(
        #     array=np.zeros([np.shape(self.det_pos)[0], 4], dtype=np.float32))
        # self.geometry = self.p3.BufferGeometry(attributes={
        #     'position': self.pts,
        #     'color': self.colors
        # })
        # self.material = self.p3.PointsMaterial(vertexColors='VertexColors',
        #                                        size=max(self.size),
        #                                        map=texture,
        #                                        depthTest=False,
        #                                        transparent=True)
        # self.pcl = self.p3.Points(geometry=self.geometry,
        #                           material=self.material)
        # Add the red green blue axes helper
        self.axes_helper = self.p3.AxesHelper(100)

        # Create the threejs scene with ambient light and camera
        self.camera = self.p3.PerspectiveCamera(position=[self.camera_pos] * 3,
                                                aspect=config.plot.width /
                                                config.plot.height)
        # self.key_light = self.p3.DirectionalLight(position=[0, 10, 10])
        # self.ambient_light = self.p3.AmbientLight()

        self.scene = self.p3.Scene(children=[
            self.mesh, #self.wireframe,
            self.camera, #self.key_light, self.ambient_light,
            self.axes_helper
        ], background="#DDDDDD")
        self.controller = self.p3.OrbitControls(controlling=self.camera)

        # Render the scene into a widget
        self.renderer = self.p3.Renderer(camera=self.camera,
                                         scene=self.scene,
                                         controls=[self.controller],
                                         width=config.plot.width,
                                         height=config.plot.height)
        self.box = self.widgets.VBox(
            [self.widgets.HBox([self.renderer, self.cbar_image]), self.vbox])
        self.box.layout.align_items = "center"

        # Update the plot elements
        self.update_colorbar()
        self.change_projection(self.buttons[projection])

        # Create members object
        self.members = {
            "widgets": {
                "sliders": self.slider,
                "buttons": self.buttons,
                "text": {
                    "nbins": self.nbins,
                    "bin_size": self.bin_size
                },
                "dropdown": self.dropdown
            },
            "points": self.mesh,
            "camera": self.camera,
            "scene": self.scene,
            "renderer": self.renderer
        }

        return

    def rebin_data(self, bins):
        """
        Rebin the original data to Tof given some bins. This is executed both
        on first instrument display and when either the number of bins or the
        bin width is changed.
        """
        for key, data_array in self.data_arrays.items():

            # Histogram the data in the Tof dimension
            if sc.is_events(data_array):
                self.hist_data_array[key] = histogram_sparse_data(
                    data_array, self.dim, bins)
            else:
                self.hist_data_array[key] = sc.rebin(
                    data_array, self.dim,
                    make_bins(data_array=data_array,
                              dim=self.dim,
                              bins=bins,
                              padding=False))

            # Parse input parameters for colorbar
            self.params[key] = parse_params(
                globs=self.globs, array=self.hist_data_array[key].values)
            if key not in self.cmap:
                self.cmap[key] = self.mpl_cm.get_cmap(self.params[key]["cmap"])
                self.cmap[key].set_bad(color=self.nan_color)
            self.scalar_map[key] = self.mpl_cm.ScalarMappable(
                cmap=self.cmap[key], norm=self.params[key]["norm"])

            # Parse mask parameters and combine masks into one
            self.masks_params[key] = parse_params(params=self.masks,
                                                  defaults={
                                                      "cmap": "gray",
                                                      "cbar": False
                                                  })
            self.masks_params[key]["show"] = (
                self.masks_params[key]["show"]
                and len(self.hist_data_array[key].masks) > 0)
            if self.masks_params[key]["show"] and (
                    key not in self.masks_variables):
                self.masks_variables[key] = sc.combine_masks(
                    self.hist_data_array[key].masks,
                    self.hist_data_array[key].dims,
                    self.hist_data_array[key].shape)

        # Update the colormap with the new normalization since binning has
        # changed
        if self.masks_cmap_or_color.value == "colormap":
            self.masks_cmap = self.mpl_cm.get_cmap(
                self.masks_params[self.key]["cmap"])
        else:
            self.mpl_colors.LinearSegmentedColormap.from_list(
                "tmp",
                [self.masks_solid_color.value, self.masks_solid_color.value])
        self.masks_scalar_map = self.mpl_cm.ScalarMappable(
            cmap=self.masks_cmap, norm=self.params[self.key]["norm"])

        return

    def update_colorbar(self):
        height_inches = config.plot.height / config.plot.dpi
        fig = self.mpl_figure.Figure(figsize=(height_inches * 0.2,
                                              height_inches),
                                     dpi=config.plot.dpi)
        canvas = self.mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes([0.05, 0.02, 0.25, 0.96])
        cb1 = self.mpl.colorbar.ColorbarBase(
            ax, cmap=self.cmap[self.key], norm=self.params[self.key]["norm"])
        cb1.set_label(
            name_with_unit(var=self.hist_data_array[self.key], name=""))
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        shp = list(fig.canvas.get_width_height())[::-1] + [3]
        self.cbar_image.value = self.pil_image.fromarray(
            image.reshape(shp))._repr_png_()

    def update_colors(self, change):
        # return
        # arr = self.hist_data_array[self.key][self.dim, change["new"]].values
        arr = np.repeat(self.hist_data_array[self.key][self.dim, change["new"]].values,
                        self.nverts, axis=0)#.astype(np.float32)
        colors = self.scalar_map[self.key].to_rgba(arr).astype(np.float32)
        if self.key in self.masks_variables and self.masks_params[
                self.key]["show"]:
            # masks_colors = self.masks_scalar_map.to_rgba(arr).astype(
            #     np.float32)
            # masks_inds = np.where(np.repeat(self.masks_variables[self.key].values, self.nverts, axis=0))
            # colors[masks_inds] = masks_colors[masks_inds]
            masks_inds = np.where(np.repeat(self.masks_variables[self.key].values, self.nverts, axis=0))

            masks_colors = self.masks_scalar_map.to_rgba(arr[masks_inds]).astype(
                np.float32)
            # masks_inds = np.where(np.repeat(self.masks_variables[self.key].values, self.nverts, axis=0))
            colors[masks_inds] = masks_colors


        self.geometry.attributes["color"].array = colors[:, :3]

        self.label.value = name_with_unit(
            var=self.hist_data_array[self.key].coords[self.dim],
            name=value_to_string(self.hist_data_array[self.key].coords[
                self.dim].values[change["new"]]))
        return

    def change_projection(self, owner):

        if owner.description == self.current_projection:
            owner.button_style = "info"
            return
        if self.current_projection is not None:
            self.buttons[self.current_projection].button_style = ""

        projection = owner.description

        # Compute cylindrical or spherical projections
        permutations = {"X": [0, 2, 1], "Y": [1, 0, 2], "Z": [2, 1, 0]}
        axis = projection[-1]

        if projection.startswith("3D"):
            xyz = self.vertices
        else:
            xyz = np.zeros_like(self.vertices)
            xyz[:, 0] = np.arctan2(self.vertices[:, permutations[axis][2]],
                                   self.vertices[:, permutations[axis][1]])
            if projection.startswith("Cylindrical"):
                xyz[:, 1] = self.vertices[:, permutations[axis][0]]
            elif projection.startswith("Spherical"):
                xyz[:, 1] = np.arcsin(
                    self.vertices[:, permutations[axis][0]] /
                    np.sqrt(self.vertices[:, 0]**2 + self.vertices[:, 1]**2 +
                            self.vertices[:, 2]**2))

        if projection.startswith("3D"):
            self.axes_helper.visible = True
            self.camera.position = [self.camera_pos * (axis=="X"),
                                    self.camera_pos * (axis=="Y"),
                                    self.camera_pos * (axis=="Z")]
        else:
            self.axes_helper.visible = False
            self.camera.position = [0, 0, self.camera_pos]
        self.renderer.controls = [
            self.p3.OrbitControls(controlling=self.camera,
                                  enableRotate=projection.startswith("3D"))
        ]
        self.geometry.attributes["position"].array = xyz

        self.update_colors({"new": self.slider.value})

        self.current_projection = owner.description
        self.buttons[owner.description].button_style = "info"

        return

    def update_nbins(self, owner):
        try:
            nbins = int(owner.value)
        except ValueError:
            print("Warning: could not convert value: {} to an "
                  "integer.".format(owner.value))
            return
        self.rebin_data(
            np.linspace(self.minmax["tof"][0], self.minmax["tof"][1],
                        nbins + 1))
        x = self.hist_data_array[self.key].coords[self.dim].values
        self.bin_size.value = str(x[1] - x[0])
        self.update_slider()

    def update_bin_size(self, owner):
        try:
            binw = float(owner.value)
        except ValueError:
            print("Warning: could not convert value: {} to a "
                  "float.".format(owner.value))
            return
        self.rebin_data(
            np.arange(self.minmax["tof"][0], self.minmax["tof"][1], binw))
        self.nbins.value = str(
            self.hist_data_array[self.key].shape[self.tof_dim_indx])
        self.update_slider()

    def update_slider(self):
        """
        Try to replace the slider around the same position
        """

        # Compute percentage position
        perc_pos = self.slider.value / self.slider.max
        # Compute new position
        nbins = int(self.nbins.value) - 1
        new_pos = int(perc_pos * nbins)
        # Either place new upper boundary first, or change slider value first
        if new_pos > self.slider.max:
            self.slider.max = nbins
            self.slider.value = new_pos
        else:
            self.slider.value = new_pos
            self.slider.max = nbins
        self.update_colorbar()

    def change_data_array(self, change):
        self.key = change["new"]
        if self.select_colormap.value == self.params[self.key]["cmap"]:
            self.update_colorbar()
            self.update_colors({"new": self.slider.value})
        else:
            # Changing the colormap triggers the pixel color update
            self.select_colormap.value = self.params[self.key]["cmap"]

    def update_colormap(self, change):
        self.params[self.key]["cmap"] = change["new"]
        self.cmap[self.key] = self.mpl_cm.get_cmap(
            self.params[self.key]["cmap"])
        self.scalar_map[self.key] = self.mpl_cm.ScalarMappable(
            cmap=self.cmap[self.key], norm=self.params[self.key]["norm"])
        self.update_colorbar()
        self.update_colors({"new": self.slider.value})

    def toggle_masks(self, change):
        self.masks_params[self.key]["show"] = change["new"]
        change["owner"].description = "Hide masks" if change["new"] else \
            "Show masks"
        self.update_colors({"new": self.slider.value})

    def update_masks_colormap(self, change):
        self.masks_params[self.key]["cmap"] = change["new"]
        self.masks_cmap = self.mpl_cm.get_cmap(
            self.masks_params[self.key]["cmap"])
        self.masks_scalar_map = self.mpl_cm.ScalarMappable(
            cmap=self.masks_cmap, norm=self.params[self.key]["norm"])
        self.update_colors({"new": self.slider.value})

    def update_masks_solid_color(self, change):
        self.masks_cmap = self.mpl_colors.LinearSegmentedColormap.from_list(
            "tmp", [change["new"], change["new"]])
        self.masks_cmap.set_bad(color=change["new"])
        self.masks_scalar_map = self.mpl_cm.ScalarMappable(
            cmap=self.masks_cmap, norm=self.params[self.key]["norm"])
        self.update_colors({"new": self.slider.value})

    def toggle_color_selection(self, change):
        is_colormap = change["new"] == "colormap"
        self.masks_colormap.disabled = not is_colormap
        self.masks_solid_color.disabled = is_colormap
        if is_colormap:
            self.update_masks_colormap({"new": self.masks_colormap.value})
        else:
            self.update_masks_solid_color(
                {"new": self.masks_solid_color.value})

    def update_opacity(self, change):
        self.material.opacity = change["new"]
