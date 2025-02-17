import json
import logging
import os
import time
from copy import deepcopy

import click
import napari
import numpy as np
import tifffile as tiff
from napari.components.viewer_model import ViewerModel
from napari.layers import Labels, Layer
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QPushButton, QSplitter, QSlider

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin=None,
        layer_type=None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(filenames, stack, plugin, layer_type, **kwargs)


def copy_layer(layer: Layer, name: str = ""):
    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class ForamAnimator:
    def __init__(self, function, interval):
        self.function = function
        self.interval = interval
        self._running = False
        self._thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.function)

    def start_process(self):
        if not self.timer.isActive():
            self.timer.start(self.interval)
            print("Process started.")

    def stop_process(self):
        if self.timer.isActive():
            self.timer.stop()
            print("Process stopped.")

    def get_status(self):
        return self.timer.isActive()


class KEYBINDS(object):
    PREV_SAMPLE = "l"
    NEXT_SAMPLE = "m"
    NEXT_SAMPLE_ALT = "n"
    REAL_ERROR = "e"
    FALSE_ERROR = "v"
    NEXT_N_SAMPLE = "j"


class Annotator3D(object):
    def __init__(
        self,
        im_folder,
        label_folder,
        errors_data,
        crop_size=32,
    ):
        self.im_folder = im_folder
        self.label_folder = label_folder
        self.errors_data_path = errors_data
        self.current_error_idx = 0
        self.crop_size = crop_size
        self._block = False

        self.errors_data = None

        self.load_errors()
        print(f"Loaded #{len(self.errors_data)} errors")

        self.img = np.zeros((1, 1, 1), dtype=np.uint8)
        self.create_viewer()

        self.animation_step_delta = 4
        self.animation_step_direction = 1
        self.animation_slider = QSlider(Qt.Horizontal)
        self.animation_slider.setMinimum(50)
        self.animation_slider.setMaximum(500)
        self.animation_slider.setTickInterval(10)

        self.default_animation_ms = 150
        self.animation_slider.setValue(self.default_animation_ms)
        self.animator = ForamAnimator(self.animation_step, self.default_animation_ms)

        # Order matters ! used in next_sample()
        # self.current_error_status = QLabel("")

        # Initialise empty points layers
        # self.lock_points_view_button = QPushButton("Lock Points Loading")
        # self.prev_button = QPushButton("(L) Previous")
        # self.next_button = QPushButton("(M) Next")
        # self.error_button = QPushButton("(E)rror")
        # self.valid_button = QPushButton("(V)alid")
        self.animation_button = QPushButton("Toggle Animation")

        # self.lock_points_view_button.clicked.connect(self.toggle_lock_point_view)
        # self.viewer.window.add_dock_widget(self.lock_points_view_button, area="left")

        # self.viewer.window.add_dock_widget(self.prev_button, area="left")
        # self.viewer.window.add_dock_widget(self.next_button, area="left")
        # self.viewer.window.add_dock_widget(self.error_button, area="left")
        # self.viewer.window.add_dock_widget(self.valid_button, area="left")
        self.viewer.window.add_dock_widget(self.animation_button, area="left")
        # self.viewer.window.add_dock_widget(self.current_error_status, area="left")
        self.viewer.window.add_dock_widget(self.animation_slider, area="left")

        # self.prev_button.clicked.connect(self.prev_sample)
        # self.next_button.clicked.connect(self.next_sample)
        # self.error_button.clicked.connect(self.set_error)
        # self.error_button.clicked.connect(self.set_valid)
        self.animation_button.clicked.connect(self.toggle_animation)
        self.animation_slider.valueChanged.connect(self.update_animation_interval)

        self.load_error(self.current_error_idx)
        self.start()

    def update_animation_interval(self, value):
        self.animator.timer.setInterval(value)

    def toggle_animation(self):
        if self.animator.get_status():
            self.animator.stop_process()
        else:
            self.animator.start_process()

    def animation_step(self):
        # Get the next step
        # If we reach the lowest point
        crop_shape = self.image_layer.data.shape
        axis = self.viewer.dims.order[0]
        current_step = self.viewer.dims.current_step[axis]
        center_step = crop_shape[axis] // 2

        lower_bound = center_step - self.animation_step_delta
        upper_bound = center_step + self.animation_step_delta

        next_step = current_step
        if current_step <= lower_bound:
            self.animation_step_direction = 1
        elif current_step >= upper_bound:
            self.animation_step_direction = -1

        next_step += self.animation_step_direction
        self.viewer.dims.set_current_step(axis, next_step)

    def set_error(self):
        self.errors_data[self.current_error_idx]["corrected"] = 1
        self.update_json()
        self.update_label(self.current_error_idx)
        self.next_sample()

    def set_valid(self):
        self.errors_data[self.current_error_idx]["corrected"] = 0
        self.update_json()
        self.update_label(self.current_error_idx)
        self.next_sample()

    def load_errors(self):
        with open(self.errors_data_path, "r") as fp:
            self.errors_data = json.load(fp)

    def update_json(self):
        with open(self.errors_data_path, "w") as fp:
            json.dump(self.errors_data, fp)

    def prev_sample(self):
        self.current_error_idx = self.current_error_idx - 1
        if self.current_error_idx < 0:
            self.current_error_idx = len(self.errors_data) - 1
        self.load_error(self.current_error_idx)

    def next_sample(self, amount=1):
        self.current_error_idx = (self.current_error_idx + amount) % len(self.errors_data)
        self.load_error(self.current_error_idx)

    def update_label(self, idx):
        error_data = self.errors_data[idx]

        status = error_data["corrected"]
        status_msg = "undecided" if status == -1 else "valid" if status == 0 else "error"
        error_type = error_data["type"]
        # vol_frac = error_data["volumic_ratio"]

        status_message = f"i={idx}/{len(self.errors_data)} [{error_type}] decision: {status_msg}"
        # self.current_error_status.text = status_message
        self.viewer.text_overlay.text = status_message

    def get_crop(self, path, x, y, z):
        im = tiff.memmap(path)
        hc = self.crop_size // 2
        x_min = max(0, x - hc)
        x_pad_left = abs(x - hc - x_min)
        x_max = min(x + hc, im.shape[0] - 1)
        x_pad_right = x + hc - x_max
        y_min = max(0, y - hc)
        y_pad_left = abs(y - hc - y_min)
        y_max = min(y + hc, im.shape[1] - 1)
        y_pad_right = y + hc - y_max
        z_min = max(0, z - hc)
        z_pad_left = abs(z - hc - z_min)
        z_max = min(z + hc, im.shape[2] - 1)
        z_pad_right = z + hc - z_max
        crop = im[x_min:x_max, y_min:y_max, z_min:z_max]

        crop = np.pad(
            crop, ((x_pad_left, x_pad_right), (y_pad_left, y_pad_right), (z_pad_left, z_pad_right))
        )

        return crop

    def load_error(self, idx):
        print(f"Loading error: {idx}")
        error_data = self.errors_data[idx]
        self.update_label(idx)

        # Load im crop
        x, y, z = error_data["x"], error_data["y"], error_data["z"]
        im_path = os.path.join(self.im_folder, f"{error_data['file']}.tif")
        label_path = os.path.join(self.label_folder, f"{error_data['file']}.tif")

        im_crop = self.get_crop(im_path, x, y, z)
        self.image_layer.data = im_crop
        self.label_layer.data = self.get_crop(label_path, x, y, z)

        current_step = im_crop.shape[0] // 2
        self.viewer.dims.set_current_step(0, current_step)
        self.viewer.camera.center = (
            im_crop.shape[0] // 2,
            im_crop.shape[1] // 2,
            im_crop.shape[2] // 2,
        )
        self.viewer.camera.zoom = 8

    def create_viewer(self):
        self.viewer = napari.Viewer()
        self.setup_multiviews()

        self.image_layer = self.viewer.add_image(
            self.img,
            name="image",
            # colormap="inferno",
        )
        self.label_layer = self.viewer.add_image(
            self.img, name="label", colormap="inferno", opacity=0.5
        )

        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 20
        self.viewer.text_overlay.color = (1.0, 1.0, 1.0, 1.0)

        @self.viewer.bind_key(KEYBINDS.PREV_SAMPLE)
        def do_prev_sample(viewer):
            self.prev_sample()

        @self.viewer.bind_key(KEYBINDS.NEXT_SAMPLE)
        @self.viewer.bind_key(KEYBINDS.NEXT_SAMPLE_ALT)
        def skip_sample(viewer):
            self.next_sample()

        @self.viewer.bind_key(KEYBINDS.REAL_ERROR)
        def set_real_error(viewer):
            self.set_error()

        @self.viewer.bind_key(KEYBINDS.FALSE_ERROR)
        def set_false_error(viewer):
            self.set_valid()

        @self.viewer.bind_key(KEYBINDS.NEXT_N_SAMPLE)
        def skip_n_sample(viewer):
            self.next_sample(50)

    def setup_multiviews(self):
        self.viewer_model1 = ViewerModel(title="model1")
        self.viewer_model2 = ViewerModel(title="model2")

        self.qt_viewer1 = QtViewerWrap(self.viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(self.viewer, self.viewer_model2)
        viewer_splitter = QSplitter()
        viewer_splitter.setOrientation(Qt.Orientation.Vertical)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.addWidget(self.qt_viewer2)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        self.viewer.window.add_dock_widget(viewer_splitter, name="views")

        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(self._layer_selection_changed)
        # self.viewer.dims.events.current_step.connect(self._point_update)
        # self.viewer_model1.dims.events.current_step.connect(self._point_update)
        # self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)

        # Sync zoom
        self.viewer.camera.events.zoom.connect(self._zoom_update)
        self.viewer_model1.camera.events.zoom.connect(self._zoom_update)
        self.viewer_model2.camera.events.zoom.connect(self._zoom_update)

        # Sync center
        self.viewer.camera.events.center.connect(self._center_update)
        self.viewer_model1.camera.events.center.connect(self._center_update)
        self.viewer_model2.camera.events.center.connect(self._center_update)

        self.viewer.dims.events.current_step.connect(self._update_current_step)
        self.viewer_model1.dims.events.current_step.connect(self._update_current_step)
        self.viewer_model2.dims.events.current_step.connect(self._update_current_step)

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            return

        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[event.value.name]
        self.viewer_model2.layers.selection.active = self.viewer_model2.layers[event.value.name]

    def _update_current_step(self, event):
        print("update current step")
        print(event.value)
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.dims is event.source:
                print("skip model")
                continue
            axis_value = event.value[event.source.order[0]]
            axis = model.dims.order[0]
            model.dims.set_current_step(axis, axis_value)

    def _center_update(self, event):
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.camera is event.source:
                continue
            model.camera.center = event.value

    def _zoom_update(self, event):
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.camera is event.source:
                continue
            model.camera.zoom = event.value

    def _point_update(self, event):
        print("Update point")
        print(event.value)
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = tuple(order)
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = tuple(order)

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        self.viewer_model1.layers.insert(event.index, copy_layer(event.value, "model1"))
        self.viewer_model2.layers.insert(event.index, copy_layer(event.value, "model2"))
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(own_partial(self._property_sync, name))

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            event.value.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[event.value.name].events.set_data.connect(
                self._set_data_refresh
            )
            self.viewer_model2.layers[event.value.name].events.set_data.connect(
                self._set_data_refresh
            )
            event.value.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[event.value.name].events.labels_update.connect(
                self._set_data_refresh
            )
            self.viewer_model2.layers[event.value.name].events.labels_update.connect(
                self._set_data_refresh
            )
        if event.value.name != ".cross":
            self.viewer_model1.layers[event.value.name].events.data.connect(self._sync_data)
            self.viewer_model2.layers[event.value.name].events.data.connect(self._sync_data)

        event.value.events.name.connect(self._sync_name)
        self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = event.new_index if event.new_index < event.index else event.new_index + 1
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(
                self.viewer_model1.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
            setattr(
                self.viewer_model2.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False

    def start(self):
        napari.run()
        logger.info("Napari viewer started !")


@click.command()
@click.option(
    "--im_folder",
    required=True,
    help="Path to image folder to use.",
)
@click.option(
    "--label_folder",
    required=True,
    help="Path to label folder to use.",
)
@click.option(
    "--errors_data",
    required=False,
    default=None,
    help="Path to the json of errors to correct.",
)
def main(im_folder, label_folder, errors_data):
    Annotator3D(im_folder, label_folder, errors_data)


if __name__ == "__main__":
    main()
