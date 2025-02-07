import json
import logging
import os

import click
import napari
import numpy as np
import tifffile as tiff
from qtpy.QtWidgets import QPushButton

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


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

        self.errors_data = None

        self.load_errors()
        print(f"Loaded #{len(self.errors_data)} errors")

        self.img = np.zeros((1, 1, 1), dtype=np.uint8)
        self.create_viewer()

        # Order matters ! used in next_sample()
        # self.current_error_status = QLabel("")

        # Initialise empty points layers
        # self.lock_points_view_button = QPushButton("Lock Points Loading")
        self.prev_button = QPushButton("(L) Previous")
        self.next_button = QPushButton("(M) Next")
        self.error_button = QPushButton("(E)rror")
        self.valid_button = QPushButton("(V)alid")

        # self.lock_points_view_button.clicked.connect(self.toggle_lock_point_view)
        # self.viewer.window.add_dock_widget(self.lock_points_view_button, area="left")

        self.viewer.window.add_dock_widget(self.prev_button, area="left")
        self.viewer.window.add_dock_widget(self.next_button, area="left")
        self.viewer.window.add_dock_widget(self.error_button, area="left")
        self.viewer.window.add_dock_widget(self.valid_button, area="left")
        # self.viewer.window.add_dock_widget(self.current_error_status, area="left")

        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.error_button.clicked.connect(self.set_error)
        self.error_button.clicked.connect(self.set_valid)

        self.load_error(self.current_error_idx)
        self.start()

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
        vol_frac = error_data["volumic_ratio"]

        status_message = f"i={idx}/{len(self.errors_data)} [{error_type}] decision: {status_msg}\nvol frac={vol_frac}"
        # self.current_error_status.text = status_message
        self.viewer.text_overlay.text = status_message

    def get_crop(self, path, x, y, z):
        im = tiff.memmap(path)
        hc = self.crop_size // 2
        print(f"Cropping image shape {im.shape}")
        print(f"Coords: x=[{x-hc};{x+hc}] y=[{y-hc};{y+hc}] z=[{z-hc};{z+hc}]")
        x_min = max(0, x - hc)
        x_max = min(x + hc, im.shape[0] - 1)
        y_min = max(0, y - hc)
        y_max = min(y + hc, im.shape[1] - 1)
        z_min = max(0, z - hc)
        z_max = min(z + hc, im.shape[2] - 1)
        crop = im[x_min:x_max, y_min:y_max, z_min:z_max]
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
        print(f"Set view current step {current_step}")
        self.viewer.dims.set_current_step(0, current_step)
        self.viewer.camera.center = (
            im_crop.shape[0] // 2,
            im_crop.shape[1] // 2,
            im_crop.shape[2] // 2,
        )
        self.viewer.camera.zoom = 8

    def create_viewer(self):
        self.viewer = napari.Viewer()

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
