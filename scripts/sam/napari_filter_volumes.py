import json
import logging
import os

import click
import napari
import numpy as np
import tifffile as tiff
from qtpy.QtWidgets import QPushButton
from glob import glob
from pathlib import Path

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def get_file_id(x):
    x = Path(x).stem
    return int(x.split("_")[1])


class KEYBINDS(object):
    PREV_SAMPLE = "l"
    NEXT_SAMPLE = "m"
    NEXT_50_SAMPLE = "n"
    DISMISS = "e"
    KEEP = "k"
    THRESHOLD1 = "&"
    THRESHOLD2 = "z"
    THRESHOLD3 = "\""


class Annotator3D(object):
    def __init__(
        self,
        im_folder,
        json_data,
    ):
        self.im_folder = im_folder
        self.json_data_path = json_data
        self.current_error_idx = 0

        self.data = None
        with open(self.json_data_path, "r") as fp:
            self.data = json.load(fp)
        print(glob(os.path.join(self.im_folder, "*")))
        self.crops_paths = sorted(glob(os.path.join(self.im_folder, "*")), key=get_file_id)

        print(f"Loaded #{len(self.data)} datapoints")

        self.img = np.zeros((1, 1, 1), dtype=np.uint8)
        self.create_viewer()

        # Initialise empty points layers
        # self.lock_points_view_button = QPushButton("Lock Points Loading")
        self.prev_button = QPushButton("(L) Previous")
        self.next_button = QPushButton("(M) Next")
        self.error_button = QPushButton("(D)ismiss")
        self.valid_button = QPushButton("(K) keep")
        # self.threshold1 = QPushButton("(&) T1")
        # self.threshold2 = QPushButton("(é) T2")
        # self.threshold3 = QPushButton("(\") T3")

        self.viewer.window.add_dock_widget(self.prev_button, area="left")
        self.viewer.window.add_dock_widget(self.next_button, area="left")
        self.viewer.window.add_dock_widget(self.error_button, area="left")
        self.viewer.window.add_dock_widget(self.valid_button, area="left")
        # self.viewer.window.add_dock_widget(self.threshold1, area="left")
        # self.viewer.window.add_dock_widget(self.threshold2, area="left")
        # self.viewer.window.add_dock_widget(self.threshold3, area="left")

        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.error_button.clicked.connect(self.set_error)
        self.valid_button.clicked.connect(self.set_valid)
        # self.threshold1.clicked.connect(self.set_threshold1)
        # self.threshold2.clicked.connect(self.set_threshold2)
        # self.threshold3.clicked.connect(self.set_threshold3)

        self.load_data(self.current_error_idx)
        self.start()
        
    def update_json(self):
        with open(self.json_data_path, "w") as fp:
            json.dump(self.data, fp)

    def update_label(self, idx):
        data = self.data[idx]
        status_msg = "dismiss" if data["keep"] == 0 else "keep"
        threshold = f" th: {data['threshold']}"

        status_message = f"i={idx}/{len(self.crops_paths)} Decision: {status_msg}"
        if data["keep"] == 1:
            status_message += threshold
        self.viewer.text_overlay.text = status_message

    def load_data(self, idx):
        print(f"Loading data crop: {idx}")
        if len(self.data) <= idx:
            self.data.append({
                "folder": self.crops_paths[idx],
                "keep": 0,
                "threshold": 0
            })

        data = self.data[idx]
        self.update_label(idx)

        # Load im crop
        im_path = os.path.join(data['folder'], "crop.tif")
        label_path = os.path.join(data['folder'], "auto_label.tif")

        im = tiff.imread(im_path)
        label = tiff.imread(label_path)

        self.image_layer.data = im
        self.label_layer.data = label

        current_step = im.shape[0] // 2
        print(f"Set view current step {current_step}")
        self.viewer.dims.set_current_step(0, current_step)
        self.viewer.camera.center = (
            im.shape[0] // 2,
            im.shape[1] // 2,
            im.shape[2] // 2,
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
        def skip_sample(viewer):
            self.next_sample()

        @self.viewer.bind_key(KEYBINDS.NEXT_50_SAMPLE)
        def skip_50_sample(viewer):
            self.next_50_sample()

        @self.viewer.bind_key(KEYBINDS.DISMISS)
        def set_real_error(viewer):
            self.set_error()
            self.next_sample()

        @self.viewer.bind_key(KEYBINDS.KEEP)
        def set_false_error(viewer):
            self.set_valid()
            self.set_threshold(3)
            self.next_sample()
            
        @self.viewer.bind_key(KEYBINDS.THRESHOLD1)
        def set_threshold1(viewer):
            self.set_threshold(1)

        @self.viewer.bind_key(KEYBINDS.THRESHOLD2)
        def set_threshold2(viewer):
            self.set_threshold(2)

        @self.viewer.bind_key(KEYBINDS.THRESHOLD3)
        def set_threshold3(viewer):
            self.set_threshold(3)

    def refresh(self):
        self.update_json()
        self.update_label(self.current_error_idx)

    def set_threshold(self, value):
        self.data[self.current_error_idx]["threshold"] = value
        self.refresh()

    def set_error(self):
        self.data[self.current_error_idx]["keep"] = 0
        self.refresh()

    def set_valid(self):
        self.data[self.current_error_idx]["keep"] = 1
        self.refresh()

    def prev_sample(self):
        self.current_error_idx = self.current_error_idx - 1
        if self.current_error_idx < 0:
            self.current_error_idx = len(self.crops_paths) - 1
        self.load_data(self.current_error_idx)

    def next_sample(self):
        self.current_error_idx = (self.current_error_idx + 1) % len(self.crops_paths)
        self.load_data(self.current_error_idx)
        
    def next_50_sample(self):
        self.current_error_idx = (self.current_error_idx + 50) % len(self.crops_paths)
        self.load_data(self.current_error_idx)

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
    "--json_data",
    required=False,
    default=None,
    help="Path to the json containing decisions.",
)
def main(im_folder, json_data):
    Annotator3D(im_folder, json_data)


if __name__ == "__main__":
    main()
