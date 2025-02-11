import numpy as np
import pandas as pd
import tifffile as tif
from scipy import ndimage as ndi
from tqdm import tqdm
import os
from pathlib import Path
import click
import cv2


def load(im_path, csv_path):
    y = pd.read_csv(csv_path, sep=";")
    x = tif.memmap(im_path)
    # From Z, Y, X to X, Y, Z
    x = np.swapaxes(x, 0, -1)
    return x, y


def s2i(value):
    if isinstance(value, str):
        value = value.replace(",", ".")
    return int(float(value))


def pt_in_dim(pt, dim):
    return pt >= 0 and pt < dim


def point_in_array(pt, array):
    for i_dim in range(len(array.shape)):
        if not pt_in_dim(pt[i_dim], array.shape[i_dim]):
            # print(f"Discard point {pt} since at dim {i_dim} point {pt[i_dim]} not in (0;{array.shape[i_dim]})")
            return False
    return True


def get_slice(pos, array_dim, patch_dim):
    half_patch_size = patch_dim // 2
    array_start = max(0, pos - half_patch_size)
    patch_start = half_patch_size - (pos - array_start)
    array_end = min(array_dim - 1, pos + half_patch_size)
    patch_end = array_end - pos + half_patch_size

    assert array_start <= array_end, print(f"start: {array_start}, end: {array_end}")
    assert patch_start <= patch_end, print(f"start: {patch_start}, end: {patch_end}")

    return array_start, array_end, patch_start, patch_end


def format_groundtruth(x, data, output_dir):
    crop_size = 64
    half = crop_size // 2
    for i, center in tqdm(data.iterrows()):
        center_x, center_y, center_z = (
            s2i(center["Xcoords"]),
            s2i(center["Ycoords"]),
            s2i(center["Zcoords"]),
        )
        center = center_x, center_y, center_z
        if not point_in_array(center, x):
            continue

        # Get crop
        crop_dir = os.path.join(output_dir, f"crop_{i}_x_{center_x}_y_{center_y}_z_{center_z}")

        crop = x[
            center_x - half : center_x + half,
            center_y - half : center_y + half,
            center_z - half : center_z + half,
        ]

        if crop.shape != (crop_size, crop_size, crop_size):
            print(crop.shape)
            print(f"Skipping crop {crop_dir}")
            continue

        os.makedirs(crop_dir, exist_ok=True)

        # X,Y,Z = 0, 1, 2
        # Y,Z,X = 1, 2, 0
        # X,Z,Y = 0, 2, 1
        rot_axes = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]

        for k in range(3):
            crop_dir_plane = os.path.join(crop_dir, f"plane_{k}")
            crop_dir_right = os.path.join(crop_dir_plane, "right")
            crop_dir_left = os.path.join(crop_dir_plane, "left")
            os.makedirs(crop_dir_plane, exist_ok=True)
            os.makedirs(crop_dir_left, exist_ok=True)
            os.makedirs(crop_dir_right, exist_ok=True)
            current_crop = np.moveaxis(crop, (0, 1, 2), rot_axes[k])
            for k in range(half):
                left = current_crop[
                    :,
                    :,
                    half - k,
                ]
                cv2.imwrite(os.path.join(crop_dir_left, f"{k}.jpg"), left)
                right = current_crop[
                    :,
                    :,
                    half + k,
                ]
                cv2.imwrite(os.path.join(crop_dir_right, f"{k}.jpg"), right)


@click.command()
@click.option(
    "--im_path",
    required=True,
    help="Path to the image to format",
)
@click.option(
    "--csv_path",
    required=True,
    help="Path to the annotation file in .csv",
)
@click.option(
    "--output_folder",
    required=True,
    help="Path to the output folder",
)
def main(im_path, csv_path, output_folder):
    x, y = load(im_path, csv_path)
    format_groundtruth(x, y, output_folder)


if __name__ == "__main__":
    main()
