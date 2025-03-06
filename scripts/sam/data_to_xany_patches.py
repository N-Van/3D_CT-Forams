import os
import sys

import click
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils import load, point_in_array, s2i


def to_patches(x, data, output_dir, crop_size, axis):
    assert axis == "x" or axis == "y" or axis == "z", axis
    half_dim = crop_size // 2
    for i, center in tqdm(data.iterrows()):
        center_x, center_y, center_z = (
            s2i(center["Xcoords"]),
            s2i(center["Ycoords"]),
            s2i(center["Zcoords"]),
        )
        center = center_x, center_y, center_z
        if not point_in_array(center, x):
            continue

        if axis == "z":
            crop = x[
                center_x - half_dim : center_x + half_dim,
                center_y - half_dim : center_y + half_dim,
                center_z,
            ]
        elif axis == "x":
            crop = x[
                center_x,
                center_y - half_dim : center_y + half_dim,
                center_z - half_dim : center_z + half_dim,
            ]
        elif axis == "y":
            crop = x[
                center_x - half_dim : center_x + half_dim,
                center_y,
                center_z - half_dim : center_z + half_dim,
            ]
        cv2.imwrite(os.path.join(output_dir, f"crop{axis}_{i}.jpg"), crop)


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
@click.option("--crop_size", required=True, type=int, help="Crop size of the 2D image crop")
@click.option(
    "--axis",
    required=True,
    type=click.Choice(["x", "y", "z"], case_sensitive=True),
    help="Select the dimension you want to slice",
)
def main(im_path, csv_path, output_folder, crop_size, axis):
    x, y = load(im_path, csv_path)
    to_patches(x, y, output_folder, crop_size, axis)


if __name__ == "__main__":
    main()
