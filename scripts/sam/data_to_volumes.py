import numpy as np
import pandas as pd
import tifffile as tif
from tqdm import tqdm
import os
import click
from adaptsam import AdaptSAMPredictor


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


def generate_volumes(x, data, predictor, output_dir, crop_size):
    half = crop_size // 2

    im_folder = os.path.join(output_dir, "images")
    mask_folder = os.path.join(output_dir, "masks")

    os.makedirs(im_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    for i, center in tqdm(data.iterrows()):
        center_x, center_y, center_z = (
            s2i(center["Xcoords"]),
            s2i(center["Ycoords"]),
            s2i(center["Zcoords"]),
        )
        center = center_x, center_y, center_z
        if not point_in_array(center, x):
            continue

        crop = x[
            center_x - half : center_x + half,
            center_y - half : center_y + half,
            center_z - half : center_z + half,
        ]

        if crop.shape != (crop_size, crop_size, crop_size):
            print(crop.shape)
            print(f"Skipping crop")
            continue

        point_prompt = [half, half, half]

        # Generate the prediction
        prediction = predictor.predict(crop, point_prompt)
        name = f"crop_{i}_x_{center_x}_y_{center_y}_z_{center_z}.tif"
        tif.imwrite(os.path.join(im_folder, name), crop)
        tif.imwrite(os.path.join(mask_folder, name), prediction)


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
    "--model_cfg",
    required=True,
    help="Path to the SAM2 model config",
)
@click.option(
    "--sam2_checkpoint",
    required=True,
    help="Path to the SAM2 checkpoint",
)
@click.option(
    "--output_folder",
    required=True,
    help="Path to the output folder",
)
@click.option(
    "--crop_size",
    required=True,
    default=64,
    help="Crop size around the center",
)
def main(im_path, csv_path, model_cfg, sam2_checkpoint, output_folder, crop_size):
    predictor = AdaptSAMPredictor(
        model_cfg=model_cfg,
        sam2_checkpoint=sam2_checkpoint,
    )

    os.makedirs(output_folder, exist_ok=True)

    x, y = load(im_path, csv_path)
    generate_volumes(x, y, predictor, output_folder, crop_size)


if __name__ == "__main__":
    main()
