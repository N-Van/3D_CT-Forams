import click
import tifffile as tif
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import os
import napari


def parse_folder_name(name):
    splits = name.split("_")
    x = int(splits[3])
    y = int(splits[5])
    z = int(splits[7])
    return x, y, z


def get_random_slice(shape, size):
    start = np.random.randint(0, shape - size - 1)
    end = start + size
    return slice(start, end)


def get_random_crop(x, label, size=64):
    x_slice = get_random_slice(x.shape[0], size)
    y_slice = get_random_slice(x.shape[1], size)
    z_slice = get_random_slice(x.shape[2], size)

    x_crop = x[x_slice, y_slice, z_slice]
    label_crop = label[x_slice, y_slice, z_slice]

    return x_crop, label_crop


@click.command()
@click.option(
    "--im_path",
    required=True,
    help="Path to the image to format",
)
@click.option(
    "--volumes_folder",
    required=True,
    help="Path to the volumes folder",
)
def main(im_path, volumes_folder):
    x = tif.memmap(im_path)
    x = np.swapaxes(x, 0, -1)

    min_quality_ratio = 0.1
    high_proba = 0.8

    valid_crop_dir = "output/true"
    os.makedirs(valid_crop_dir, exist_ok=True)
    im_dir = os.path.join(valid_crop_dir, "im")
    label_dir = os.path.join(valid_crop_dir, "label")

    os.makedirs(im_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    non_zero_labels = np.zeros_like(x, dtype=np.uint8)

    for crop_folder in tqdm(glob(os.path.join(volumes_folder, "*"))):
        folder_name = Path(crop_folder).stem
        coords = np.array(parse_folder_name(folder_name))
        crop_label = os.path.join(crop_folder, "auto_label.tif")
        crop = tif.imread(crop_label)

        crop_center = crop.shape[0] // 2
        crop_start = coords - crop_center

        x_slice = slice(crop_start[0], crop_start[0] + crop.shape[0])
        y_slice = slice(crop_start[1], crop_start[1] + crop.shape[1])
        z_slice = slice(crop_start[2], crop_start[2] + crop.shape[2])
        x_crop = x[x_slice, y_slice, z_slice]

        non_zero_crop = crop[np.nonzero(crop)]
        non_zero_count = np.count_nonzero(non_zero_crop)
        non_zero_high_prob_count = np.count_nonzero(non_zero_crop[non_zero_crop > high_proba])
        quality_ratio = float(non_zero_high_prob_count) / float(non_zero_count)

        if quality_ratio > min_quality_ratio:
            tif.imwrite(os.path.join(im_dir, f"{folder_name}.tif"), x_crop)
            tif.imwrite(os.path.join(label_dir, f"{folder_name}.tif"), crop)

        # Mark this area as non empty
        non_zero_labels[x_slice, y_slice, z_slice] = 255

    n_negative_samples = 2000
    # Random negative samples
    current_negative_samples = 0
    negative_label = np.zeros_like(crop, np.uint8)
    while current_negative_samples < n_negative_samples:
        # Draw random crop
        x_crop, label_crop = get_random_crop(x, non_zero_labels)

        # Not fully negative then skip it
        if np.count_nonzero(label_crop) > 0:
            continue

        # If it is negative add it
        tif.imwrite(os.path.join(im_dir, f"negative_{current_negative_samples}.tif"), x_crop)
        tif.imwrite(
            os.path.join(label_dir, f"negative_{current_negative_samples}.tif"), negative_label
        )

        if current_negative_samples % 100 == 0:
            print(current_negative_samples)

        current_negative_samples += 1


if __name__ == "__main__":
    main()
