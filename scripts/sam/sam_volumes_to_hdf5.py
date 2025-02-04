import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import click
import h5py
import json
import tifffile as tif


def keep_center_cc(binary_mask):
    from skimage.measure import label

    labels = label(binary_mask)
    label_id = labels[labels.shape[0] // 2, labels.shape[1] // 2, labels.shape[2] // 2]
    mask = (labels == label_id).astype(np.uint8)
    return mask


def add_dataset(hf, group_name, data):
    grp = hf.create_group(group_name)
    im_grp = grp.create_group("im")
    label_grp = grp.create_group("label")

    print(f"Creating dataset of length #{len(data)} in group: {group_name}")
    for sample in tqdm(data):
        path = sample["folder"]
        folder_name = os.path.basename(path)

        x_path = os.path.join(path, "crop.tif")
        y_path = os.path.join(path, "auto_label.tif")

        x = tif.imread(x_path)
        y = tif.imread(y_path)

        threshold = (255.0 * sample["threshold"] / 3) - 1
        y_binary = np.where(y > threshold, 1, 0)
        y_binary = keep_center_cc(y_binary).astype(np.int64)

        im_grp.create_dataset(f"{folder_name}", data=x, compression="gzip", chunks=True)
        label_grp.create_dataset(f"{folder_name}", data=y_binary, compression="gzip", chunks=True)


@click.command()
@click.option(
    "--json_path",
    required=True,
    help="Path to the annotated json data crops",
)
@click.option(
    "--output_hdf5",
    required=True,
    help="Path to the output hdf5 file",
)
def main(json_path, output_hdf5):
    # Read data json
    with open(json_path, "r") as fp:
        data = json.load(fp)

    # Get all keep == 1
    correct_data = []
    for sample in data:
        if sample["keep"] == 1:
            correct_data.append(sample)
    fraction = len(correct_data) // 2

    # Split in two sets train/val
    train_data = correct_data[:-fraction]
    val_data = correct_data[-fraction:]

    # Load and write to hdf5
    with h5py.File(output_hdf5, "w") as hf:
        add_dataset(hf, "train", train_data)
        add_dataset(hf, "val", val_data)


if __name__ == "__main__":
    main()
