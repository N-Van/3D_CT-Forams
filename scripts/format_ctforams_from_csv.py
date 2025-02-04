import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import click
import h5py

from common.utils import load, format_groundtruth


def add_dataset(im_grp, label_grp, x, y, basename, step_size=250):
    n_split = max(1, int(np.ceil(x.shape[-1] / step_size)))
    for i in tqdm(range(n_split)):
        start = i * step_size
        end = min(start + step_size, x.shape[-1])
        s = slice(start, end)

        im_grp.create_dataset(f"{basename}_{i}", data=x[..., s], compression="gzip", chunks=True)
        label_grp.create_dataset(f"{basename}_{i}", data=y[..., s], compression="gzip", chunks=True)


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
@click.option(
    "--group_name",
    required=False,
    default="test",
    help="The name of the group in the hdf5 file. One of train/val/test",
)
def main(
    im_path,
    csv_path,
    output_folder,
    group_name,
):
    # Load training data
    print("Loading training data")
    x, y = load(im_path, csv_path)
    print("Format groundtruth from csv data points")
    x, y = format_groundtruth(x, y)

    print(x.shape, y.shape)

    basename = Path(im_path).stem

    # Write data to h5
    h5_path = os.path.join(output_folder, f"{basename}.h5")
    with h5py.File(h5_path, "w") as hf:
        test_grp = hf.create_group(group_name)
        test_im_grp = test_grp.create_group("im")
        test_label_grp = test_grp.create_group("label")
        print(f"Writing {group_name} data...")
        add_dataset(test_im_grp, test_label_grp, x, y, basename)


if __name__ == "__main__":
    main()
