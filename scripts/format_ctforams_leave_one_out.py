import os
from glob import glob
from pathlib import Path

import click
from common.utils import format_groundtruth, load
from format_ctforams_cross_val_single import generate_split
from itertools import combinations


def leave_one_out_combinations(lst):
    return [(lst[i], lst[:i] + lst[i + 1 :]) for i in range(len(lst))]


@click.command()
@click.option(
    "--im_folder",
    required=True,
    help="Path to the images folder",
)
@click.option(
    "--csv_folder",
    required=True,
    help="Path to the csv folder",
)
@click.option(
    "--output_folder",
    required=True,
    help="Path to the output folder",
)
@click.option(
    "--kfold",
    required=False,
    default=4,
    help="The number of cross validation splits",
)
@click.option(
    "--kindex",
    required=False,
    default=-1,
    help="Index of the cross validation split to generation. Default is -1 which means all splits will be generated. kindex is -1 or between [0;kfold-1]",
)
@click.option(
    "--dim",
    required=False,
    default=1,
    help="Axis number to use for the split. Example if dim=1 and order is X,Y,Z then we split following the Y axis.",
)
def main(im_folder, csv_folder, output_folder, kfold, kindex, dim):
    # Find all tif file in im folder
    images = glob(os.path.join(im_folder, "*.tif"))
    im_names = [Path(im).stem for im in images]

    # For all tif file verify that there is a csv file with the same name
    for im_name in im_names:
        csv_path = os.path.join(csv_folder, f"{im_name}.csv")
        assert os.path.exists(csv_path), f"Could not find {csv_path}"

    # Create all permutations possible
    permutations = leave_one_out_combinations(im_names)

    for out_name, train_list in permutations:
        # For each permutations (train size is n-1)
        basename = f"all_but_{out_name}"
        print(basename)
        print(train_list)

        basename_with_index = basename if kindex == -1 else f"{basename}_{kindex}"
        h5_path = os.path.join(output_folder, f"{basename_with_index}.h5")

        # For each data in train
        for train_name in train_list:
            im_path = os.path.join(im_folder, f"{train_name}.tif")
            csv_path = os.path.join(csv_folder, f"{train_name}.csv")

            # Load data
            x, y = load(im_path, csv_path)
            # Format groundtruth
            x, y = format_groundtruth(x, y)

            if kindex == -1:
                for i in range(kfold):
                    generate_split(x, y, kfold, i, output_folder, train_name, h5_path, dim=dim)
            else:
                generate_split(x, y, kfold, kindex, output_folder, train_name, h5_path, dim=dim)


if __name__ == "__main__":
    main()
