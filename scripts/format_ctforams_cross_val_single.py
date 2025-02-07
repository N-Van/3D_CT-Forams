from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import click
import h5py


from common.utils import load, format_groundtruth


def get_split(x, y, s):
    assert len(x.shape) == 3 and len(y.shape) == 3
    return x[:, :, s], y[:, :, s]


def swapaxes(x, y, dim):
    x = np.swapaxes(x, -1, dim)
    y = np.swapaxes(y, -1, dim)
    return x, y


def split_data(x, y, kfold, fold_index, dim=1):
    print(f"Split using dimension {dim} of shape {x.shape}")
    x = np.swapaxes(x, dim, -1)
    y = np.swapaxes(y, dim, -1)

    print(x.shape, y.shape)

    fold_size = x.shape[-1] // kfold
    assert 0 <= fold_index < kfold, "Invalid fold_index. Should be in the range 0 to k-1."

    # X, Y, Z
    val_start = fold_index * fold_size
    val_end = (fold_index + 1) * fold_size

    # Handle last slice case
    if fold_index == kfold - 1:
        val_end = x.shape[-1]

    val_slice = slice(val_start, val_end)
    train_slice_left = slice(0, val_start)
    train_slice_right = slice(val_slice.stop, x.shape[-1])

    print(f"Validation slice {val_slice}")
    print(f"Training slice {train_slice_left} ; {train_slice_right}")

    x_val, y_val = get_split(x, y, val_slice)
    x_train_left, y_train_left = get_split(x, y, train_slice_left)
    x_train_right, y_train_right = get_split(x, y, train_slice_right)

    # Now move axes back to original positions
    x_val, y_val = swapaxes(x_val, y_val, dim)
    x_train_left, y_train_left = swapaxes(x_train_left, y_train_left, dim)
    x_train_right, y_train_right = swapaxes(x_train_right, y_train_right, dim)

    x_trains, y_trains = [], []
    if x_train_left.shape[dim] != 0:
        x_trains.append(x_train_left)
        y_trains.append(y_train_left)
    if x_train_right.shape[dim] != 0:
        x_trains.append(x_train_right)
        y_trains.append(y_train_right)

    assert x_val.shape[dim] > 0 and y_val.shape[0] > 0
    return x_trains, y_trains, x_val, y_val


def generate_split(x, y, kfold, fold_index, output_folder, basename, h5_path, dim):
    x_train, y_train, x_val, y_val = split_data(x, y, kfold, fold_index, dim=dim)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        train_grp = hf.create_group("train")
        train_im_grp = train_grp.create_group("im")
        train_label_grp = train_grp.create_group("label")
        print("Writing training data...")
        if not isinstance(x_train, list):
            add_dataset(train_im_grp, train_label_grp, x_train, y_train, f"{basename}")
        else:
            for i in range(len(x_train)):
                add_dataset(
                    train_im_grp, train_label_grp, x_train[i], y_train[i], f"{basename}_{i}"
                )

        val_grp = hf.create_group("val")
        val_im_grp = val_grp.create_group("im")
        val_label_grp = val_grp.create_group("label")
        print("Writing validation data...")
        val_im_grp.create_dataset(basename, data=x_val, compression="gzip", chunks=True)
        val_label_grp.create_dataset(basename, data=y_val, compression="gzip", chunks=True)

        hf.flush()
        hf.close()


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
def main(im_path, csv_path, output_folder, kfold, kindex, dim):
    # Load data
    x, y = load(im_path, csv_path)
    # Format groundtruth
    x, y = format_groundtruth(x, y)

    basename = Path(im_path).stem

    if kindex == -1:
        for i in range(kfold):
            h5_path = os.path.join(output_folder, f"{basename}_{i}.h5")
            generate_split(x, y, kfold, i, output_folder, basename, h5_path, dim=dim)
    else:
        h5_path = os.path.join(output_folder, f"{basename}_{kindex}.h5")
        generate_split(x, y, kfold, kindex, output_folder, basename, h5_path, dim=dim)


if __name__ == "__main__":
    main()
