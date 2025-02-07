from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

import pandas as pd
import tifffile as tif
import numpy as np
from tqdm import tqdm


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


def get_sphere_template(radius):
    sphere_array = np.zeros((radius * 2 + 1, radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    center_x, center_y, center_z = (
        sphere_array.shape[0] // 2,
        sphere_array.shape[1] // 2,
        sphere_array.shape[2] // 2,
    )
    X, Y, Z = np.ogrid[: sphere_array.shape[0], : sphere_array.shape[1], : sphere_array.shape[2]]
    mat_distance = (X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2
    sphere_array[mat_distance <= radius**2] = 1
    return sphere_array


def format_groundtruth(x, data):
    # TODO use temp dir to store the tmp tif file
    y = tif.memmap("tmp.tif", shape=x.shape, dtype="uint8")

    sphere_template = get_sphere_template(radius=8)
    for _, center in tqdm(data.iterrows()):
        center_x, center_y, center_z = (
            s2i(center["Xcoords"]),
            s2i(center["Ycoords"]),
            s2i(center["Zcoords"]),
        )
        center = center_x, center_y, center_z
        if not point_in_array(center, x):
            continue

        y = draw_sphere_at(y, sphere_template, center)

    return x, y


def draw_sphere_at(mat, sphere_template, center):
    center_x, center_y, center_z = center
    # Draw sphere at given point
    mat_start_x, mat_end_x, patch_start_x, patch_end_x = get_slice(
        center_x, mat.shape[0], sphere_template.shape[0]
    )
    mat_start_y, mat_end_y, patch_start_y, patch_end_y = get_slice(
        center_y, mat.shape[1], sphere_template.shape[1]
    )
    mat_start_z, mat_end_z, patch_start_z, patch_end_z = get_slice(
        center_z, mat.shape[2], sphere_template.shape[2]
    )
    mat[mat_start_x:mat_end_x, mat_start_y:mat_end_y, mat_start_z:mat_end_z] = np.maximum(
        mat[mat_start_x:mat_end_x, mat_start_y:mat_end_y, mat_start_z:mat_end_z],
        sphere_template[
            patch_start_x:patch_end_x,
            patch_start_y:patch_end_y,
            patch_start_z:patch_end_z,
        ],
    )
    return mat


def get_slice(pos, array_dim, patch_dim):
    half_patch_size = patch_dim // 2
    array_start = max(0, pos - half_patch_size)
    patch_start = half_patch_size - (pos - array_start)
    array_end = min(array_dim - 1, pos + half_patch_size)
    patch_end = array_end - pos + half_patch_size

    assert array_start <= array_end, print(f"start: {array_start}, end: {array_end}")
    assert patch_start <= patch_end, print(f"start: {patch_start}, end: {patch_end}")

    return array_start, array_end, patch_start, patch_end
