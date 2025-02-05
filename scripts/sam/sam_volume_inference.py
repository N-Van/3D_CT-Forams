import click
import tifffile as tif
import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm

from scripts.common.utils import load, s2i, point_in_array


def load_model(model_path):
    print(f"Using device : {ort.get_device()}")
    ort_sess = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(ort_sess.get_providers())
    return ort_sess


def preprocess(batch):
    batch = batch.astype(np.float32) / 255.0
    # Add channel dim
    batch = np.expand_dims(batch, axis=1)
    return batch


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def inference(model, batch_image):
    batch_image = np.asarray(batch_image)
    batch_image = preprocess(batch_image)
    outputs = model.run(None, {"modelInput": batch_image})
    return sigmoid(outputs[0])


def get_slices_with_padding(center, dim_len, half_size):
    left = center - half_size
    right = center + half_size

    pad_left, pad_right = 0, 0
    # Padding if left < 0
    if left < 0:
        pad_left = abs(left)
        left = 0

    # Padding if right >= dim_len
    if right >= dim_len:
        pad_right = right - dim_len
        right = dim_len

    return slice(left, right), pad_left, pad_right


def remove_padding(crop, padding):
    (lpad_x, rpad_x), (lpad_y, rpad_y), (lpad_z, rpad_z) = padding
    assert len(crop.shape) == 4
    crop = crop[
        ...,
        lpad_x : crop.shape[-3] - rpad_x,
        lpad_y : crop.shape[-2] - rpad_y,
        lpad_z : crop.shape[-1] - rpad_z,
    ]
    return crop


def process_batch(y, batch_crop_y, batch_slices, batch_padding):
    for i in range(len(batch_crop_y)):
        crop_y = (batch_crop_y[i] * 255.0).astype(np.uint8)
        slices = batch_slices[i]
        crop_y = remove_padding(crop_y, batch_padding[i])

        # print(slices)
        # print(batch_padding[i])
        # print(crop_y.shape)

        # tif.imwrite("crop_x.tif", crop_x)
        # tif.imwrite("crop_y.tif", crop_y)
        # print("inference done")
        # input()

        y[slices] = np.maximum(crop_y, y[slices])


@click.command()
@click.option(
    "--im_path",
    required=True,
    help="Path to the annotated tif file",
)
@click.option(
    "--csv_path",
    required=True,
    help="Path to csv containing the centers",
)
@click.option(
    "--model_checkpoint",
    required=True,
    help="Path to the model .onnx checkpoint",
)
@click.option(
    "--crop_size",
    required=False,
    default=64,
    help="Crop size of the given model",
)
def main(im_path, csv_path, model_checkpoint, crop_size=64, batch_size=16):
    # Load 3D tif as memmap
    # Load csv data
    x, csv_data = load(im_path, csv_path)

    # Load model
    model = load_model(model_checkpoint)

    # Prepare label matrix
    y = tif.memmap("label.tif", shape=x.shape, dtype=np.uint8)
    half_crop = crop_size // 2

    batch = []
    batch_slices = []
    batch_padding = []

    for _, center in tqdm(csv_data.iterrows(), total=len(csv_data)):
        center_x, center_y, center_z = (
            s2i(center["Xcoords"]),
            s2i(center["Ycoords"]),
            s2i(center["Zcoords"]),
        )
        center = center_x, center_y, center_z
        if not point_in_array(center, x):
            continue

        # Compute slice and necessary padding
        slice_x, lpad_x, rpad_x = get_slices_with_padding(center_x, x.shape[0], half_crop)
        slice_y, lpad_y, rpad_y = get_slices_with_padding(center_y, x.shape[1], half_crop)
        slice_z, lpad_z, rpad_z = get_slices_with_padding(center_z, x.shape[2], half_crop)

        slices = (slice_x, slice_y, slice_z)
        padding = ((lpad_x, rpad_x), (lpad_y, rpad_y), (lpad_z, rpad_z))

        # Crop possibly incomplete (not crop_size³)
        crop_x = x[slices]

        # Apply padding
        crop_x = np.pad(crop_x, padding, "constant", constant_values=0)

        assert crop_x.shape == (64, 64, 64), f"{crop_x.shape}, {padding}, {slices}"

        # Add crop to the batch
        batch.append(crop_x)
        batch_slices.append(slices)
        batch_padding.append(padding)

        # Batch is full
        if len(batch) == batch_size:
            batch_crop_y = inference(model, batch)
            process_batch(y, batch_crop_y, batch_slices, batch_padding)

            batch = []
            batch_slices = []
            batch_padding = []

    if len(batch) > 0:
        batch_crop_y = inference(model, batch)
        process_batch(y, batch_crop_y, batch_slices, batch_padding)

    # y = tif.imwrite("label.tif", y)


if __name__ == "__main__":
    main()
