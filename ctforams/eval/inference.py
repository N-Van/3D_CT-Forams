import os
import torch
from tqdm import tqdm
from ctforams.utils import RankedLogger
import monai
import numpy as np
import tifffile as tif

log = RankedLogger(__name__, rank_zero_only=True)


def segmentation_inference(
    dataset, model, device, crop_size, batch_size, overlap=0.25, output_dir=None, progress=False
):
    # torch.backends.cudnn.benchmark = True

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    crop_size = (crop_size, crop_size, crop_size)

    device_name = device if isinstance(device, str) else device.type
    use_cuda = device_name != "cpu"

    log.info(f"Using {'GPU' if use_cuda else 'CPU'}")
    if output_dir is not None:
        log.info(f"Prediction will be stored in folder: {output_dir}")

    print(f"Using parameters:")
    print(f"batch_size: {batch_size}")
    print(f"crop_size: {crop_size}")
    print(f"overlap: {overlap}")
    print(f"sliding window device: {device}")

    # Make sure model is in eval mode
    model.eval()

    predictions_paths = {}

    z_size = crop_size[0]
    z_step = z_size // 2

    assert output_dir is not None, "You must set the output dir"

    # Loop over volumes
    for i in tqdm(range(len(dataset.x)), desc="Running segmentation inference..."):
        # X,Y,Z memmap
        x = dataset.x[i]

        print(f"Input shape: {x.shape}")
        output_path = os.path.join(output_dir, f"{dataset.names[i]}.tif")
        overlap_path = os.path.join(output_dir, f"{dataset.names[i]}_overlap.tif")
        print(f"Running inference on {dataset.names[i]}")
        predictions_paths[dataset.names[i]] = output_path

        # DEBUG ??
        if os.path.exists(output_path):
            continue

        # Create a memmap of the same shape directly on disk to avoid using too much memory
        y = tif.memmap(output_path, shape=x.shape, dtype=np.float16)

        # Iterate over Z
        n_over_z = int(np.ceil(x.shape[-1] / z_step))

        counts = tif.memmap(overlap_path, shape=x.shape, dtype=np.float16)

        for k in range(n_over_z):
            z_start = min(k * z_step, x.shape[-1] - z_size)
            z_end = z_start + z_size
            sub_arr = counts[..., z_start:z_end]
            gaussian = get_gaussian(sub_arr.shape)
            counts[..., z_start:z_end] += gaussian

        for k in tqdm(range(n_over_z), desc="Iterating over Z slices", total=n_over_z, leave=False):
            z_start = min(k * z_step, x.shape[-1] - z_size)
            z_end = z_start + z_size
            sub_x = x[..., z_start:z_end]

            # Infer
            sub_y = infer_part(
                sub_x, dataset, crop_size, batch_size, overlap, model, device, progress
            )

            # Write data to memmap
            # Compute the mean (but should be weighted by a gaussian)
            gaussian = get_gaussian(sub_y.shape)

            # a*(1-w) + b*w => weighted sum with w in [0;1.0]
            wsub_y = (sub_y * gaussian) / counts[..., z_start:z_end]

            y[..., z_start:z_end] += wsub_y

        y = (y * 255).astype(np.uint8)

    return predictions_paths


def get_gaussian(size, dim=2, s=0.125):
    s = s * size[dim]
    x = np.arange(start=-(size[dim] - 1) / 2.0, stop=(size[dim] - 1) / 2.0 + 1)
    x = np.exp(x**2 / (-2 * s**2))  # 1D gaussian
    min_non_zero = max(np.min(x), 1e-3)
    x = np.clip(x, a_min=min_non_zero, a_max=np.max(x))

    z = np.ones(size)
    z[:, :] = x

    return z


def infer_part(x, dataset, crop_size, batch_size, overlap, model, device, progress):
    x = dataset.norm_patch(x)

    x = torch.tensor(x, device="cpu")
    # Add channel dim
    x = torch.unsqueeze(x, dim=0)
    # Add batch dim
    x = torch.unsqueeze(x, dim=0)

    # Perform inference on current sequence
    with torch.no_grad():
        y = monai.inferers.sliding_window_inference(
            inputs=x,
            roi_size=crop_size,  # window crop size
            sw_batch_size=batch_size,  # window batch size
            overlap=overlap,  # qty overlap
            predictor=model,
            mode="gaussian",  # weighted sum
            sw_device=device,  # perform inference on device
            device=device,  # hold the full tensor on cpu
            progress=progress,
        )

    y = torch.nn.functional.sigmoid(y)
    y = y.detach().cpu().numpy()
    # FP32 to uint8 to limit memory usagey[i
    # y = (y * 255).astype(np.uint8)

    # Remove batch dim
    y = np.squeeze(y, axis=0)
    # Remove channel dim
    y = np.squeeze(y, axis=0)
    return y
