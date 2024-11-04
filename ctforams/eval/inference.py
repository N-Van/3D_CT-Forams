import os
import torch
from tqdm import tqdm
from ctforams.utils import RankedLogger
import monai
import numpy as np
import tifffile as tif

log = RankedLogger(__name__, rank_zero_only=True)


def segmentation_inference(
    dataset, model, device, crop_size, batch_size, overlap=0.5, output_dir=None
):
    torch.backends.cudnn.benchmark = True

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

    predictions = []

    z_size = crop_size[0]
    z_step = z_size // 2

    # Loop over volumes
    for i in tqdm(range(len(dataset.x)), desc="Running segmentation inference..."):
        # X,Y,Z memmap
        x = dataset.x[i]

        y = np.zeros_like(x, dtype=np.uint8)

        # Iterate over Z
        n_over_z = int(np.ceil(x.shape[-1] / z_step))
        for k in tqdm(range(n_over_z), desc="Iterating over Z slices", total=n_over_z, leave=False):
            z_start = min(k * z_step, x.shape[-1] - z_size)
            z_end = z_start + z_size
            sub_x = x[..., z_start:z_end]
            sub_y = infer_part(sub_x, dataset, crop_size, batch_size, overlap, model, device)
            y[..., z_start:z_end] = sub_y

            # DEBUG
            break

        if output_dir is not None:
            tif.imwrite(
                os.path.join(output_dir, f"{dataset.names[i]}.tif"),
                y,
                bigtiff=True,
                compression="zlib",
            )

        predictions.append(y)
    return predictions


def infer_part(x, dataset, crop_size, batch_size, overlap, model, device):
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
            progress=True,
        )

    y = torch.nn.functional.sigmoid(y)
    y = y.detach().cpu().numpy()
    # FP32 to uint8 to limit memory usage
    y = (y * 255).astype(np.uint8)

    # Remove batch dim
    y = np.squeeze(y, axis=0)
    # Remove channel dim
    y = np.squeeze(y, axis=0)
    return y
