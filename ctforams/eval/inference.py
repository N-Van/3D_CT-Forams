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
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    crop_size = (crop_size, crop_size, crop_size)

    device_name = device if isinstance(device, str) else device.type
    use_cuda = device_name != "cpu"

    log.info(f"Using {'GPU' if use_cuda else 'CPU'}")
    if output_dir is not None:
        log.info(f"Prediction will be stored in folder: {output_dir}")

    # Make sure model is in eval mode
    model.eval()

    predictions = []
    # Loop over volumes
    for i in tqdm(range(len(dataset.x)), desc="Running segmentation inference..."):
        # X,Y,Z
        x = dataset.x[i]

        x = torch.tensor(x, device="cpu")
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
                device="cpu",  # hold the full tensor on cpu
                progress=True,
            )

        # FP32 to uint8 to limit memory usage
        y = (y * 255).astype(np.uint8)

        if output_dir is not None:
            tif.imsave(os.path.join(output_dir, f"{dataset.names[i]}.tif"), y, check_contrast=False)

        predictions.append(y)
    return predictions
