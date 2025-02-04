import os
import shutil
from glob import glob
from pathlib import Path

import click


@click.command()
@click.option(
    "--crops_dir",
    required=True,
    help="Directory where the annotated crops are",
)
@click.option(
    "--output_dir",
    required=True,
    help="Output directory",
)
def main(crops_dir, output_dir):
    crops = glob(os.path.join(crops_dir, "*.jpg"))
    masks_dir = os.path.join(crops_dir, "masks")

    im_dir = os.path.join(output_dir, "im")
    label_dir = os.path.join(output_dir, "label")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(im_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for crop_path in crops:
        stem = Path(crop_path).stem
        masks_path = os.path.join(masks_dir, f"{stem}.png")
        if os.path.exists(masks_path):
            shutil.copyfile(crop_path, os.path.join(im_dir, f"{stem}.png"))
            shutil.copyfile(masks_path, os.path.join(label_dir, f"{stem}.png"))


if __name__ == "__main__":
    main()
