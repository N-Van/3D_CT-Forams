import os
from glob import glob
from pathlib import Path

import click
import cv2


@click.command()
@click.option(
    "--in_folder",
    required=True,
    help="The folder containing the annotations.",
)
@click.option(
    "--output_folder",
    required=True,
    help="The folder to output the DAVIS format.",
)
def main(in_folder, output_folder):
    """
    DAVIS format output

    /DAVIS

        /JPEGImages

            /<img or video name>

                /00001.jpg <- color image [0; 255]^3 any size

                /00002.jpg

                /...

        /Annotations

            /<img or video name>

                /00001.png <- binary mask [0; 255] same as img

                /00002.png
    """

    jpg_folder = os.path.join(output_folder, "JPEGImages")
    annotations_folder = os.path.join(output_folder, "Annotations")

    os.makedirs(jpg_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # List all crops images
    img_files = glob(os.path.join(in_folder, "*.jpg"))
    for img_file in img_files:
        img_stem = Path(img_file).stem
        # We should find a mask in in_folder/masks/
        mask_file = os.path.join(in_folder, "masks", f"{img_stem}.png")

        # Read img as color
        im = cv2.imread(img_file, cv2.IMREAD_COLOR)
        # Read mask as greyscale
        if not os.path.exists(mask_file):
            print(f"Warning: mask does not exists for file {img_stem}")
            continue

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        local_im_folder = os.path.join(jpg_folder, img_stem)
        local_mask_folder = os.path.join(annotations_folder, img_stem)

        os.makedirs(local_im_folder, exist_ok=True)
        os.makedirs(local_mask_folder, exist_ok=True)

        # Write img
        cv2.imwrite(os.path.join(local_im_folder, "00001.jpg"), im)
        # Write mask
        cv2.imwrite(os.path.join(local_mask_folder, "00001.png"), mask)


if __name__ == "__main__":
    main()
