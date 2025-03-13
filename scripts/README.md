# CTForams scripts

## Forams detection

`annotate_ctforams.py` : napari script to correct inference errors FP/FN
`exploration.ipynb` : format and create a label matrix from img/csv pair
`format_ctforams_cross_val_single.py` : Create all cross val split from a single pair img/csv
`format_ctforams_from_csv.py`: Create an hdf5 file with the given pair img/csv into the given group (train/val/test)
`merge_annotations.py` : useful to merge an existing csv annotated file with corrected FP/FN errors from a json file
`format_ctforams_cross_val_multi.py` : Create all possible split combination (leave one out)

## SAM - Estimating 3D shape

### Annotations for finetuning

1. First we generate 2D crops around ROIs to be able to annotate using xany-labelling

```sh
python sam/data_to_xany_patches.py --help

Usage: data_to_xany_patches.py [OPTIONS]

Options:
  --im_path TEXT        Path to the image to format  [required]
  --csv_path TEXT       Path to the annotation file in .csv  [required]
  --output_folder TEXT  Path to the output folder  [required]
  --crop_size INTEGER   Crop size of the 2D image crop  [required]
  --axis [x|y|z]        Select the dimension you want to slice  [required]
  --help                Show this message and exit.
```

Example:

```sh
python sam/data_to_xany_patches.py \
--im_path data/crops/im/Aq1T0.tif \
--csv_path data/crops/csv/Aq1T0.csv \
--output_folder data/sam/crops \
--crop_size 64 \
--axis x
```

2. Annotate images (using x-anylabelling or others)

Here is the export configuration file used by x-anylabelling to obtain the segmentation masks:

```
{
    "type": "grayscale",
    "colors": {
        "forams": 255
    }
}
```

3. We format the data folder to fit the davis format taken by sam to perform finetuning directly with the original code.

Input folder structure should look like this:

```
in_folder/
   masks/
      0.png
      1.png
      ...
   0.jpg
   1.jpg
   ...
```

```sh
python sam/annotations_to_davis_format.py --help
Usage: annotations_to_davis_format.py [OPTIONS]
  DAVIS output format
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

Options:
  --in_folder TEXT      The folder containing the annotations.  [required]
  --output_folder TEXT  The folder to output the DAVIS format.  [required]
  --help                Show this message and exit.
```

4. Configure SAM2 training config to use the dataset to finetune

### From 2d point to 3D shape

0. Install [AdaptSAM-3D](https://github.com/KarpRom/adaptSAM-3D/tree/main)

1. `sam/data_to_volumes.py`
   Use this script to generate all 3D segmentations from sam2D with a tif image and csv file containing points location

```sh
python sam/data_to_volumes.py \
--im_path ../data/crops/im/Aq1T0_270z_770z.tif \
--csv_path ../data/crops/csv/v1/Aq1T0_270z_770z.csv \
--model_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
--sam2_checkpoint ../data/sam/models/checkpoint.pt \
--output_folder ../data/sam/vol_output \
--crop_size 64
```

2. `sam/napari_filter_volumes.py`
   This script is a small napari helper script to select or discard 3D shapes estimations to create a curated dataset of 3D shapes. We select the best ones that we will use to train on.

3. `sam/volumes_to_hdf5.py`
   Format the curated volumes and put them into an HDF5 file to be used for training.
   This hdf5 can be used to train a neural network (same as for foraminifera detection).

4. `sam/volumes_inference.py`
   Can process a data (tiff img + csv) to produce a 3D shape inference centered around each annotated center.
