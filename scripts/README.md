# CTForams scripts

## Forams detection

`annotate_ctforams.py` : napari script to correct inference errors FP/FN
`exploration.ipynb` : format and create a label matrix from img/csv pair
`format_ctforams_cross_val_single.py` : Create all cross val split from a single pair img/csv
`format_ctforams_from_csv.py`: Create an hdf5 file with the given pair img/csv into the given group (train/val/test)
`merge_annotations.py` : useful to merge an existing csv annotated file with corrected FP/FN errors from a json file
`format_ctforams_cross_val_multi.py` : Create all possible split combination (leave one out)

## SAM - Estimating 3D shape

1. `sam/data_to_xany_patches.py`
   First we generate 2D crops around ROIs to be able to annotate using xany-labelling

2. `sam/xany_crop_to_sam_data.py`
   Then we extract annotated patches (some can be missing) from the patches folder and build
   pairs of im/label 2D crops to train sam on it

3. `sam/data_to_volumes.py`
   Use this script to generate all possible sequence of 2D images for the 3 different planes.
   It will be used by the sam tracking inference script to obtain 3D shapes estimations

4. `TBD`
   Uses the image sequences generated in 3. to perform the tracking and provide 3D shapes estimations.

5. `sam/napari_filter_volumes.py`
   This script is a small napari helper script to select or discard 3D shapes estimations to create a curated dataset of 3D shapes. We select the best ones that we will use to train on.

6. `sam/volumes_to_hdf5.py`
   Format the curated volumes and put them into an HDF5 file to be used for training.

7. `sam/volumes_inference.py`
   Can process a data (tiff img + csv) to produce a 3D shape inference centered around each annotated center.
