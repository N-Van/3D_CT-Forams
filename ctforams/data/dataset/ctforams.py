import json
import os
from pathlib import Path

import numpy as np
import tifffile
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval
from torch.utils.data import Dataset
from tqdm import tqdm

from ctforams.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CTForamsDataset(Dataset):
    def __init__(self, data_root, paths, crop_size, max_steps_per_epoch, training=False):
        super().__init__()
        self.data_root = data_root
        self.paths = paths
        self.crop_size = crop_size
        self.training = training
        self.aug = None
        self.max_steps_per_epoch = max_steps_per_epoch

    def init(self):
        log.info("Loading data pairs...")
        # Load pairs of image and labels as mem map
        self.x, self.y, self.names = self.load_data_pairs()

        log.info("Compute all possible windows from data...")
        # Compute all possible windows
        self.windows = self.compute_sequences_index()

        log.info("Splitting data into positive/negative samples")
        # Create two lists: track positive/negative samples
        self.positive_window_index, self.negative_window_index = self.separate_pos_neg_windows()

        # Done
        log.info("Done")

    def get_total_num_windows(self):
        log.info("Loading data pairs...")
        # Load pairs of image and labels as mem map
        self.x, self.y, self.names = self.load_data_pairs()

        log.info("Compute all possible windows from data...")
        # Compute all possible windows
        windows = self.compute_sequences_index()
        self.x = None
        self.y = None
        self.names = None
        return min(len(windows), self.max_steps_per_epoch)

    def separate_pos_neg_windows(self):
        positive_indexes = []
        negative_indexes = []

        for i, window in tqdm(enumerate(self.windows)):
            x_i, window_slice = window.values()
            y = self.y[x_i][0][window_slice]
            if np.any(y):
                positive_indexes.append(i)
            else:
                negative_indexes.append(i)
        return positive_indexes, negative_indexes

    def load_data_pairs(self):
        x = []
        y = []
        names = []

        with open(self.paths, "r") as paths_file:
            json_paths = json.load(paths_file)
        for paths in json_paths:
            im_path = os.path.join(self.data_root, paths["image"])
            label_path = os.path.join(self.data_root, paths["label"])
            im = tifffile.memmap(im_path, mode="r")
            label = tifffile.memmap(label_path, mode="r")
            name = Path(im_path).stem

            if len(im.shape) == 4 and im.shape[0] == 1:
                im = im[0]

            x.append(im)
            y.append(label)
            names.append(name)
        return x, y, names

    def compute_sequences_index(self):
        windows = []
        for i, x_ in enumerate(self.x):
            windows += self.add_crop_windows(x_, index=i)
        return windows

    def compute_window_indices(self, frame_size, size=None, overlap=0.0):
        size = [size, size, size]
        scan_interval = _get_scan_interval(frame_size, size, 3, [overlap, overlap, overlap])
        slices = dense_patch_slices(frame_size, size, scan_interval, return_slice=True)
        return slices

    def add_crop_windows(self, x, index):
        windows = []
        windows_indices = self.compute_window_indices(x.shape, size=self.crop_size, overlap=0)
        for slices in windows_indices:
            windows.append(
                {
                    "movie_index": index,
                    "window_slice": slices,
                }
            )
        return windows

    def norm_patch(self, x):
        x = x.astype(np.float32) / 255.0
        return x

    def __getitem__(self, i):
        # Draw random positive/negative patch evenly (1 out of 2)
        if self.training:
            set_ = self.positive_window_index
            # If no positive sample or one out of 2 sample
            if len(set_) == 0 or (i % 2 == 1 and len(self.negative_window_index) > 0):
                set_ = self.negative_window_index
            i = np.random.choice(set_)

        # Get the image index and window slice
        window = self.windows[i]
        x_i, window_slice = window.values()

        # Extract window
        # M,X,Y,Z with M the tiff image index
        x = self.x[x_i][window_slice]
        # M,C,X,Y,Z with C the class channel
        y = self.y[x_i][0][window_slice]

        # Normalize x data range
        x = self.norm_patch(x)

        # Add channel dim
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # Augment sample if train data
        if self.aug is not None:
            augmented = self.aug({"image": x, "label": y})
            x, y = augmented["image"], augmented["label"]

        return x, y

    def __len__(self):
        if self.training:
            return min(len(self.windows), self.max_steps_per_epoch)
        return len(self.windows)
