import os

import h5py
import numpy as np
from ctforams.utils import RankedLogger
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval
from torch.utils.data import Dataset
from tqdm import tqdm

log = RankedLogger(__name__, rank_zero_only=True)


class CTForamsDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        group_name,
        crop_size,
        max_steps_per_epoch,
        training=False,
        window_overlap=0.5,
        discard_volume_ratio_th=0.5,
        sphere_radius=8,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.group_name = group_name
        self.crop_size = crop_size
        self.training = training
        self.aug = None
        self.max_steps_per_epoch = max_steps_per_epoch
        self.window_overlap = window_overlap
        self.theorical_volume = 4 / 3 * np.pi * sphere_radius**3
        self.discard_volume_ratio_th = discard_volume_ratio_th

    def init(self, compute_windows=True):
        log.info("Loading data pairs...")
        # Load pairs of image and labels as mem map
        self.x, self.y, self.names = self.load_hdf5()

        if compute_windows:
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
        self.x, self.y, self.names = self.load_hdf5()

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

        discarded_count = 0
        for i, window in tqdm(enumerate(self.windows)):
            x_i, window_slice = window.values()
            y = self.y[x_i][window_slice]
            if np.any(y):
                annotation_volume = np.count_nonzero(y)
                # We compute how much of the sphere volume is present in the crop
                # If we don't have enough, this is not a good sample since it will be on the border so we discard it
                # If there is more than one annotation this doesn't work but usually there is only one object in a
                # single window
                if float(annotation_volume / self.theorical_volume) >= self.discard_volume_ratio_th:
                    positive_indexes.append(i)
                else:
                    discarded_count += 1
            else:
                negative_indexes.append(i)
        log.info(
            f"Discarded #{discarded_count} positive windows that were having unsufficient annotation volume"
        )
        log.info(
            f"Found #{len(positive_indexes)} positive windows and #{len(negative_indexes)} negative windows"
        )
        return positive_indexes, negative_indexes

    def load_hdf5(self):
        x = []
        y = []
        names = []

        assert os.path.exists(self.hdf5_path), f"Path {self.hdf5_path} does not exist."

        self.hf = h5py.File(self.hdf5_path, "r")
        ds_group = self.hf[self.group_name]
        images = ds_group["im"]
        labels = ds_group["label"]
        for image_name in images.keys():
            im = images[image_name]
            label = labels[image_name]

            assert (
                len(im.shape) == 3
            ), f"Expected 3D grayscale image matrix to be 3D (W,H,D) but found {im.shape}"
            assert (
                len(label.shape) == 3
            ), f"Expected 3D binary label matrix to be 3D (W,H,D) but found {label.shape}"

            if (
                im.shape[0] < self.crop_size
                or im.shape[1] < self.crop_size
                or im.shape[2] < self.crop_size
            ):
                print(
                    f"Must pad array since it is too small with crop size={self.crop_size}: {im.shape}"
                )

            x.append(im)
            y.append(label)
            names.append(image_name)
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
        windows_indices = self.compute_window_indices(
            x.shape, size=self.crop_size, overlap=self.window_overlap
        )
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
        # M,X,Y,Z with C the class channel
        y = self.y[x_i][window_slice]

        # Normalize x data range
        x = self.norm_patch(x)

        # Add channel dim
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        if y.dtype != np.float32:
            y = y.astype(np.float32)
            if y.max() > 1:
                y /= 255.0

        assert y.min() >= 0 and y.max() <= 1.0
        assert y.dtype == np.float32, f"Expected float32 but found : {y.dtype}"

        # Augment sample if train data
        if self.aug is not None:
            augmented = self.aug({"image": x, "label": y})
            x, y = augmented["image"], augmented["label"]

        # Compute loss on all pixels
        weights = np.ones_like(y)

        # Contains object inside
        # if np.count_nonzero(y) > 0:
        #     # compute loss on high confidence pixels only
        #     weights[y < 0.5] = 0.0

        return x, y, weights

    def __len__(self):
        if len(self.negative_window_index) == 0:
            return len(self.positive_window_index)
        if len(self.positive_window_index) == 0:
            return len(self.negative_window_index)
        if self.training:
            return min(len(self.positive_window_index), len(self.negative_window_index)) * 2
            # return min(len(self.windows), self.max_steps_per_epoch)
        return len(self.windows)
