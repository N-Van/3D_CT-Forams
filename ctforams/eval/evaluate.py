import os
import json
from typing import List, Tuple

import cc3d
import numpy as np
import tifffile as tif
from tqdm import tqdm

from ctforams.eval.inference import segmentation_inference


def infer_and_evaluate_segmentation(
    dataset,
    model,
    device,
    crop_size,
    batch_size,
    threshold=None,
    iou_threshold=0.1,
    min_weighted_prob=0.1,
    output_dir=None,
    progress=False,
):
    y_hat_paths = segmentation_inference(
        dataset=dataset,
        model=model,
        device=device,
        crop_size=crop_size,
        batch_size=batch_size,
        output_dir=output_dir,
        progress=progress,
    )
    y = dataset.y

    return evaluate_segmentation(
        y_hat_paths,
        y,
        dataset.names,
        threshold,
        iou_threshold,
        min_weighted_prob,
        output_dir,
    )


def compute_iou(outputs: np.array, labels: np.array, epsilon=1e-4):
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + epsilon) / (union + epsilon)
    return iou


def iterative_matching(dist_mat: np.ndarray, max_distance: float) -> List[Tuple[int, int]]:
    """
    Iteratively match items in a distance matrix using a threshold value.

    Parameters
    ----------
    dist_mat : np.ndarray
        The distance matrix to be matched.
    max_distance : float
        The maximum distance between two items for them to be considered a match.

    Returns
    -------
    matched_items : List[Tuple[int, int]]
        A list of tuples containing the indices of the matched items in the distance matrix.
    """
    matched_items = []

    cdist_mat = dist_mat.copy()
    max_value = np.max(cdist_mat) + 1

    while np.min(cdist_mat) < max_value:
        # Find the min value
        i, j = np.unravel_index(cdist_mat.argmin(), cdist_mat.shape)

        # If the minimum distance is above the threshold it means
        # that we won't have anymore matches so we stop
        if cdist_mat[i, j] >= max_distance:
            break

        matched_items.append((i, j))

        # "Disable" the true and pred items by setting their distance to
        # the max value + 1
        # This is a trick to avoid counting true and pred elements twice
        cdist_mat[i, :] = max_value
        cdist_mat[:, j] = max_value

    return matched_items


def add_errors(matched_idx, centroids, im_name, error_type):
    errors = []
    for idx in range(len(centroids)):
        if idx not in matched_idx:
            centroid = centroids[idx]
            errors.append(
                {
                    "file": im_name,
                    "x": int(centroid[0]),
                    "y": int(centroid[1]),
                    "z": int(centroid[2]),
                    "corrected": -1,
                    "type": error_type,
                }
            )
    return errors


def get_fp_fn(matched_items, truth_stats, pred_stats, im_name):
    matched_true = []
    matched_pred = []

    for true_id, pred_id in matched_items:
        matched_true.append(true_id)
        matched_pred.append(pred_id)

    data_errors = []
    data_errors += add_errors(matched_true, truth_stats["centroids"][1:], im_name, "FN")
    data_errors += add_errors(matched_pred, pred_stats["centroids"][1:], im_name, "FP")
    return data_errors


def evaluate_segmentation(
    y_hat_paths, y, names, threshold, iou_threshold, min_weighted_pro, output_dir
):
    print(f"Evaluation running with threshold = {threshold}")
    # For each volume
    uint_8_th = int(threshold * 255)

    TP, FP, FN = 0, 0, 0

    data_errors = []

    for i in tqdm(range(len(y)), desc="Evaluating matrices..."):
        im_name = names[i]
        pred_path = y_hat_paths[im_name]
        y_hat = tif.memmap(pred_path)

        print(f"Processing file {im_name}")
        # DEBUG
        if im_name != "Aq3T1_13":
            continue

        # Threshold both
        pred = (y_hat > uint_8_th).astype(np.uint8)
        truth = (y[i][:] > 0).astype(np.uint8)

        # Binary [0;1] to [0;255] uint8
        pred *= 255
        truth *= 255

        # Get all 3D ccs from both volume y and y_hat
        pred_ccs = cc3d.connected_components(pred)
        truth_ccs = cc3d.connected_components(truth)

        # keys are: voxel_counts, bounding_boxes, centroids
        pred_stats = cc3d.statistics(pred_ccs)
        truth_stats = cc3d.statistics(truth_ccs)

        # Skip first (bg)
        true_bboxes = truth_stats["bounding_boxes"][1:]
        pred_bboxes = pred_stats["bounding_boxes"][1:]

        n_true = len(true_bboxes)
        n_pred = len(pred_bboxes)

        print(f"True ccs : {n_true} vs Pred ccs : {n_pred}")

        tp, fn, fp = 0, 0, 0

        if n_true == 0:
            fp = n_pred
        elif n_pred == 0:
            fn = n_true
        else:
            iou_matrix = np.zeros((len(true_bboxes), len(pred_bboxes)), np.float32)

            for i, true_bbox in tqdm(
                enumerate(true_bboxes),
                desc="Comparing true bboxes...",
                leave=False,
                total=len(true_bboxes),
            ):
                true_rect = BBox(true_bbox)
                for j, pred_bbox in enumerate(pred_bboxes):
                    pred_rect = BBox(pred_bbox)

                    if true_rect.intersects(pred_rect):

                        # Merge both rectangle to get area of interest for iou optimization
                        roi = BBox.merge_as_slices(true_rect, pred_rect)

                        # Get both pred and true roi
                        pred_roi = pred[roi]
                        truth_roi = truth[roi]

                        # Compute iou
                        iou = compute_iou(pred_roi, truth_roi)
                        # We use inverse iou since the matching algorithm
                        # match items that are the closest
                        # ie 0.1 is close while 1.0 is far
                        iou_matrix[i, j] = 1.0 - iou

            # List of pairs (i,j) i=true j=pred
            matched_items = iterative_matching(iou_matrix, 1.0 - iou_threshold)
            tp = len(matched_items)
            fp = iou_matrix.shape[1] - tp
            fn = iou_matrix.shape[0] - tp

            data_errors += get_fp_fn(matched_items, truth_stats, pred_stats, im_name)

        TP += tp
        FP += fp
        FN += fn

        print(f"TP: {TP} ; FP: {FP} ; FN: {FN}")

    errors_path = os.path.join(output_dir, "errors.json")
    with open(errors_path, "w") as fp:
        json.dump(data_errors, fp)

    metrics = {"tp": TP, "fp": FP, "fn": FN}
    return metrics


class Segment(object):

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def intersects(self, other):
        return not (self.start > other.end or other.start > self.end)


class BBox(object):

    def __init__(self, slices):
        self.x_seg = Segment(slices[0].start, slices[0].stop)
        self.y_seg = Segment(slices[1].start, slices[1].stop)
        self.z_seg = Segment(slices[2].start, slices[2].stop)

    def intersects(self, other):
        return (
            self.x_seg.intersects(other.x_seg)
            and self.y_seg.intersects(other.y_seg)
            and self.z_seg.intersects(other.z_seg)
        )

    def merge_as_slices(a, b):
        x_start = min(a.x_seg.start, b.x_seg.start)
        x_end = min(a.x_seg.end, b.x_seg.end)
        y_start = min(a.y_seg.start, b.y_seg.start)
        y_end = min(a.y_seg.end, b.y_seg.end)
        z_start = min(a.z_seg.start, b.z_seg.start)
        z_end = min(a.z_seg.end, b.z_seg.end)
        return slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end)

    def __str__(self):
        return f"x=[{self.x_seg.start};{self.x_seg.end}] y=[{self.y_seg.start};{self.y_seg.end}] z=[{self.z_seg.start};{self.z_seg.end}]"
