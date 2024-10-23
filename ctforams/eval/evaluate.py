from ctforams.eval.inference import segmentation_inference
import cc3d
import numpy as np
from scipy.spatial import KDTree


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
):
    y_hat = segmentation_inference(
        dataset=dataset,
        model=model,
        device=device,
        crop_size=crop_size,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    y = dataset.y

    return evaluate_segmentation(
        y_hat,
        y,
        dataset.names,
        threshold,
        iou_threshold,
        min_weighted_prob,
        output_dir,
    )


def evaluate_segmentation(y_hat, y, names, threshold, iou_threshold, min_weighted_pro, output_dir):
    # For each volume
    for i in range(len(y)):
        # Threshold both
        pred = (y_hat[i] > threshold).astype(np.uint8)
        truth = (y[i] > 0).astype(np.uint8)

        # Get all 3D ccs from both volume y and y_hat
        pred_ccs = cc3d.connected_components(pred)
        truth_ccs = cc3d.connected_components(truth)

        pred_stats = cc3d.statistics(pred_ccs)
        truth_stats = cc3d.statistics(truth_ccs)

        # Build a KD tree
        true_bboxes = []
        for true_stat in truth_stats:
            for pred_stat in pred_stats:
                print(pred_stat)

        # Check for all intersections above iou threshold on bounding boxes
        # If it passes, then check real iou
