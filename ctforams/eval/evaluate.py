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
        print(y[i])
        truth = (y[i][:] > 0).astype(np.uint8)

        print(pred.shape, truth.shape)

        # Get all 3D ccs from both volume y and y_hat
        pred_ccs = cc3d.connected_components(pred)
        truth_ccs = cc3d.connected_components(truth)

        # keys are: voxel_counts, bounding_boxes, centroids
        pred_stats = cc3d.statistics(pred_ccs)
        truth_stats = cc3d.statistics(truth_ccs)

        true_bboxes = []
        #
        for true_bbox in truth_stats["bounding_boxes"]:
            true_rect = BBox(true_bbox)
            for pred_bbox in pred_stats["bounding_boxes"]:
                pred_rect = BBox(pred_bbox)

                if true_rect.intersects(pred_rect):
                    print("intersection")
                    print(true_rect)
                    print(pred_rect)

        # Check for all intersections above iou threshold on bounding boxes
        # If it passes, then check real iou


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
