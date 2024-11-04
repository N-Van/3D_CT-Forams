import os

import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch_3d as sm
import torch
from tqdm import tqdm
import time
from functools import partial

backbones = [
    "resnet18",
    "efficientnet-b0",
    # "resnext50_32x4d",
    # "densenet121",
    "dpn68",
    "vgg11",
    "mobileone_s0",
]

models = [
    sm.Unet,
    # sm.UnetPlusPlus,
    # sm.MAnet,
    # sm.Linknet,
    sm.FPN,
    # sm.DeepLabV3,
    sm.DeepLabV3Plus,
]


def main():
    crop_size = 64
    batch_size = 16

    best_fps = 0
    best_combo = {}

    for model in models:
        for backbone in backbones:
            print()
            print(f"Testing backbone {backbone} with model {model}")
            # Instanciate
            net = model(encoder_name=backbone, in_channels=1, classes=1, encoder_weights=None)

            # Convert to onnx format
            ort_path = convert(net, crop_size, "/tmp")

            fps = run_test(ort_path, crop_size, batch_size)

            if fps > best_fps:
                best_fps = fps
                best_combo = {"model": model, "backbone": backbone}
    print(f"Best model: {best_combo} with fps={best_fps}")


def run_test(model_path, crop_size, batch_size):
    model = load_model(model_path)
    image = np.zeros((batch_size, 1, crop_size, crop_size, crop_size), dtype=np.float32)

    n_iter = 100
    n_batch = n_iter // batch_size
    mean_elapsed = run_n_inference(model, image, n_batch)

    # Batch time in seconds to iteration time in ms
    mean_elapsed_iter = 1000 * mean_elapsed / batch_size
    fps = 1000 / mean_elapsed_iter
    print(f"Doing {n_iter} iterations took an average of {mean_elapsed_iter} ms per iter")
    print(f"FPS: {fps:.3f}")

    return fps


def load_model(model_path):
    print(f"Using device : {ort.get_device()}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        sess_options=sess_options,
    )
    return ort_sess


def run_n_inference(model, image, n_iter):
    elapsed = []
    for n in tqdm(range(n_iter)):
        start_time = time.time()
        outputs = model.run(None, {"modelInput": image})
        end_time = time.time()
        elapsed.append(end_time - start_time)
    elapsed = np.asarray(elapsed)
    return np.mean(elapsed)


def convert(model, crop_size, dir):
    dummy_input = torch.randn((1, 1, crop_size, crop_size, crop_size), requires_grad=True)
    model_path = os.path.join(dir, "model.onnx")

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        # do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {0: "batch_size"},  # variable length axes
            "modelOutput": {0: "batch_size"},
        },
    )
    return model_path


if __name__ == "__main__":
    main()
