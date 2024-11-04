import onnxruntime as ort
import click
import numpy as np
from tqdm import tqdm
import time


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to the onnx model.",
)
@click.option(
    "--crop_size",
    required=False,
    default=64,
    help="Expected size of the input.",
)
@click.option(
    "--batch_size",
    required=False,
    default=32,
    help="Inference batch size.",
)
def main(model_path, crop_size, batch_size) -> None:
    model = load_model(model_path)

    image = np.zeros((batch_size, 1, crop_size, crop_size, crop_size), dtype=np.float32)

    n_iter = 1000
    n_batch = n_iter // batch_size
    mean_elapsed = run_n_inference(model, image, n_batch)

    # Batch time in seconds to iteration time in ms
    mean_elapsed_iter = 1000 * mean_elapsed / batch_size

    print(f"Doing n_iter took an average of {mean_elapsed_iter} ms")
    print(f"FPS: {(1000 / mean_elapsed_iter):.3f}")


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


if __name__ == "__main__":
    main()
