import os
import hydra
from hydra import initialize, compose
import onnx
import torch
from omegaconf import DictConfig, OmegaConf
import click
from onnxsim import simplify


os.environ["PROJECT_ROOT"] = "../"


@click.command()
@click.option(
    "--xp_dir",
    required=True,
    help="Path to experiment log folder.",
)
def main(xp_dir) -> None:
    with initialize(
        version_base="1.3",
        config_path=os.path.join("..", "..", xp_dir, ".hydra"),
        job_name="notebook_visualization",
    ):
        cfg = compose(config_name="config.yaml")

        ckpt_path = os.path.join(xp_dir, "checkpoints", "best.ckpt")

        # Create and load model
        model = hydra.utils.instantiate(cfg.model)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        net = model.net
        net.eval()
        crop_size = cfg.crop_size

        convert(net, crop_size, xp_dir)


def convert(model, crop_size, xp_dir):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn((1, 1, crop_size, crop_size, crop_size), requires_grad=True)

    export_path = os.path.join(xp_dir, "model.onnx")
    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        export_path,  # where to save the model
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

    onnx_model = onnx.load(export_path)
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, export_path)
    print(f"Model has been converted to ONNX: path={export_path}")


if __name__ == "__main__":
    main()
