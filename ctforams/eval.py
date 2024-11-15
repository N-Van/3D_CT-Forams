import json
import os
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from ctforams.utils import (
    RankedLogger,
    instantiate_loggers,
    log_hyperparameters,
)

from ctforams.eval.evaluate import infer_and_evaluate_segmentation


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


log = RankedLogger(__name__, rank_zero_only=True)


def evaluate_segmentation(cfg: DictConfig):
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Retrieve the mlflow logger if it exists
    mlflow_logger = None
    for logg in logger:
        if isinstance(logg, MLFlowLogger):
            mlflow_logger = logg

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    cfg.trainer.accelerator = cfg.device
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Test using the loss
    if cfg.lightning_test:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    else:
        # Must load the model by hand if it has not been done before when calling trainer.test()
        state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.net.eval()
        datamodule.data_test.init(compute_windows=False)

    output_dir = cfg.model_dir

    device_name = cfg.device
    net = model.net.to(device_name)

    stats = infer_and_evaluate_segmentation(
        datamodule.data_test,
        net,
        device=device_name,
        crop_size=cfg.crop_size,
        batch_size=cfg.data.batch_size,
        threshold=cfg.threshold,
        iou_threshold=cfg.iou_threshold,
        min_weighted_prob=cfg.min_weighted_prob,
        output_dir=output_dir,
        progress=cfg.progress,
    )

    metric_dict = trainer.callback_metrics
    mlflow_logger.log_metrics(stats)
    return stats, output_dir, metric_dict


def evaluate(cfg) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    stats = {}
    seg_stats, output_dir, _ = evaluate_segmentation(cfg)
    stats["segmentation_results"] = seg_stats

    print("Final results:")
    pprint(stats)

    output_csv = os.path.join(output_dir, "scores.csv")
    print(f"Scores are written to: {output_csv}")
    with open(output_csv, "w") as file:
        json.dump(stats, file, cls=NumpyFloatValuesEncoder)


def load_config(cfg, allow_missing=False):
    hydra_config_path = os.path.join(cfg.model_dir, cfg.hydra_dir, "config.yaml")
    checkpoint_config_path = os.path.join(cfg.model_dir, cfg.ckpt_dir, cfg.ckpt_name)

    if not os.path.exists(hydra_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find hydra config file path: {hydra_config_path}")
    if not os.path.exists(checkpoint_config_path):
        if allow_missing:
            return None
        raise ValueError(f"Could not find checkpoint path: {checkpoint_config_path}")

    train_cfg = OmegaConf.load(hydra_config_path)
    OmegaConf.set_struct(train_cfg, None)
    train_cfg.merge_with(cfg)

    cfg = train_cfg

    OmegaConf.resolve(cfg)

    # Edit checkpoint path
    cfg.ckpt_path = checkpoint_config_path
    return cfg


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Look for hydra config
    cfg = load_config(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
