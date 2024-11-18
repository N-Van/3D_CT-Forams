""""""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryJaccardIndex


class CTForamsLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        criterion=None,
        compile=False,
        scheduler_interval: str = "step",
        criterion_on_center=False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net
        self.criterion_on_center = criterion_on_center

        # loss function
        self.criterion = criterion
        self.optimizer = self.hparams.optimizer(params=self.net.parameters())

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryJaccardIndex(threshold=0.5)
        self.val_acc = BinaryJaccardIndex(threshold=0.5)
        self.test_acc = BinaryJaccardIndex(threshold=0.5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def reset_metrics(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, weights = batch

        outputs = self.forward(x)

        if self.criterion_on_center:
            crop_size = outputs.shape[-1]
            center = crop_size // 2
            quarter_size = crop_size // 4
            cslice = slice(center - quarter_size, center + quarter_size)
            outputs = outputs[:, :, cslice, cslice, cslice]
            y = y[:, :, cslice, cslice, cslice]

        loss = self.criterion(outputs, y)

        # loss = loss * weights
        # loss = loss.sum() / (weights.sum() + 1e-8)

        return loss, outputs, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.safe_update_metric(self.train_loss, loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        binary_preds = torch.nn.functional.sigmoid(preds)
        binary_targets = targets > 0.5
        self.safe_update_metric(self.train_acc, binary_preds.flatten(), binary_targets.flatten())
        self.log("train/iou", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.scheduler is not None:
            self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def safe_update_metric(self, metric, value, *args):
        if not torch.isnan(value.sum()):
            metric.update(value, *args)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.safe_update_metric(self.val_loss, loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        binary_preds = torch.nn.functional.sigmoid(preds)
        binary_targets = targets > 0.5
        self.safe_update_metric(self.val_acc, binary_preds.flatten(), binary_targets.flatten())
        self.log("val/iou", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best.update(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/iou_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            self.scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
