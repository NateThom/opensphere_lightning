from typing import Any, Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class Backbone_Head_Module(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_classes,
        backbone: torch.nn.Module,
        # optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        optimizer_backbone: torch.optim.Optimizer,
        scheduler_backbone: torch.optim.lr_scheduler,
        head: torch.nn.Module,
        optimizer_head: torch.optim.Optimizer,
        scheduler_head: torch.optim.lr_scheduler,
        clip_grad_norm: float,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.head = head

    def evaluate(self, dataset, feats, FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']):
        # pair-wise scores
        feats = F.normalize(feats, dim=1)
        feats0 = feats[dataset.indices0, :]
        feats1 = feats[dataset.indices0, :]
        scores = torch.sum(feats0 * feats1, dim=1).tolist()
    
        # eer and auc
        fpr, tpr, _ = metrics.roc_curve(dataset.labels, scores, pos_label=1)
        roc_curve = interp1d(fpr, tpr)
        EER = 100. * brentq(lambda x : 1. - x - roc_curve(x), 0., 1.)
        AUC = 100. * metrics.auc(fpr, tpr)

        # get acc
        tnr = 1. - fpr
        pos_num = dataset.labels.count(1)
        neg_num = dataset.labels.count(0)
        ACC = 100. * max(tpr * pos_num + tnr * neg_num) / len(dataset.labels)

        # TPR @ FPR
        if isinstance(FPRs, list):
            TPRs = [
                ('TPR@FPR={}'.format(FPR), 100. * roc_curve(float(FPR)))
                for FPR in FPRs
            ]
        else:
            TPRs = []

        return [('ACC', ACC), ('EER', EER), ('AUC', AUC)] + TPRs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.backbone(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        opt1.zero_grad()
        opt2.zero_grad()

        x, y = batch
        logits = self.forward(x)
        loss = self.head(logits, y)

        self.manual_backward(loss)

        b_grad = clip_grad_norm_(
                self.backbone.parameters(),
                max_norm=self.hparams.clip_grad_norm, norm_type=2)
        h_grad = clip_grad_norm_(
                self.head.parameters(),
                max_norm=self.hparams.clip_grad_norm, norm_type=2)

        opt1.step()
        sch1.step()
        
        opt2.step()
        sch2.step()

        # update and log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bkb_grad", b_grad, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/head_grad", h_grad, on_step=True, on_epoch=False, prog_bar=True)

        # return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_epoch_start(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_feats_aggregate = torch.zeros([len(self.trainer.datamodule.data_val), self.backbone.fc.out_features], dtype=torch.float32).to(self.device)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, indices = batch
        logits = self.forward(x)
        x = torch.flip(x, [3])
        logits += self.forward(x)
        logits = logits.float()

        self.val_feats_aggregate[indices, :] = logits

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_feats_aggregate = self.all_gather(self.val_feats_aggregate)
        results = self.evaluate(self.trainer.datamodule.data_val, self.val_feats_aggregate.cpu())

        self.log("val/acc", results[0][1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/eer", results[1][1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", results[2][1], on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.backbone = torch.compile(self.backbone)
            self.head = torch.compile(self.head)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_backbone = self.hparams.optimizer_backbone(params=self.backbone.parameters())
        optimizer_head = self.hparams.optimizer_head(params=self.head.parameters())
        if self.hparams.scheduler_backbone is not None:
            scheduler_backbone = self.hparams.scheduler_backbone(optimizer=optimizer_backbone)
            scheduler_head = self.hparams.scheduler_head(optimizer=optimizer_head)

            return [optimizer_backbone, optimizer_head], [{"scheduler": scheduler_backbone}, {"scheduler": scheduler_head}]
        return [optimizer_backbone, optimizer_head]

        # optimizer = self.hparams.optimizer(params=self.parameters())
        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)

        #     return [optimizer], [{"scheduler": scheduler}]
        # return [optimizer]

if __name__ == "__main__":
    _ = Backbone_Head_Module(None, None, None, None, None, None)
