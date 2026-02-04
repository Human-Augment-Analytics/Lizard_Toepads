from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ModelConfig:
    num_keypoints: int
    lr: float = 1e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    freeze_backbone_until: Optional[int] = None


class KeypointRegressor(LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config.num_keypoints * 2)

        if config.freeze_backbone_until is not None:
            self._freeze_layers(config.freeze_backbone_until)

        self.criterion = nn.SmoothL1Loss(beta=0.1)

    def _freeze_layers(self, num_layers: int) -> None:
        frozen = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
        for child in list(self.backbone.children())[::-1]:
            for param in child.parameters():
                param.requires_grad = True
            frozen += 1
            if frozen >= num_layers:
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx: int):
        images = batch["image"]
        targets = batch["keypoints"]
        preds = self(images)
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx: int):
        images = batch["image"]
        targets = batch["keypoints"]
        preds = self(images)
        loss = self.criterion(preds, targets)
        l1 = torch.mean(torch.abs(preds - targets))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_l1", l1, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.config.lr,
            weight_decay=self.hparams.config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

