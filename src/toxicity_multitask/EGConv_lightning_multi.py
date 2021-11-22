from abc import ABC
from pathlib import Path
from pprint import pformat
from typing import Optional, Dict
import dataclasses
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import auroc, average_precision
from src.toxicity_multitask.EGConv_multi import EGConvModel
import pytorch_lightning as pl


root = Path(__file__).resolve().parents[2].absolute()


@dataclasses.dataclass(
    # frozen=True
)
class Conf:
    gpus: int = 1
    seed: int = 42
    use_16bit: bool = False
    save_dir = '{}/models/'.format(root)
    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 300
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
    hidden_channels: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_bases: int = 4

    def to_hparams(self) -> Dict:
        excludes = [
            'ckpt_path',
            'reduce_lr',
        ]
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in excludes
        }

    def __str__(self):
        return pformat(dataclasses.asdict(self))


class EGConvNet(pl.LightningModule, ABC):
    def __init__(
            self,
            hparams,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.reduce_lr = reduce_lr
        self.model = EGConvModel(
            self.hparams.hidden_channels,
            self.hparams.num_layers,
            self.hparams.num_heads,
            self.hparams.num_bases,
            aggregator=['sum', 'mean', 'max']
        )
        pl.seed_everything(hparams['seed'])

    def forward(self, x, edge_index, batch):
        out = self.model(x, edge_index, batch)
        return out

    def training_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("loss", metrics.get("loss"))
        return metrics.get("loss")

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        self.log('val_loss',
                 metrics.get("loss"),
                 prog_bar=False, on_step=False)
        self.log('val_loss_epoch',
                 metrics.get("loss"),
                 on_step=False, on_epoch=True, prog_bar=False)
        return {
            "predictions": metrics.get("predictions"),
            "targets": metrics.get("targets"),
            "tox_targets": metrics.get("tox_targets")
        }

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        targets = torch.cat([x.get('targets') for x in outputs], 0)
        tox_targets = torch.cat([x.get('tox_targets') for x in outputs], 0)
        ap = average_precision(predictions[:, 0], targets)
        auc = auroc(predictions[:, 0], targets)

        ap_tox = average_precision(predictions[:, 1:], tox_targets, num_classes=9)
        auc_tox = auroc(predictions[:, 1:], tox_targets, num_classes=9, average='weighted')

        weights = torch.Tensor([
            torch.Tensor([1.1]),
            torch.Tensor([66.37]),
            torch.Tensor([132.74]),
            torch.Tensor([60.05]),
            torch.Tensor([109.65]),
            torch.Tensor([100.88]),
            torch.Tensor([47.58]),
            torch.Tensor([114.63]),
            torch.Tensor([109.65]),
        ])

        weighted = torch.mul(torch.Tensor(ap_tox), weights)
        ap_tox = torch.sum(torch.Tensor(weighted)) / 742.65

        #ap_tox = torch.sum(torch.Tensor(ap_tox)) / 9
        ap_tox = torch.sum(torch.Tensor(weighted)) / 742.65

        log_metrics = {
            'val_ap_epoch': (ap + ap_tox) / 2,
            'val_auc_epoch': (auc + auc_tox) / 2,
        }
        self.log_dict(log_metrics)
        self.log('val_ap',
                 (ap + ap_tox),
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        return {
            "predictions": metrics.get("predictions"),
            "targets": metrics.get("targets"),
            "tox_targets": metrics.get("tox_targets")
        }

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        target = torch.cat([x.get('targets') for x in outputs], 0)
        tox_targets = torch.cat([x.get('tox_targets') for x in outputs], 0)

        ap = average_precision(predictions[:, 0], target)
        auc = auroc(predictions[:, 0], target)

        ap_tox = average_precision(predictions[:, 1:], tox_targets, num_classes=9)
        auc_tox = auroc(predictions[:, 1:], tox_targets, num_classes=9, average='weighted')

        print(torch.Tensor(ap_tox))
        weights = torch.Tensor([
            torch.Tensor([1.1]),
            torch.Tensor([66.37]),
            torch.Tensor([132.74]),
            torch.Tensor([60.05]),
            torch.Tensor([109.65]),
            torch.Tensor([100.88]),
            torch.Tensor([47.58]),
            torch.Tensor([114.63]),
            torch.Tensor([109.65]),
        ])

        weighted = torch.mul(torch.Tensor(ap_tox), weights)
        ap_tox = torch.sum(torch.Tensor(weighted)) / 742.65

        log_metrics = {
            'test_ap': ap,
            'test_auc': auc,
            'test_ap_tox': ap_tox,
            'test_auc_tox': auc_tox
        }
        self.log_dict(log_metrics)

    def shared_step(self, data, batchidx):
        y_hat = self.model(data.x, data.edge_index, data.batch)
        withdrawn = y_hat[:, 0]
        toxicities = y_hat[:, 1:]
        loss_fn_withdrawn = torch.nn.BCEWithLogitsLoss()
        loss_fn_toxicity = torch.nn.CrossEntropyLoss()
        loss_wd = loss_fn_withdrawn(withdrawn, data.y)
        loss_tox = loss_fn_toxicity(toxicities, data.toxicity.long())
        loss = loss_wd + loss_tox

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': data.y.long(),
            'tox_targets': data.toxicity.long(),
        }

    def configure_optimizers(self):
        opt = Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
        )

        sched = {
            'scheduler': ReduceLROnPlateau(
                opt,
                mode='min',
                patience=5,
                factor=0.5,
            ),
            'monitor': 'val_loss'
        }

        if self.reduce_lr is False:
            return [opt]

        return [opt], [sched]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # version = self.trainer.logger.version[-10:]
        # items["v_num"] = version
        return items
