from abc import ABC
from pathlib import Path
from pprint import pformat
from typing import Optional, Dict
import dataclasses
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import auroc, average_precision
from src.smiles_only.EGConv import EGConvModel
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
            problem,
            hparams,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.task = problem
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
        }

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        targets = torch.cat([x.get('targets') for x in outputs], 0)

        if self.task == 'ap':
            ap = average_precision(predictions, targets)
            self.log('result',
                     ap,
                     on_step=False, on_epoch=True, prog_bar=True)

        elif self.task == 'auc':
            auc = auroc(predictions, targets)
            self.log('result',
                     auc,
                     on_step=False, on_epoch=True, prog_bar=True)

        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(predictions, targets))
            self.log('result',
                     loss,
                     on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        return {
            "predictions": metrics.get("predictions"),
            "targets": metrics.get("targets")
        }

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        targets = torch.cat([x.get('targets') for x in outputs], 0)

        if self.task == 'ap':
            ap = average_precision(predictions, targets)
            auc = auroc(predictions, targets)

            log_metrics = {
                'test_ap': ap,
                'test_auc': auc,
            }
            self.log_dict(log_metrics)

        elif self.task == 'auc':
            auc = auroc(predictions, targets)
            ap = average_precision(predictions, targets)

            log_metrics = {
                'test_ap': ap,
                'test_auc': auc,
            }
            self.log_dict(log_metrics)

        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(predictions, targets))
            self.log('test_mse',
                     loss)

    def shared_step(self, data, batchidx):
        y_hat = self.model(data.x, data.edge_index, data.batch)
        if self.task == 'ap' or self.task == 'auc':
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(y_hat, data.y.unsqueeze(-1))
        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(y_hat, data.y.unsqueeze(-1)))

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': data.y.unsqueeze(-1),
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
