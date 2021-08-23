import dataclasses
import shutil
from abc import ABC
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Optional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Integer, Real, Categorical
from torchmetrics.functional import average_precision, auroc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch_geometric.data import DataLoader
import torch
from skopt.utils import use_named_args
from data_utils import smiles2graph
from EGConv import EGConvNet
import click

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
    epochs: int = 100
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
    hidden_channels: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_bases: int = 4
    pos_weight: torch.Tensor = torch.Tensor([8])


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


class TransformerNet(pl.LightningModule, ABC):
    def __init__(
            self,
            hparams,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.reduce_lr = reduce_lr
        self.model = EGConvNet(
            self.hparams.hidden_channels,
            self.hparams.num_layers,
            self.hparams.num_heads,
            self.hparams.num_bases,
            aggregator=['sum', 'mean', 'max']
        )
        pl.seed_everything(hparams['seed'])

    def forward(self, data):
        out = self.model(data.x, data.edge_index, data.batch, None)
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

        ap = average_precision(predictions, targets)
        auc = auroc(predictions, targets)

        log_metrics = {
            'val_ap_epoch': ap,
            'val_auc_epoch': auc
        }
        self.log_dict(log_metrics)
        self.log('val_ap',
                 ap,
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        metrics = self.shared_step(batch, batch_idx)
        return {
            "predictions": metrics.get("predictions"),
            "targets": metrics.get("targets")
        }

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x.get('predictions') for x in outputs], 0)
        target = torch.cat([x.get('targets') for x in outputs], 0)

        ap = average_precision(predictions, target)
        auc = auroc(predictions, target)

        log_metrics = {
            'test_ap': ap,
            'test_auc': auc,
        }
        self.log_dict(log_metrics)

    def shared_step(self, data, batchidx):
        y_hat = self.model(data.x, data.edge_index, data.batch)
        pos_weight = self.hparams.pos_weight.to("cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(y_hat, data.y.unsqueeze(-1))

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': data.y.unsqueeze(-1).long(),
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
                patience=10,
                factor=0.5,
            ),
            'monitor': 'val_loss'
        }

        if self.reduce_lr is False:
            return [opt]

        return [opt], [sched]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        return items

@click.command()
@click.option('-train_data', default='chembl_train.csv')
@click.option('-test_data', default='chembl_test.csv')
@click.option('-withdrawn_col', default='withdrawn')
@click.option('-batch_size', default=16)
@click.option('--gpu', default=1)
def main(train_data, test_data, withdrawn_col, batch_size, gpu):
    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=2048,
        num_layers=2,
        num_heads=4,
        num_bases=7,
    )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='transformer_net',
        version='{}'.format(str(int(time()))),
    )

    # Copy this script and all files used in training
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__), log_dir)

    early_stop_callback = EarlyStopping(monitor='val_ap_epoch',
                                        min_delta=0.00,
                                        mode='max',
                                        patience=15,
                                        verbose=False)

    train_data = pd.read_csv(root / 'data/{}'.format(train_data))[['smiles', withdrawn_col]]
    train_data = train_data.sample(frac=1, random_state=0)

    test_data = pd.read_csv(root / 'data/{}'.format(test_data))[['smiles', withdrawn_col]]
    test_data_list = []
    for index, row in test_data.iterrows():
        test_data_list.append(smiles2graph(row, withdrawn_col))
    test_loader = DataLoader(test_data_list, num_workers=0, batch_size=conf.batch_size)

    train, val = train_test_split(train_data, test_size=0.15, stratify=train_data[withdrawn_col], shuffle=True)

    train_data_list = []
    for index, row in train.iterrows():
        train_data_list.append(smiles2graph(row, withdrawn_col))
    train_loader = DataLoader(train_data_list, num_workers=0, batch_size=conf.batch_size)

    val_data_list = []
    for index, row in val.iterrows():
        val_data_list.append(smiles2graph(row, withdrawn_col))
    val_loader = DataLoader(val_data_list, num_workers=0, batch_size=conf.batch_size)

    pos_weight = torch.Tensor([(len(train) / len(train.loc[train['withdrawn'] == 1]))])
    conf.pos_weight = pos_weight

    model = TransformerNet(
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )

    model_checkpoint = ModelCheckpoint(
            dirpath=(logger.log_dir + '/checkpoint/'),
            monitor='val_ap_epoch',
            mode='max',
            save_top_k=1,
    )

    print("Starting training")
    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        gpus=[gpu],  # [0]
        logger=logger,
        resume_from_checkpoint=conf.ckpt_path,  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback, model_checkpoint],
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(model, test_loader)

    test_ap = round(results[0]['test_ap'], 3)
    test_auc = round(results[0]['test_auc'], 3)

    print('Test Average AP: {}'.format(test_ap))
    print('Test ROC-AUC: {}'.format(test_auc))

    results_path = Path(root / "results")
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "production_results.txt", "w") as file:
            file.write("Production version")
            file.write("\n")


    version = {'version': logger.version}
    results = {logger.name: version}
    with open(results_path / "production_results.txt", "a") as file:
        print(results, file=file)
        file.write("\n")


if __name__ == '__main__':
    main()
