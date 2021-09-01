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
from torchmetrics.functional import average_precision, auroc
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch_geometric.data import DataLoader
import torch
from data_utils import smiles2graph_descriptors
from EGConv_descriptors import EGConvNet
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
    epochs: int = 300
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
    pos_weight: torch.Tensor = torch.Tensor([8])
    hidden_channels: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    num_bases: int = 4
    descriptors_len: int = 476


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
            self.hparams.descriptors_len,
            self.hparams.hidden_channels,
            self.hparams.num_layers,
            self.hparams.num_heads,
            self.hparams.num_bases,
            aggregator=['sum', 'mean', 'max']
        )
        pl.seed_everything(hparams['seed'])

    def forward(self, data):
        out = self.model(data.x, data.edge_index, data.batch, data.descriptors)
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
        y_hat = self.model(data.x, data.edge_index, data.batch, data.descriptors)
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
        # version = self.trainer.logger.version[-10:]
        # items["v_num"] = version
        return items


@click.command()
@click.option('-train_data', default='chembl_4_smiles.csv')
@click.option('-dataset', default='all')
@click.option('-withdrawn_col', default='withdrawn')
@click.option('-batch_size', default=16)
@click.option('-descriptors_from', default=9)
@click.option('-gpu', default=1)
def main(train_data, dataset, withdrawn_col, descriptors_from, batch_size, gpu):
    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data), index_col=0)
        data = data.sample(frac=1, random_state=0)

    else:
        data = pd.read_csv(root / 'data/{}'.format(train_data), index_col=0)
        data = data.loc[(data['dataset'] == dataset) |
                        (data['dataset'] == 'both') |
                        (data['dataset'] == 'withdrawn')]
        data = data.sample(frac=1, random_state=0)


    train_test_splitter = StratifiedKFold(n_splits=5, random_state=0)

    fold_ap = []
    fold_auc_roc = []
    cv_fold = []

    for k, (train_index, test_index) in enumerate(
            train_test_splitter.split(data, data[withdrawn_col])
    ):

        conf = Conf(
            lr=1e-4,
            batch_size=batch_size,
            epochs=100,
            reduce_lr=True,
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

        test = data.iloc[test_index]
        test_data_list = []
        for index, row in test.iterrows():
            test_data_list.append(smiles2graph_descriptors(row, withdrawn_col, descriptors_from=descriptors_from))
        test_loader = DataLoader(test_data_list, num_workers=0, batch_size=conf.batch_size)

        train_set = data.iloc[train_index]

        train, val = train_test_split(train_set, test_size=0.15, stratify=train_set[withdrawn_col], shuffle=True,
                                      random_state=0)

        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph_descriptors(row, withdrawn_col, descriptors_from=descriptors_from))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=conf.batch_size, shuffle=True)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph_descriptors(row, withdrawn_col, descriptors_from=descriptors_from))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=conf.batch_size)

        pos_weight = torch.Tensor([(len(train) / len(train.loc[train[withdrawn_col] == 1]))])
        conf.pos_weight = pos_weight

        model = TransformerNet(
            conf.to_hparams(),
            reduce_lr=conf.reduce_lr,
        )

        print("Starting training")
        trainer = pl.Trainer(
            max_epochs=conf.epochs,
            gpus=[gpu],  # [0]
            logger=logger,  # load from checkpoint instead of resume
            weights_summary='top',
            callbacks=[early_stop_callback],
            checkpoint_callback=ModelCheckpoint(
                dirpath=(logger.log_dir + '/checkpoint/'),
                monitor='val_ap_epoch',
                mode='max',
                save_top_k=1,
            ),
            deterministic=True,
            auto_lr_find=False,
            num_sanity_val_steps=0
        )

        trainer.fit(model, train_loader, val_loader)
        results = trainer.test(model, test_loader)
        results_path = Path(root / "results")
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
        cv_fold.append(k)

        fold_ap.append(test_ap)
        fold_auc_roc.append(test_auc)

        if not results_path.exists():
            results_path.mkdir(exist_ok=True, parents=True)
            with open(results_path / "classification_results.txt", "w") as file:
                file.write("Classification results")
                file.write("\n")

        results = {'Test AP': test_ap,
                   'Test AUC-ROC': test_auc,
                   'CV_fold': cv_fold}
        version = {'version': logger.version}
        results = {logger.name: [results, version]}
        with open(results_path / "classification_results.txt", "a") as file:
            print(results, file=file)
            file.write("\n")

    print('Average AP across folds: {}'.format(np.mean(fold_ap)))
    print('Average AUC across folds: {}'.format(np.mean(fold_auc_roc)))
    print('\n')

    for i, result in enumerate(fold_ap):
        print('AP for fold {}= {}'.format(i, result))

    for i, result in enumerate(fold_auc_roc):
        print('AUC for fold {}= {}'.format(i, result))

    results_df = pd.DataFrame({'CV_fold': cv_fold, 'AP': fold_ap, 'AUC': fold_auc_roc}).to_csv(
        results_path / "{}_metrics.csv".format(logger.version))


if __name__ == '__main__':
    main()
