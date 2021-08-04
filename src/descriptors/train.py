import dataclasses
import shutil
from abc import ABC
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import average_precision, auroc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from transformer import Network
from dataset import MoleculesDataset, mol_collate_func
import click
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[2].absolute()


@dataclasses.dataclass()
class Conf:
    gpus: int = 2
    seed: int = 42
    use_16bit: bool = False
    save_dir = '{}/models/'.format(root)
    lr: float = 1e-4
    batch_size: int = 32
    epochs: int = 300
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
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
            descriptors_len,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.reduce_lr = reduce_lr
        self.model = Network(descriptors_len=descriptors_len)
        pl.seed_everything(hparams['seed'])

    def forward(self, node_features, batch_mask, adjacency_matrix, distance_matrix, descriptors):
        out = self.model(node_features, batch_mask, adjacency_matrix, distance_matrix, descriptors)
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
        predictions = torch.cat([x.get('predictions') for x in outputs], 0).squeeze(-1)
        targets = torch.cat([x.get('targets') for x in outputs], 0).squeeze(-1)

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
        predictions = torch.cat([x.get('predictions') for x in outputs], 0).squeeze(-1)
        target = torch.cat([x.get('targets') for x in outputs], 0).squeeze(-1)

        ap = average_precision(predictions, target)
        auc = auroc(predictions, target)

        log_metrics = {
            'test_ap': ap,
            'test_auc': auc,
        }
        self.log_dict(log_metrics)

    def shared_step(self, batch, batchidx):
        adjacency_matrix, node_features, distance_matrix, descriptors, labels = batch
        labels = labels.int()
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_hat = self.forward(node_features, batch_mask, adjacency_matrix, distance_matrix, descriptors).squeeze(-1)
        pos_weight = self.hparams.pos_weight.to("cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(y_hat, labels.float())

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': labels,
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
        version = self.trainer.logger.version[-10:]
        items["v_num"] = version
        return items

@click.command()
@click.option('-train_data')
@click.option('-withdrawn_col')
@click.option('-batch_size', type=int)
@click.option('-descriptors_from', type=int)
@click.option('-gpu', type=int)
def main(train_data, withdrawn_col, batch_size, gpu, descriptors_from):
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

    data = pd.read_csv(root / 'data/descriptors_datasets/{}'.format(train_data), index_col=0)
    data = data.sample(frac=1, random_state=0)

    descriptors_len = len(data.iloc[0][descriptors_from:])

    train_test_splitter = StratifiedKFold(n_splits=5)
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15)

    fold_ap = []
    fold_auc_roc = []
    cv_fold = []

    for k, (train_index, test_index) in enumerate(
            train_test_splitter.split(data, data[withdrawn_col])
    ):
        test_data = data.iloc[test_index]
        test_data = MoleculesDataset(test_data, withdrawn_col, descriptors_from)
        test_loader = DataLoader(test_data,
                                 num_workers=0,
                                 collate_fn=mol_collate_func,
                                 batch_size=conf.batch_size)

        train_data = data.iloc[train_index]

        for train_index_2, val_index in train_val_splitter.split(train_data, train_data[withdrawn_col]):
            val_data = train_data.iloc[val_index]
            train_data = train_data.iloc[train_index_2]

            val_data = MoleculesDataset(val_data, withdrawn_col, descriptors_from)
            val_loader = DataLoader(val_data,
                                    num_workers=0,
                                    collate_fn=mol_collate_func,
                                    batch_size=conf.batch_size)

            pos_weight = torch.Tensor([(train_data[withdrawn_col].value_counts()[0] /
                                        train_data[withdrawn_col].value_counts()[1])])
            conf.pos_weight = pos_weight

            train_data = MoleculesDataset(train_data, withdrawn_col, descriptors_from)
            train_loader = DataLoader(train_data,
                                      num_workers=0,
                                      collate_fn=mol_collate_func,
                                      batch_size=conf.batch_size)

            model = TransformerNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
                descriptors_len=descriptors_len
            )

            print("Starting training")
            trainer = pl.Trainer(
                max_epochs=conf.epochs,
                gpus=[gpu],  # [0]
                logger=logger,
                resume_from_checkpoint=conf.ckpt_path,  # load from checkpoint instead of resume
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
                with open(results_path / "descriptors_classification_results.txt", "w") as file:
                    file.write("Descriptors Classification results")
                    file.write("\n")

            results = {'Test AP': test_ap,
                       'Test AUC-ROC': test_auc,
                       'CV_fold': cv_fold}
            version = {'version': logger.version}
            results = {logger.name: [results, version]}
            with open(results_path / "descriptors_classification_results.txt", "a") as file:
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
