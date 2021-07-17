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
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import average_precision, auroc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch
from featurization.data_utils import load_data_from_df, construct_loader, load_data_from_smiles
from transformer import make_model
root = Path(__file__).resolve().parents[1].absolute()


@dataclasses.dataclass(frozen=True)
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
        self.model_params = {
            'd_atom': 28,
            'd_model': 1024,
            'N': 8,
            'h': 16,
            'N_dense': 1,
            'lambda_attention': 0.33,
            'lambda_distance': 0.33,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'relu',
            'distance_matrix_kernel': 'exp',
            'dropout': 0.0,
            'aggregation_type': 'mean'
        }

        self.model = make_model(**self.model_params)
        pretrained_name = root / 'pretrained_weights.pt'
        pretrained_state_dict = torch.load(pretrained_name)
        pl.seed_everything(hparams['seed'])

        model_state_dict = self.model.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

    def forward(self, node_features, batch_mask, adjacency_matrix, distance_matrix):
        out = self.model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
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

    def shared_step(self, batch):
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_hat = self.forward(node_features, batch_mask, adjacency_matrix, distance_matrix)
        pos_weight = torch.Tensor([(1369 / 103)])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(y_hat, y)

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': y,
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

"""
    def train_dataloader(self):
        X, y = load_data_from_df('/home/dfa/AI4EU/train_input.csv', one_hot_formal_charge=True)
        data_loader = construct_loader(X, y, self.hparams.batch_size)
        return DataLoader(data_loader,
                          self.hparams.batch_size,
                          num_workers=8, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        X, y = load_data_from_df('/home/dfa/AI4EU/valid_input.csv', one_hot_formal_charge=True)
        data_loader = construct_loader(X, y, self.batch_size)
        return DataLoader(data_loader,
                          self.hparams.batch_size,
                          num_workers=8, drop_last=True,
                          pin_memory=True)

    def test_dataloader(self):
        X, y = load_data_from_df('/home/dfa/AI4EU/test_input.csv', one_hot_formal_charge=True)
        data_loader = construct_loader(X, y, self.hparams.batch_size)
        return DataLoader(data_loader,
                          self.hparams.batch_size,
                          num_workers=8, drop_last=True,
                          pin_memory=True)
"""


def main():
    conf = Conf(
        lr=1e-4,
        batch_size=64,
        epochs=300,
        reduce_lr=True,
    )

    model = TransformerNet(
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
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

    print("Starting training")
    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        gpus=[1],  # [0]
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
    )
    chembl_4_smiles = pd.read_csv(root / 'data/chembl_4_smiles.csv')[['smiles', 'withdrawn']]
    chembl_4_smiles = chembl_4_smiles.sample(frac=1, random_state=0)
    X, y = load_data_from_smiles(chembl_4_smiles['smiles'], chembl_4_smiles['withdrawn'])
    data_loader = construct_loader(X, y, batch_size=conf.batch_size)

    train_test_splitter = StratifiedKFold(n_splits=5)
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15)

    fold_ap = []
    fold_auc_roc = []

    for k, (train_index, test_index) in enumerate(train_test_splitter.split(data_loader.dataset, y)):
        test_data_loader = DataLoader(Subset(data_loader.dataset, test_index.tolist()))
        train_data_loader = DataLoader(Subset(data_loader.dataset, train_index.tolist()))
        train_labels = [i.y[0] for i in train_data_loader.dataset]

        for train_index, val_index in train_val_splitter.split(train_data_loader.dataset, train_labels):
            train_data_loader = DataLoader(Subset(train_data_loader.dataset, train_index.tolist()))
            val_data_loader = DataLoader(Subset(train_data_loader.dataset, val_index.tolist()))

            trainer.fit(model, train_data_loader, val_data_loader)
            results = trainer.test(test_data_loader)
            results_path = Path(root / "results")
            test_ap = round(results[0]['test_ap'], 3)
            test_auc = round(results[0]['test_auc'], 3)
            cv_fold = k

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
        print('AP for fold {}= {}'.format(i, result))


if __name__ == '__main__':
    main()