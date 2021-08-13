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
from torch.utils.data import DataLoader
import torch
from data_utils import construct_dataset, load_data_from_smiles, mol_collate_func
from transformer import make_model
import click

root = Path(__file__).resolve().parents[2].absolute()


@dataclasses.dataclass(
    #frozen=True
)
class Conf:
    gpus: int = 1
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

    def shared_step(self, batch, batchidx):
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        # y_hat = self.forward(node_features, batch_mask, adjacency_matrix, distance_matrix)
        y_hat = self.model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        pos_weight = self.hparams.pos_weight.to("cuda")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(y_hat, y)

        return {
            'loss': loss,
            'predictions': y_hat,
            'targets': y.int(),
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
        #version = self.trainer.logger.version[-10:]
        #items["v_num"] = version
        return items

@click.command()
@click.option('-train_data', default='chembl_4_smiles.csv')
@click.option('-dataset', default='all')
@click.option('-withdrawn_col', default='withdrawn')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
def main(train_data, dataset, withdrawn_col, batch_size, gpu):

    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data))[['smiles', withdrawn_col]]
        data = data.sample(frac=1, random_state=0)

    else:
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.loc[(data['dataset'] == dataset) |
                        (data['dataset'] == 'both') |
                        (data['dataset'] == 'withdrawn')][['smiles', withdrawn_col]]
        data = data.sample(frac=1, random_state=0)

    X_data, y_data = load_data_from_smiles(data['smiles'], data[withdrawn_col], one_hot_formal_charge=True)
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    train_test_splitter = StratifiedKFold(n_splits=5)

    fold_ap = []
    fold_auc_roc = []
    cv_fold = []

    for k, (train_index, test_index) in enumerate(
            train_test_splitter.split(X_data, y_data)
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

        X_test = X_data[test_index]
        y_test = y_data[test_index]
        test_dataset = construct_dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=mol_collate_func, batch_size=conf.batch_size)

        train_set = X_data[train_index]
        train_labels = y_data[train_index]

        X_train, X_val, y_train, y_val = train_test_split(train_set,train_labels,
                                                          test_size=0.15, stratify=train_labels, shuffle=True)

        val_dataset = construct_dataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, collate_fn=mol_collate_func, num_workers=0, batch_size=conf.batch_size)

        train_dataset = construct_dataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, collate_fn=mol_collate_func, num_workers=0,
                                  batch_size=conf.batch_size)

        pos_weight = torch.Tensor([(len(y_train) / np.count_nonzero(y_train))])
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
