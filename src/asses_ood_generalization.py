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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch
from featurization.data_utils import load_data_from_df, construct_dataset, load_data_from_smiles, mol_collate_func
from transformer import make_model
import click

root = Path(__file__).resolve().parents[1].absolute()


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
        version = self.trainer.logger.version[-10:]
        items["v_num"] = version
        return items

@click.command()
@click.option('-train_data', default='chembl_4_smiles.csv')
@click.option('-train_set', default='chembl')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
def main(train_data, train_set, batch_size, gpu):
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

    if train_set == 'chembl':
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.sample(frac=1, random_state=0)
        ood_1 = data.loc[(data['dataset'] == 'drugbank')][['smiles', 'drugbank_withdrawn']]
        ood_2 = data.loc[(data['dataset'] == 'withdrawn')][['smiles', 'withdrawn_label']]
        data = data.loc[(data['dataset'] == 'chembl') |
                        (data['dataset'] == 'both')][['smiles', 'chembl_withdrawn']]

        data.rename(columns={'chembl_withdrawn': 'withdrawn'}, inplace=True)
        ood_1.rename(columns={'drugbank_withdrawn': 'withdrawn'}, inplace=True)
        ood_2.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)

    if train_set == 'drugbank':
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.sample(frac=1, random_state=0)
        ood_1 = data.loc[(data['dataset'] == 'chembl')][['smiles', 'chembl_withdrawn']]
        ood_2 = data.loc[(data['dataset'] == 'withdrawn')][['smiles', 'withdrawn_label']]
        data = data.loc[(data['dataset'] == 'drugbank') |
                        (data['dataset'] == 'both')][['smiles', 'drugbank_withdrawn']]

        data.rename(columns={'drugbank_withdrawn': 'withdrawn'}, inplace=True)
        ood_1.rename(columns={'chembl_withdrawn': 'withdrawn'}, inplace=True)
        ood_2.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)

    if train_set == 'wd_db':
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.sample(frac=1, random_state=0)
        ood_1 = data.loc[(data['dataset'] == 'chembl')][['smiles', 'chembl_withdrawn']]
        ood_2 = data.loc[(data['dataset'] == 'withdrawn')][['smiles', 'withdrawn_label']]
        data = data.loc[(data['dataset'] == 'drugbank') |
                        (data['dataset'] == 'both')][['smiles', 'withdrawn_label']]

        data.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)
        ood_1.rename(columns={'chembl_withdrawn': 'withdrawn'}, inplace=True)
        ood_2.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)

    if train_set == 'wd_chembl':
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.sample(frac=1, random_state=0)
        ood_1 = data.loc[(data['dataset'] == 'drugbank')][['smiles', 'drugbank_withdrawn']]
        ood_2 = data.loc[(data['dataset'] == 'withdrawn')][['smiles', 'withdrawn_label']]
        data = data.loc[(data['dataset'] == 'chembl') |
                        (data['dataset'] == 'both')][['smiles', 'withdrawn_label']]

        data.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)
        ood_1.rename(columns={'drugbank_withdrawn': 'withdrawn'}, inplace=True)
        ood_2.rename(columns={'withdrawn_label': 'withdrawn'}, inplace=True)

    train_test_splitter = StratifiedKFold(n_splits=5)
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15)

    fold_ap = []
    fold_auc_roc = []
    cv_fold = []

    ood_1_prior = []
    ood_1_ap = []
    ood_1_auc_roc = []

    ood_2_ap = []
    ood_2_auc_roc = []

    for k, (train_index, test_index) in enumerate(
            train_test_splitter.split(data, data['withdrawn'])
    ):
        X_test, y_test = load_data_from_smiles(data.iloc[test_index]['smiles'],
                                               data.iloc[test_index]['withdrawn'],
                                               one_hot_formal_charge=True)
        test_dataset = construct_dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=mol_collate_func, batch_size=conf.batch_size)

        train_data = data.iloc[train_index]

        for train_index_2, val_index in train_val_splitter.split(train_data, train_data['withdrawn']):
            X_val, y_val = load_data_from_smiles(train_data.iloc[val_index]['smiles'],
                                                 train_data.iloc[val_index]['withdrawn'],
                                                 one_hot_formal_charge=True)
            val_dataset = construct_dataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, collate_fn=mol_collate_func, num_workers=0, batch_size=conf.batch_size)

            X_train, y_train = load_data_from_smiles(train_data.iloc[train_index_2]['smiles'],
                                                     train_data.iloc[train_index_2]['withdrawn'],
                                                     one_hot_formal_charge=True)
            train_dataset = construct_dataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, collate_fn=mol_collate_func, num_workers=0,
                                      batch_size=conf.batch_size)

            pos_weight = torch.Tensor([(train_data.iloc[train_index_2]['withdrawn'].value_counts()[0] /
                                        train_data.iloc[train_index_2]['withdrawn'].value_counts()[1])])

            conf.pos_weight = pos_weight

            model = TransformerNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
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


            X_ood_1, y_ood_1 = load_data_from_smiles(ood_1['smiles'], ood_1['withdrawn'])
            ood_1_dataset = construct_dataset(X_ood_1, y_ood_1)
            ood_1_loader = DataLoader(ood_1_dataset, collate_fn=mol_collate_func, num_workers=0,
                                      batch_size=conf.batch_size)

            results_ood_1 = trainer.test(model, ood_1_loader)
            ood_1_ap_result = round(results_ood_1[0]['test_ap'], 3)
            ood_1_auc_result = round(results_ood_1[0]['test_auc'], 3)

            X_ood_2, y_ood_2 = load_data_from_smiles(ood_2['smiles'], ood_2['withdrawn'])
            ood_2_dataset = construct_dataset(X_ood_2, y_ood_2)
            ood_2_loader = DataLoader(ood_2_dataset, collate_fn=mol_collate_func, num_workers=0,
                                      batch_size=conf.batch_size)

            results_ood_2 = trainer.test(model, ood_2_loader)
            ood_2_ap_result = round(results_ood_2[0]['test_ap'], 3)
            ood_2_auc_result = round(results_ood_2[0]['test_auc'], 3)

            ood_1_ap.append(ood_1_ap_result)
            ood_1_auc_roc.append(ood_1_auc_result)

            ood_2_ap.append(ood_2_ap_result)
            ood_2_auc_roc.append(ood_2_auc_result)

            ood_1_prior_fold = ood_1['withdrawn'].value_counts()[1] / (
                ood_1['withdrawn'].value_counts()[1] + ood_1['withdrawn'].value_counts()[0]
            )

            ood_1_prior.append(ood_1_prior_fold)


            fold_ap.append(test_ap)
            fold_auc_roc.append(test_auc)

            if not results_path.exists():
                results_path.mkdir(exist_ok=True, parents=True)
                with open(results_path / "classification_results_ood.txt", "w") as file:
                    file.write("OOD Classification results")
                    file.write("\n")

            results = {'Test AP': test_ap,
                       'Test AUC-ROC': test_auc,
                       'CV_fold': cv_fold,
                       'ood_1_ap': ood_1_ap_result,
                       'ood_1_auc': ood_1_auc_result,
                       'ood_1_prior': ood_1_prior_fold,
                       'withdrawn_ap': ood_2_ap_result,
                       'withdrawn_auc': ood_2_auc_result}
            version = {'version': logger.version}
            results = {logger.name: [results, version]}
            with open(results_path / "classification_results_ood.txt", "a") as file:
                print(results, file=file)
                file.write("\n")

    print('Average AP across folds: {}'.format(np.mean(fold_ap)))
    print('Average AUC across folds: {}'.format(np.mean(fold_auc_roc)))
    print('\n')

    print('Average ood_1 AP across folds: {}'.format(np.mean(ood_1_ap)))
    print('Average ood_1 AUC across folds: {}'.format(np.mean(ood_1_auc_roc)))
    print('Prior probability of ood_1: {}'.format(np.mean(ood_1_prior)))

    print('\n')
    print('Average ood_2 AP across folds: {}'.format(np.mean(ood_2_ap)))
    print('Average ood_2 AUC across folds: {}'.format(np.mean(ood_2_auc_roc)))
    print('Prior probability of ood_1: 1')


    results_df = pd.DataFrame({'CV_fold': cv_fold, 'AP': fold_ap, 'AUC': fold_auc_roc,
                               'ood_1_prior': ood_1_prior, 'ood_1_AP': ood_1_ap, 'ood_1_AUC': ood_1_auc_roc,
                               'ood_2_AP': ood_2_ap, 'ood_2_AUC': ood_2_auc_roc}).to_csv(
        results_path / "{}_ood_metrics.csv".format(logger.version))


if __name__ == '__main__':
    main()
