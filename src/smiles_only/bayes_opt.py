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
from torch.utils.data import DataLoader
import torch
from skopt.utils import use_named_args
from data_utils import construct_dataset, load_data_from_smiles, mol_collate_func
from transformer import make_model
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
    lr: float = 0.0005
    batch_size: int = 32
    epochs: int = 100
    ckpt_path: Optional[str] = None
    reduce_lr: Optional[bool] = False
    d_model: int = 64
    N: int = 4
    h: int = 8
    N_dense: int = 2
    lambda_attention: float = 0.33
    lambda_distance: float = 0.33
    leaky_relu_slope: int = 0.1
    distance_matrix_kernel: str = 'softmax'
    dropout: float = 0.0
    dense_output_nonlinearity: str = 'tanh'
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
            'd_model': self.hparams.d_model,
            'N': self.hparams.N,
            'h': self.hparams.h,
            'N_dense': self.hparams.N_dense,
            'lambda_attention': self.hparams.lambda_attention,
            'lambda_distance': self.hparams.lambda_distance,
            'leaky_relu_slope': self.hparams.leaky_relu_slope,
            'dense_output_nonlinearity': self.hparams.dense_output_nonlinearity,
            'distance_matrix_kernel': self.hparams.distance_matrix_kernel,
            'dropout': self.hparams.dropout,
            'aggregation_type': 'mean',
        }

        pl.seed_everything(hparams['seed'])
        self.model = make_model(**self.model_params)

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
                patience=6,
                factor=0.1,
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

    """
    param_space = {
        'lr': Real(0.0005, 0.01, name='lr'),
        'd_model': Integer(28, 1024, name='d_model'),
        'N': Integer(1, 16, name='N'),
        'h': Integer(1, 16, name='h'),
        'N_dense': Integer(1, 4, name='N_dense'),
        'lambda_attention': Real(0.1, 0.9, name='lambda_attention'),
        'lambda_distance': Real(0.1, 0.9, name='lambda_distance'),
        'leaky_relu_slope': Real(0.01, 0.5, name='leaky_relu_slope'),
        'distance_matrix_kernel': Categorical(['exp', 'softmax'], name='distance_matrix_kernel'),
        'dropout': Real(0.05, 0.5, name='dropout'),
        'dense_output_nonlinearity': Categorical(['tanh', 'relu'], name='dense_output_nonlinearity')
    }
    """

    dim_1 = Real(0.0005, 0.01, name='lr')
    dim_2 = Categorical([32, 64, 128, 256, 512, 1024], name='d_model')
    dim_3 = Integer(2, 16, name='N')
    dim_4 = Categorical([4, 8, 16], name='h')
    dim_5 = Integer(1, 6, name='N_dense')
    dim_6 = Real(0.1, 0.9, name='lambda_attention')
    dim_7 = Real(0.1, 0.9, name='lambda_distance')
    dim_8 = Real(0.01, 0.5, name='leaky_relu_slope')
    dim_9 = Categorical(['exp', 'softmax'], name='distance_matrix_kernel')
    dim_10 = Real(0.05, 0.5, name='dropout')
    dim_11 = Categorical(['tanh', 'relu'], name='dense_output_nonlinearity')

    dimensions = [dim_1, dim_2, dim_3, dim_4, dim_5,
                  dim_6, dim_7, dim_8, dim_9, dim_10, dim_11]

    @use_named_args(dimensions=dimensions)
    def maximize_ap(lr, d_model, N, h, N_dense, lambda_attention, lambda_distance, leaky_relu_slope,
                    distance_matrix_kernel, dropout, dense_output_nonlinearity):
        train_test_splitter = StratifiedKFold(n_splits=2)

        fold_ap = []

        for k, (train_index, test_index) in enumerate(
                train_test_splitter.split(X_data, y_data)
        ):

            conf = Conf(
                lr=lr,
                batch_size=batch_size,
                reduce_lr=True,
                d_model=d_model,
                N=N,
                h=h,
                N_dense=N_dense,
                lambda_attention=lambda_attention,
                lambda_distance=lambda_distance,
                leaky_relu_slope=leaky_relu_slope,
                dense_output_nonlinearity=dense_output_nonlinearity,
                distance_matrix_kernel=distance_matrix_kernel,
                dropout=dropout,
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
                                                patience=10,
                                                verbose=False)

            model_checkpoint = ModelCheckpoint(
                    dirpath=(logger.log_dir + '/checkpoint/'),
                    monitor='val_ap_epoch',
                    mode='max',
                    save_top_k=1,
            )

            X_test = X_data[test_index]
            y_test = y_data[test_index]
            test_dataset = construct_dataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, num_workers=0, collate_fn=mol_collate_func,
                                     batch_size=conf.batch_size)

            train_set = X_data[train_index]
            train_labels = y_data[train_index]

            X_train, X_val, y_train, y_val = train_test_split(train_set, train_labels,
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
                callbacks=[early_stop_callback, model_checkpoint],
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0
            )

            trainer.fit(model, train_loader, val_loader)
            results = trainer.test(model, test_loader)
            test_ap = round(results[0]['test_ap'], 3)

            fold_ap.append(test_ap)

        return 1/np.mean(fold_ap)

    start = time()
    res = gp_minimize(maximize_ap,  # the function to minimize
                      dimensions=dimensions,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=20,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=1234)  # the random seed
    end = time()
    elapsed = (end-start) / 3600

    print('Value of the minimum: {}'.format(res.fun))
    print('Res space: {}'.format(res.space))
    print('Time elapsed in hrs: {}'.format(elapsed))

    results_path = Path(root / 'bayes_opt')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "bayes_opt.txt", "w") as file:
            file.write("Bayes opt")
            file.write("\n")

        with open(results_path / "bayes_opt.txt", "a") as file:
            print('Maximum AP: {}'.format(res.fun), file=file)
            print('Res space: {}'.format(res.space), file=file)
            file.write("\n")
            file.write("\n")

if __name__ == '__main__':
    main()
