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
from tdc.single_pred import ADME, Tox
from torchmetrics.functional import average_precision, auroc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
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
    save_dir = '{}/models/TDC/'.format(root)
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
            problem,
            hparams,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.task = problem
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

        if self.task == 'classification':
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

        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(predictions, targets))

            self.log('val_loss',
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

        if self.task == 'classification':
            ap = average_precision(predictions, targets)
            auc = auroc(predictions, targets)

            log_metrics = {
                'test_ap': ap,
                'test_auc': auc
            }
            self.log_dict(log_metrics)

        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(predictions, targets))
            self.log('test_loss',
                     loss,
                     on_step=False, on_epoch=True, prog_bar=True)

    def shared_step(self, batch, batchidx):
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        # y_hat = self.forward(node_features, batch_mask, adjacency_matrix, distance_matrix)
        y_hat = self.model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        if self.task == 'classification':
            pos_weight = self.hparams.pos_weight.to("cuda")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(y_hat, y)
        else:
            loss_fn = torch.nn.MSELoss()
            loss = torch.sqrt(loss_fn(y_hat, y))

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
        version = self.trainer.logger.version[-10:]
        items["v_num"] = version
        return items


@click.command()
@click.option('-problem')
@click.option('-dataset')
@click.option('-batch_size', default=8)
@click.option('-gpu', default=1)
def main(problem, dataset, batch_size, gpu):
    conf = Conf(
        lr=1e-4,
        batch_size=batch_size,
        epochs=100,
        reduce_lr=True,
    )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='transformer_net',
        version='{}_{}'.format(problem, dataset),
    )

    # Copy this script and all files used in training
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__), log_dir)

    if problem == 'ADME':
        data = ADME(name=dataset)

    elif problem == 'Tox':
        data = Tox(name=dataset)

    splits = data.get_split()
    train = splits['train']
    val = splits['valid']
    test = splits['test']

    X_train, y_train = load_data_from_smiles(train['Drug'],
                                             train['Y'],
                                             one_hot_formal_charge=True)
    train_dataset = construct_dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, collate_fn=mol_collate_func, num_workers=0, batch_size=conf.batch_size)

    X_val, y_val = load_data_from_smiles(val['Drug'],
                                         val['Y'],
                                         one_hot_formal_charge=True)
    val_dataset = construct_dataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, collate_fn=mol_collate_func, num_workers=0, batch_size=conf.batch_size)

    X_test, y_test = load_data_from_smiles(test['Drug'],
                                           test['Y'],
                                           one_hot_formal_charge=True)
    test_dataset = construct_dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, collate_fn=mol_collate_func, num_workers=0, batch_size=conf.batch_size)

    if len(train['Y'].value_counts()) == 2:
        task = 'classification'

        pos_weight = torch.Tensor([(train['Y'].value_counts()[0] /
                                    train['Y'].value_counts()[1])])
        conf.pos_weight = pos_weight

    else:
        task = 'regression'

    early_stop_callback = EarlyStopping(monitor=('val_ap_epoch' if task == 'classification' else 'val_loss_epoch'),
                                        min_delta=0.00,
                                        mode=('max' if task == 'classification' else 'min'),
                                        patience=10,
                                        verbose=False)

    model = TransformerNet(
        hparams=conf.to_hparams(),
        problem=task,
        reduce_lr=conf.reduce_lr,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=(logger.log_dir + '/checkpoint/'),
        monitor=('val_ap_epoch' if task == 'classification' else 'val_loss_epoch'),
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

    if task == 'classification':
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
        print('Task {}-{} Test AUC {}'.format(task, dataset, test_auc))
        print('Task {}-{} Test AP {}'.format(task, dataset, test_ap))

    else:
        test_rmse = round(results[0]['test_loss'], 3)
        print('Task {}-{} Test RMSE {}'.format(task, dataset, test_rmse))

    results_path = Path(root / "results")
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "TDC_results.txt", "w") as file:
            file.write("Production version")
            file.write("\n")

    version = {'version': logger.version,
               'task': '{}-{}'.format(problem, dataset)}
    if task == 'classification':
        results = {'Test AP': test_ap,
                   'Test AUC-ROC': test_auc}
        results = {logger.name: [results, version]}
    else:
        results = {'Test RMSE': test_rmse}
        results = {logger.name: [results, version]}

    with open(results_path / "TDC_results.txt", "a") as file:
        print(results, file=file)
        file.write("\n")


if __name__ == '__main__':
    main()
