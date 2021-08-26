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
from nonconformist.icp import IcpClassifier
from nonconformist.cp import TcpClassifier
from nonconformist.acp import CrossSampler, AggregatedCp, CrossConformalClassifier
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import average_precision, auroc
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch_geometric.data import DataLoader
import torch
from data_utils import smiles2graph
from EGConv import EGConvNet
import click
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc, MarginErrFunc
from sklearn.metrics import average_precision_score
from nonconformist.evaluation import class_mean_errors

root = Path(__file__).resolve().parents[2].absolute()

class TcpClassifier(TcpClassifier):
    def __init__(self, nc_function, condition=None, smoothing=True):
        self.train_x, self.train_y = None, None
        self.nc_function = nc_function
        super(TcpClassifier, self).__init__()

        # Check if condition-parameter is the default function (i.e.,
        # lambda x: 0). This is so we can safely clone the object without
        # the clone accidentally having self.conditional = True.
        default_condition = lambda x: 0
        is_default = (callable(condition) and
                      (condition.__code__.co_code ==
                       default_condition.__code__.co_code))

        if is_default:
            self.condition = condition
            self.conditional = False
        elif callable(condition):
            self.condition = condition
            self.conditional = True
        else:
            self.condition = lambda x: 0
            self.conditional = False

        self.smoothing = smoothing

        self.base_icp = IcpClassifier(
            self.nc_function,
            self.condition,
            self.smoothing
        )

        self.classes = None

    def fit(self, train):
        self.train_x = train
        self.train_y = train['withdrawn']
        self.classes = np.unique(train['withdrawn'])

class


class ClassifierAdapter(ClassifierAdapter):
    def __init__(self, trainer, model, batch_size):
        super(ClassifierAdapter, self).__init__(model)
        self.trainer = trainer
        self.model = model
        self.batch_size = batch_size

    def fit(self, train, val):
        train_data_list = []
        for row in train:
            train_data_list.append(smiles2graph(row[0], row[1]))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=self.batch_size)

        val_data_list = []
        for row in val:
            val_data_list.append(smiles2graph(row[0], row[1]))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=self.batch_size)


        self.trainer.fit(self.model, train_loader, val_loader)

    def _underlying_predict(self, test):
        self.model.eval()
        test_data_list = []
        for row in test:
            test_data_list.append(smiles2graph(row[0], row[1]))
        test_loader = DataLoader(test_data_list, num_workers=0, batch_size=self.batch_size)

        predictions = []
        for batch in test_loader:
            prediction = self.model.forward(batch).cpu().detach().numpy()
            predictions.append(prediction)

        predictions = np.array([cell for row in predictions for cell in row]) # flatten the list
        odds = np.exp(predictions)
        predictions = odds / (1 + odds)
        return predictions

        # obtain predictions from self.model and fill `predictions`
        # appropriately, with one real-valued prediction per
        # test object


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
    hidden_channels: int = 512
    num_layers: int = 7
    num_heads: int = 2
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
        out = self.model(data.x, data.edge_index, data.batch)
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
        # version = self.trainer.logger.version[-10:]
        # items["v_num"] = version
        return items


@click.command()
@click.option('-train_data', default='chembl_train.csv')
@click.option('-test_data', default='chembl_test.csv')
@click.option('-dataset', default='all')
@click.option('-withdrawn_col', default='withdrawn')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
def main(train_data, test_data, dataset, withdrawn_col, batch_size, gpu):
    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data))[['smiles', withdrawn_col]]
        data = data.sample(frac=1, random_state=0)

    else:
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.loc[(data['dataset'] == dataset) |
                        (data['dataset'] == 'both') |
                        (data['dataset'] == 'withdrawn')][['smiles', withdrawn_col]]
        data = data.sample(frac=1, random_state=0)

    test = pd.read_csv(root / 'data/{}'.format(test_data))[['smiles', withdrawn_col]]
    test = np.array(test)

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

    """
    train, calibration = train_test_split(data, test_size=0.15, stratify=data[withdrawn_col], shuffle=True,
                                          random_state=conf.seed)

    y_calibration = np.array(calibration[withdrawn_col])
    calibration = np.array(calibration)
    """

    train, val = train_test_split(data, test_size=0.15, stratify=data[withdrawn_col], shuffle=True,
                                  random_state=conf.seed)
    pos_weight = torch.Tensor([(len(train) / len(train.loc[train['withdrawn'] == 1]))])
    conf.pos_weight = pos_weight

    train = np.array(train)
    y_val = np.array(val[withdrawn_col])
    val = np.array(val)


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

    nonconform_adapter = ClassifierAdapter(trainer, model, conf.batch_size)
    nc = ClassifierNc(nonconform_adapter, MarginErrFunc())
    icp = IcpClassifier(nc, condition=lambda x: x[1])
    ccp = CrossConformalClassifier(icp, CrossSampler())

    ccp.fit(train, val)
    ccp.calibrate(val, y_val)
    prediction = icp.predict_conf(test)
    significance = icp.predict(test, significance=0.1)

    significance = pd.DataFrame(significance)
    significance.rename(columns={0: 'class_0', 1: 'class_1'}, inplace=True)


    test_data_list = []
    y_test = []
    for row in test:
        test_data_list.append(smiles2graph(row[0], row[1]))
        y_test.append(row[1])
    test_loader = DataLoader(test_data_list, num_workers=0, batch_size=conf.batch_size)

    model_outputs = []
    for batch in test_loader:
        output = model.forward(batch).cpu().detach().numpy()
        model_outputs.append(output)

    model_outputs = np.array([cell for row in model_outputs for cell in row])  # flatten the list
    prediction = pd.DataFrame(prediction)
    prediction['y_true'] = y_test
    prediction['model_outputs'] = model_outputs
    prediction['class_0'] = significance['class_0']
    prediction['class_1'] = significance['class_1']


    results_path = Path(root / "results")
    prediction.to_csv(results_path / "mondrian_prediction.csv")

    error_rate = class_mean_errors(icp.predict(test), np.array(y_test), significance=0.1)
    print('Error rate: {}'.format(error_rate))


if __name__ == '__main__':
    main()
