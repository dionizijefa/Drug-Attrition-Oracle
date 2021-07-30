import dataclasses
import shutil
from abc import ABC
from pathlib import Path
from pprint import pformat
from time import time
from typing import Dict, Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import average_precision, auroc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from featurization.data_utils import load_data_from_df, construct_loader
from transformer import Network
from sklearn.metrics import f1_score
from dataset import MolDataset, mol_collate_func

root = Path(__file__).resolve().parents[2].absolute()


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
        self.model = Network()
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
        adjacency_matrix, node_features, distance_matrix, descriptors, labels = batch[0]
        names = batch[1]
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_hat = self.forward(node_features, batch_mask, adjacency_matrix, distance_matrix, descriptors).squeeze(-1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat, labels)

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

    def train_dataloader(self):
        dataset = MolDataset('/home/dfa/AI4EU-group/AI4EU-descriptors/train_bono3.pt')
        class_sample_count = [1565, 169]  # 169, 1565
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weights = weights[dataset.labels]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        return DataLoader(dataset, self.hparams.batch_size, collate_fn=mol_collate_func, sampler=sampler)

    def val_dataloader(self):
        dataset = MolDataset('/home/dfa/AI4EU-group/AI4EU-descriptors/val_bono3.pt')
        return DataLoader(dataset, self.hparams.batch_size, collate_fn=mol_collate_func)

    def test_dataloader(self):
        dataset = MolDataset('/home/dfa/AI4EU-group/AI4EU-descriptors/test_bono3.pt')
        return DataLoader(dataset, self.hparams.batch_size, collate_fn=mol_collate_func)

def main():
    conf = Conf(
        lr=1e-5,
        batch_size=4,
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
                                        patience=25,
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
        num_sanity_val_steps=0,
    )
    trainer.fit(model)
    results = trainer.test()
    results_path = Path(root / "results")
    test_ap = round(results[0]['test_ap'], 3)
    test_auc = round(results[0]['test_auc'], 3)
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "classification_results.txt", "w") as file:
            file.write("Classification results")
            file.write("\n")

    results = {'Test AP': test_ap,
               'Test AUC-ROC': test_auc,
               }
    version = {'version': logger.version}
    results = {logger.name: [results, version]}
    with open(results_path / "classification_results.txt", "a") as file:
        print(results, file=file)
        file.write("\n")

if __name__ == '__main__':
    main()
