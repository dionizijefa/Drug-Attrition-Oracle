import sys
sys.path.append('../..')
import shutil
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import click
from EGConv_lightning import Conf, EGConvNet
from src.utils.data_func import cross_val, create_loader, calibrate, conformal_prediction, smiles2graph
from src.utils.metrics import table_metrics, optimal_threshold_f1, metrics_at_significance
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from torch import Tensor, cat

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=32)
@click.option('-epochs', default=100)
@click.option('-gpu', default=1)
@click.option('-production', default=False)
@click.option('-hidden', default=256)
@click.option('-layers', default=3)
@click.option('-heads', default=4)
@click.option('-bases', default=7)
@click.option('-seed', default=0)
def main(
        train_data,
        test_data,
        withdrawn_col,
        batch_size,
        epochs,
        gpu,
        production,
        hidden,
        layers,
        heads,
        bases,
        seed,
):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds',
                                                             'chembl_id']]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    test_data = pd.read_csv(root / 'data/{}'.format(test_data))[['standardized_smiles', withdrawn_col, 'scaffolds',
                                                                 'chembl_id']]
    test_loader = create_loader(test_data, withdrawn_col, batch_size)

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=hidden,
        num_layers=layers,
        num_heads=heads,
        num_bases=bases,
        seed=seed,
        lr=0.000169,
    )

    cross_approved_p = []
    cross_withdrawn_p = []
    cross_probabilities = []
    threshold_optimal_f1 = []

    for count, fold in enumerate(cross_val(data, withdrawn_col, batch_size, seed, n_splits=6)):
        model = EGConvNet(
            conf.to_hparams(),
            reduce_lr=conf.reduce_lr,
        )

        logger = TensorBoardLogger(
            conf.save_dir,
            name='egconv_conformal',
            version='{}'.format(str(int(time()))),
        )

        model_checkpoint = ModelCheckpoint(
            dirpath=(logger.log_dir + '/checkpoint/'),
            monitor='val_auc_epoch',
            mode='max',
            save_top_k=1,
        )

        early_stop_callback = EarlyStopping(monitor='val_auc_epoch',
                                            min_delta=0.00,
                                            mode='max',
                                            patience=10,
                                            verbose=False)

        # Copy this script and all files used in training
        log_dir = Path(logger.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(Path(__file__), log_dir)

        print("Starting training")
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=[gpu],  # [0]  # load from checkpoint instead of resume
            weights_summary='top',
            callbacks=[early_stop_callback, model_checkpoint],
            deterministic=True,
            auto_lr_find=False,
            num_sanity_val_steps=0,
            logger=logger,
        )

        train_loader, val_loader, calib_loader = fold
        trainer.fit(model, train_loader, val_loader)

        #calibrate the model
        model.eval()
        approved_calibration, withdrawn_calibration = calibrate(model, calib_loader)
        p_values_approved, p_values_withdrawn, test_probabilities = conformal_prediction(
            test_loader,
            model,
            approved_calibration,
            withdrawn_calibration
        )
        cross_approved_p.append(p_values_approved)
        cross_withdrawn_p.append(p_values_withdrawn)
        cross_probabilities.append(test_probabilities)
        threshold_calib = DataLoader(train_loader.dataset, batch_size, num_workers=0)
        training_threshold = optimal_threshold_f1(model, threshold_calib)
        val_threshold = optimal_threshold_f1(model, val_loader)
        threshold_optimal_f1.append(np.mean([training_threshold, val_threshold]))

    results_path = Path(root / 'cross_conformal')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)

    mean_p_approved = np.mean(np.array(cross_approved_p), axis=0)
    mean_p_withdrawn = np.mean(np.array(cross_withdrawn_p), axis=0)
    mean_probabilities = np.mean(np.array(cross_probabilities), axis=0)

    conformal_output = test_data[["chembl_id", withdrawn_col]]
    conformal_output['p_approved'] = mean_p_approved
    conformal_output['p_withdrawn'] = mean_p_withdrawn
    conformal_output['probabilities'] = mean_probabilities

    optimal_threshold = np.mean(threshold_optimal_f1)

    conformal_output.to_csv(results_path / 'test_set_outputs.csv')
    results = table_metrics(conformal_output, withdrawn_col, optimal_threshold)
    results_at_significance = metrics_at_significance(conformal_output, withdrawn_col, optimal_threshold)
    results_at_significance.to_csv(results_path / 'results_at_significance.csv')
    results.to_csv(results_path / 'results.csv')

    if production:
        conf.save_dir = '{}/production/'.format(root)
        data = pd.concat([data, test_data])

        train, calib = train_test_split(data, test_size=0.15,
                                       stratify=data[withdrawn_col], shuffle=True,
                                       random_state=seed)
        train, val = train_test_split(train, test_size=0.15,
                                       stratify=train[withdrawn_col], shuffle=True,
                                       random_state=seed)

        calib_data_list = []
        for index, row in calib.iterrows():
            calib_data_list.append(smiles2graph(row, withdrawn_col))
        calib_loader = DataLoader(calib_data_list, num_workers=0, batch_size=batch_size)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph(row, withdrawn_col))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, withdrawn_col))

        withdrawn = train[withdrawn_col].value_counts()[1]
        approved = train[withdrawn_col].value_counts()[0]
        class_sample_count = [approved, withdrawn]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train[withdrawn_col].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler)

        model = EGConvNet(
            conf.to_hparams(),
            reduce_lr=conf.reduce_lr,
        )

        logger = TensorBoardLogger(
            conf.save_dir,
            name='egconv_production',
            version='production',
        )

        model_checkpoint = ModelCheckpoint(
            dirpath=(logger.log_dir + '/checkpoint/'),
            monitor='val_ap_epoch',
            mode='max',
            save_top_k=1,
        )

        early_stop_callback = EarlyStopping(monitor='val_ap_epoch',
                                            min_delta=0.00,
                                            mode='max',
                                            patience=10,
                                            verbose=False)

        # Copy this script and all files used in training
        log_dir = Path(logger.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(Path(__file__), log_dir)

        print("Starting training")
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=[gpu],  # [0]  # load from checkpoint instead of resume
            weights_summary='top',
            callbacks=[early_stop_callback, model_checkpoint],
            deterministic=True,
            auto_lr_find=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_loader, val_loader)
        model.eval()
        approved_calibration, withdrawn_calibration = calibrate(model, calib_loader)

        threshold_calib = DataLoader(train_loader.dataset, batch_size, num_workers=0)
        training_threshold = optimal_threshold_f1(model, threshold_calib)
        val_threshold = optimal_threshold_f1(model, val_loader)
        threshold_optimal_f1 = np.mean([training_threshold, val_threshold])
        with open(conf.save_dir+'optimal_threshold', 'w') as file:
            file.write('Optimal threshold: {}'.format(threshold_optimal_f1))

        calib_targets = []
        for i in calib_loader:
            calib_targets.append(i.y)
        calib_targets = np.array(cat(calib_targets).detach().cpu().numpy().flatten())

        np.savetxt(conf.save_dir+'/egconv_production/approved_calibration.csv', approved_calibration)
        np.savetxt(conf.save_dir+'/egconv_production/withdrawn_calibration.csv', withdrawn_calibration)
        np.savetxt(conf.save_dir+'/egconv_production/calib_classes.csv', calib_targets)


if __name__ == '__main__':
    main()
