import shutil
import sys
sys.path.append("../..")
from functools import reduce
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, cat
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from src.utils.metrics import optimal_threshold_f1, table_metrics, metrics_at_significance
from pathlib import Path
from time import time, sleep
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import click
from descriptors_lightning import Conf, EGConvNet
from src.utils.data_func import cross_val, create_loader, smiles2graph, calibrate, conformal_prediction
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.descriptors_list import rdkit_descriptors_len, alvadesc_descriptors_len, padel_descriptors_10pct_len
from src.utils.descriptors_list import toxprint_descriptors_10pct_len, feature_selected_len, ozren_selected, \
    adme_japtox_rdkit_len

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-descriptors', default='ozren_selected')
@click.option('-batch_size', default=32)
@click.option('-epochs', default=100)
@click.option('-gpu', default=1)
@click.option('-production', default=False)
@click.option('-hidden', default=1024)
@click.option('-layers', default=4)
@click.option('-heads', default=4)
@click.option('-bases', default=7)
@click.option('-lr', default=0.000344)
@click.option('-seed', default=0)
def main(
        train_data,
        test_data,
        withdrawn_col,
        descriptors,
        batch_size,
        epochs,
        gpu,
        production,
        hidden,
        layers,
        heads,
        bases,
        lr,
        seed,
):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['chembl_id', withdrawn_col, 'standardized_smiles']]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    test_data = pd.read_csv(root / 'data/{}'.format(test_data))[['chembl_id', withdrawn_col, 'standardized_smiles']]

    if withdrawn_col == 'wd_withdrawn':
        data['wd_withdrawn'] = data['wd_withdrawn'].fillna(0) # withdrawn has only withdrawn mols
        test_data['wd_withdrawn'] = test_data['wd_withdrawn'].fillna(0)
    else:
        data = data.dropna(subset=[withdrawn_col])  # some molecules don't share all labels
        test_data = test_data.dropna(subset=[withdrawn_col])

    if descriptors == 'alvadesc':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/alvadesc_descriptors.csv')
        #descriptors_len = alvadesc_descriptors_len
        descriptors_len = 100

    elif descriptors == 'padel1560':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/padel1560_descriptors.csv')
        descriptors_len = padel_descriptors_10pct_len

    elif descriptors == 'toxprint':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/toxprint_descriptors.csv')
        descriptors_len = toxprint_descriptors_10pct_len

    elif descriptors == 'rdkit':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/rdkit_descriptors.csv')
        descriptors_len = rdkit_descriptors_len

    elif descriptors == 'ozren_selected':
        toxprint = pd.read_csv(root / 'data/processing_pipeline/descriptors/toxprint_descriptors.csv')
        padel = pd.read_csv(root / 'data/processing_pipeline/descriptors/padel1560_descriptors.csv')
        list_of_desc = [toxprint, padel]
        descriptors_df = reduce(lambda left, right: pd.merge(left, right, on=['chembl_id'],
                                                             how='inner', suffixes=[None, "_right"]), list_of_desc)
        descriptors_len = len(ozren_selected)

    elif descriptors == 'adme_japtox_rdkit':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/ADME-JapTox-RDKIT.csv')
        descriptors_len = adme_japtox_rdkit_len

    elif descriptors == 'adme_japtox_rdkit_60':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/ADME-JapTox-RDKIT.csv')
        descriptors_len = 60

    elif descriptors == 'adme_japtox_rdkit_33_140':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/ADME-JapTox-RDKIT.csv')
        descriptors_len = 107

    elif descriptors == 'adme_japtox_rdkit_6':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/ADME-JapTox-RDKIT.csv')
        descriptors_len = 6

    elif descriptors == 'adme_japtox_rdkit_7':
        descriptors_df = pd.read_csv(root / 'data/processing_pipeline/descriptors/ADME-JapTox-RDKIT.csv')
        descriptors_len = 7

    else:
        rdkit = pd.read_csv(root / 'data/processing_pipeline/descriptors/rdkit_descriptors.csv')
        toxprint = pd.read_csv(root / 'data/processing_pipeline/descriptors/toxprint_descriptors.csv')
        alvadesc = pd.read_csv(root / 'data/processing_pipeline/descriptors/alvadesc_descriptors.csv')
        padel = pd.read_csv(root / 'data/processing_pipeline/descriptors/padel1560_descriptors.csv')
        list_of_desc = [rdkit, toxprint, alvadesc, padel]

        descriptors_df = reduce(lambda left, right: pd.merge(left, right, on=['chembl_id'],
                                                        how='inner', suffixes=[None, "_right"]), list_of_desc)
        descriptors_len = feature_selected_len

    data = data.merge(descriptors_df, how='inner', on='chembl_id')
    test_data = test_data.merge(descriptors_df, how='inner', on='chembl_id')
    test_loader = create_loader(test_data, withdrawn_col, batch_size, descriptors=descriptors)

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=hidden,
        num_layers=layers,
        num_heads=heads,
        num_bases=bases,
        lr=lr,
        seed=seed,
    )

    cross_approved_p = []
    cross_withdrawn_p = []
    cross_probabilities = []
    threshold_optimal_f1 = []
    for fold in cross_val(data, withdrawn_col, batch_size, seed, descriptors=descriptors):
        model = EGConvNet(
            conf.to_hparams(),
            reduce_lr=conf.reduce_lr,
            descriptors_len=descriptors_len,
            options='concat_early',
        )

        logger = TensorBoardLogger(
            conf.save_dir,
            name='egconv_descriptors_{}_conformal'.format(descriptors),
            version='{}'.format(str(int(time()))),
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
            logger=logger,
        )

        train_loader, val_loader, calib_loader = fold
        trainer.fit(model, train_loader, val_loader)

        #calibrate the model
        model.eval()
        approved_calibration, withdrawn_calibration = calibrate(model, calib_loader, descriptors=True)
        p_values_approved, p_values_withdrawn, test_probabilities = conformal_prediction(
            test_loader,
            model,
            approved_calibration,
            withdrawn_calibration,
            descriptors=True
        )
        cross_approved_p.append(p_values_approved)
        cross_withdrawn_p.append(p_values_withdrawn)
        cross_probabilities.append(test_probabilities)

        threshold_calib = DataLoader(train_loader.dataset, batch_size, num_workers=0)
        training_threshold = optimal_threshold_f1(model, threshold_calib, descriptors=True)
        val_threshold = optimal_threshold_f1(model, val_loader, descriptors=True)
        threshold_optimal_f1.append(np.mean([training_threshold, val_threshold]))

    mean_p_approved = np.mean(np.array(cross_approved_p), axis=0)
    mean_p_withdrawn = np.mean(np.array(cross_withdrawn_p), axis=0)
    mean_probabilities = np.mean(np.array(cross_probabilities), axis=0)

    conformal_output = test_data[["chembl_id", withdrawn_col]]
    conformal_output['p_approved'] = mean_p_approved
    conformal_output['p_withdrawn'] = mean_p_withdrawn
    conformal_output['probabilities'] = mean_probabilities

    optimal_threshold = np.mean(threshold_optimal_f1)

    results_path = Path(root / 'cross_conformal_descriptors')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)

    conformal_output.to_csv(results_path / '{}_testset_outputs.csv'.format(descriptors))
    results = table_metrics(conformal_output, withdrawn_col, optimal_threshold)
    results_at_significance = metrics_at_significance(conformal_output, withdrawn_col, optimal_threshold)
    results_at_significance.to_csv(results_path / '{}_results_at_significance.csv'.format(descriptors))
    results.to_csv(results_path / '{}_results.csv'.format(descriptors))

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
            calib_data_list.append(smiles2graph(row, withdrawn_col, descriptors=descriptors))
        calib_loader = DataLoader(calib_data_list, num_workers=0, batch_size=batch_size)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph(row, withdrawn_col, descriptors=descriptors))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, withdrawn_col, descriptors=descriptors))

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
            descriptors_len=descriptors_len,
            options='concat_early',
        )

        logger = TensorBoardLogger(
            conf.save_dir,
            name='descriptors_production',
            version='{}'.format(descriptors),
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
        approved_calibration, withdrawn_calibration = calibrate(model, calib_loader, descriptors=True)

        calib_targets = []
        for i in calib_loader:
            calib_targets.append(i.y)
        calib_targets = np.array(cat(calib_targets).detach().cpu().numpy().flatten())

        threshold_calib = DataLoader(train_loader.dataset, batch_size, num_workers=0)
        training_threshold = optimal_threshold_f1(model, threshold_calib, descriptors=True)
        val_threshold = optimal_threshold_f1(model, val_loader, descriptors=True)
        threshold_optimal_f1 = np.mean([training_threshold, val_threshold])
        with open(conf.save_dir+'/descriptors_production/{}_optimal_threshold'.format(descriptors), 'w') as file:
            file.write('Optimal threshold: {}'.format(threshold_optimal_f1))
        np.savetxt(conf.save_dir+'/descriptors_production/{}_approved_calibration.csv'.format(descriptors),
                   approved_calibration)
        np.savetxt(conf.save_dir+'/descriptors_production/{}_withdrawn_calibration.csv'.format(descriptors),
                   withdrawn_calibration)
        np.savetxt(conf.save_dir+'/descriptors_production/{}_calib_classes.csv'.format(descriptors), calib_targets)

if __name__ == '__main__':
    main()
