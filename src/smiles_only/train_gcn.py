import shutil
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import click
from EGConv_lightning import Conf, EGConvNet
from data_func import cross_val, create_loader, calibrate, conformal_prediction
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=32)
@click.option('-epochs', default=100)
@click.option('-gpu', default=1)
@click.option('-production', default=False)
@click.option('-hidden', default=1024)
@click.option('-layers', default=4)
@click.option('-heads', default=4)
@click.option('-bases', default=3)
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
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    test_data = pd.read_csv(root / test_data)[['standardized_smiles', withdrawn_col, 'scaffolds']]
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

    for fold in cross_val(data, withdrawn_col, batch_size, seed):
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

        train_loader, val_loader, calib_loader = fold
        trainer.fit(model, train_loader, val_loader)

        #calibrate the model
        approved_calibration, withdrawn_calibration = calibrate(model, calib_loader)
        p_values_approved, p_values_withdrawn, test_probabilities = conformal_prediction(
            test_loader,
            model,
            approved_calibration,
            withdrawn_calibration
        )
        cross_approved_p.append(p_values_approved)
        cross_withdrawn_p.append(p_values_withdrawn)

        median_p_approved = np.median(np.array(cross_approved_p), axis=0)
        median_p_withdrawn = np.median(np.array(cross_withdrawn_p), axis=0)
        mean_p_approved = np.mean(np.array(cross_approved_p), axis=0)
        mean_p_withdrawn = np.mean(np.array(cross_withdrawn_p), axis=0)
        median_probabilities = np.median(np.array(cross_probabilities), axis=0)
        mean_probabilities = np.median(np.array(cross_probabilities), axis=0)

        conformal_output = test_data[["chembl_id", withdrawn_col]]
        conformal_output['median_p_approved'] = median_p_approved
        conformal_output['median_p_withdrawn'] = median_p_withdrawn
        conformal_output['mean_p_approved'] = mean_p_approved
        conformal_output['mean_p_withdrawn'] = mean_p_withdrawn
        conformal_output['median_probabilities'] = median_probabilities
        conformal_output['mean_probabilities'] = mean_probabilities

        results_path = Path(root / 'cross_conformal')
        if not results_path.exists():
            results_path.mkdir(exist_ok=True, parents=True)

        conformal_output.to_csv(results_path / 'test_set_conformal.csv')

    if production:
        conf.save_dir = '{}/production/'.format(root)
        data = pd.concat([data, test_data])
        approved_calibrations = []
        withdrawn_calibrations = []

        for count, fold in enumerate(cross_val(data, withdrawn_col, batch_size, seed)):
            model = EGConvNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
            )

            logger = TensorBoardLogger(
                conf.save_dir,
                name='egconv_production',
                version='{}'.format(count),
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

            train_loader, val_loader, calib_loader = fold
            trainer.fit(model, train_loader, val_loader)
            approved_calibration, withdrawn_calibration = calibrate(model, calib_loader)
            approved_calibrations.append(approved_calibration)
            withdrawn_calibrations.append(withdrawn_calibration)

        np.savetxt('approved_calibration.txt', np.array(approved_calibration), delimiter=',')
        np.savetxt('withdrawn_calibration.txt', np.array(withdrawn_calibration), delimiter=',')


if __name__ == '__main__':
    main()
