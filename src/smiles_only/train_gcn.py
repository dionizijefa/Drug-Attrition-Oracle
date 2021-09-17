import shutil
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
import click
from EGConv_lightning import Conf, EGConvNet
from scaffold_cross_val import cross_val, create_loader, calibrate, conformal_prediction
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.train_test_split import train_test_split

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-bayes_opt', default=False)
@click.option('-conformal', default=False)
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=16)
@click.option('-epochs', default=100)
@click.option('-gpu', default=1)
@click.option('-production', default=False)
@click.option('-seed', default=0)
def main(
        train_data,
        test_data,
        withdrawn_col,
        bayes_opt,
        conformal,
        batch_size,
        epochs,
        gpu,
        production,
        seed
):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    test_data = pd.read_csv(root / test_data)[['standardized_smiles', withdrawn_col, 'scaffolds']]
    test_loader = create_loader(test_data, withdrawn_col, batch_size)

    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dimensions = [dim_1, dim_2, dim_3, dim_4]

    @use_named_args(dimensions=dimensions)
    def inverse_ap(hidden_channels, num_layers, num_heads, num_bases):
        fold_ap = []
        fold_auroc = []
        conf = Conf(
            batch_size=batch_size,
            reduce_lr=True,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bases=num_bases,
        )

        for fold in cross_val(data, withdrawn_col, batch_size, seed):
            model = EGConvNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
            )

            logger = TensorBoardLogger(
                conf.save_dir,
                name='egconv_bayes_opt',
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

            print("Starting training")
            trainer = pl.Trainer(
                max_epochs=epochs,
                gpus=[gpu],  # [0]  # load from checkpoint instead of resume
                weights_summary='top',
                callbacks=[early_stop_callback, model_checkpoint],
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0
            )

            train_loader, val_loader, test_loader = fold
            trainer.fit(model, train_loader, val_loader)
            results = trainer.test(model, test_loader)
            test_ap = round(results[0]['test_ap'], 3)
            test_auc = round(results[0]['test_auc'], 3)
            fold_ap.append(test_ap)
            fold_auroc.append(test_auc)

        print('Average AP across folds: {}'.format(np.mean(fold_ap)))
        print('Average AUC across folds: {}'.format(np.mean(fold_auroc)))
        print('\n')

        for i, result in enumerate(fold_ap):
            print('AP for fold {}= {}'.format(i, result))

        for i, result in enumerate(fold_auroc):
            print('AUC for fold {}= {}'.format(i, result))

        return 1 / np.mean(fold_ap)

    if bayes_opt:
        start = time()
        res = gp_minimize(inverse_ap,  # minimize the inverse of average precision
                          dimensions=dimensions,  # hyperparams
                          acq_func="EI",  # the acquisition function
                          n_calls=25,  # the number of evaluations of f
                          n_random_starts=5,  # the number of random initialization points
                          random_state=seed)  # the random seed
        end = time()
        elapsed = (end-start) / 3600

        print('Value of the minimum: {}'.format(res.fun))
        print('Res space: {}'.format(res.x))
        print('Time elapsed in hrs: {}'.format(elapsed))

        if production:
            results_path = Path(root / 'production')
            if not results_path.exists():
                results_path.mkdir(exist_ok=True, parents=True)
                with open(results_path / "bayes_opt.txt", "w") as file:
                    file.write("Bayes opt - EGConv")
                    file.write("\n")

                with open(results_path / "bayes_opt.txt", "a") as file:
                    print('Maximum AP: {}'.format(res.fun), file=file)
                    print('Res space: {}'.format(res.space), file=file)
                    file.write("\n")
                    file.write("\n")

        else:
            results_path = Path(root / 'bayes_opt')
            if not results_path.exists():
                results_path.mkdir(exist_ok=True, parents=True)
                with open(results_path / "bayes_opt.txt", "w") as file:
                    file.write("Bayes opt - EGConv")
                    file.write("\n")

                with open(results_path / "bayes_opt.txt", "a") as file:
                    print('Maximum AP: {}'.format(res.fun), file=file)
                    print('Res space: {}'.format(res.space), file=file)
                    file.write("\n")
                    file.write("\n")

    if conformal:
        if bayes_opt:
            # use results from the previous step
            conf = Conf(
                batch_size=batch_size,
                reduce_lr=True,
                hidden_channels=res.x[0],
                num_layers=res.x[1],
                num_heads=res.x[2],
                num_bases=res.x[3],
            )
        else:
            #use defaults
            conf = Conf(
                batch_size=batch_size,
                reduce_lr=True,
                hidden_channels=512,
                num_layers=7,
                num_heads=4,
                num_bases=6,
            )

        if production:
            conf.save_dir = '{}/production/'.format(root)

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
