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
from scaffold_cross_val import scaffold_cross_val, create_loader
from torch import cat
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-bayes_opt', default=False)
@click.option('-conformal', default=False)
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, bayes_opt, conformal, batch_size, gpu, seed):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dimensions = [dim_1, dim_2, dim_3, dim_4]

    @use_named_args(dimensions=dimensions)
    def inverse_ap(hidden_channels, num_layers, num_heads, num_bases):
        fold_ap = []
        fold_auroc = []
        conf=Conf(
            batch_size=batch_size,
            reduce_lr=True,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bases=num_bases,
        )

        for fold in scaffold_cross_val(data, withdrawn_col, batch_size, seed):
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
                max_epochs=conf.epochs,
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

        cross_conformal_p_approved = []
        cross_conformal_p_withdrawn = []
        class_probabilities = []

        for fold in scaffold_cross_val(data, withdrawn_col, batch_size, seed):
            model = EGConvNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
            )

            logger = TensorBoardLogger(
                conf.save_dir,
                name='egconv_inference',
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
                max_epochs=100,
                gpus=[gpu],  # [0]  # load from checkpoint instead of resume
                weights_summary='top',
                callbacks=[early_stop_callback, model_checkpoint],
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0,
            )

            train_loader, val_loader, calib_loader = fold
            trainer.fit(model, train_loader, val_loader)

            #calibration data
            calib_probabilities = []
            targets = []
            for i in calib_loader:
                calib_probabilities.append(model.forward(i))
                targets.append(i.y)
            calib_probabilities = np.array(cat(calib_probabilities).detach().cpu().numpy().flatten())
            calib_probabilities = 1 / (1 + np.exp(-calib_probabilities))
            targets = np.array(cat(targets).detach().cpu().numpy().flatten())
            calibration_df = pd.DataFrame({'probabilities': calib_probabilities, 'class': targets})
            approved_probabilities = calibration_df.loc[calibration_df['class'] == 0]['probabilities'].values
            approved_probabilities = 1 - approved_probabilities
            withdrawn_probabilities = calibration_df.loc[calibration_df['class'] == 1]['probabilities'].values
            approved_probabilities = np.sort(approved_probabilities)
            withdrawn_probabilities = np.sort(withdrawn_probabilities)

            test = pd.read_csv(root / 'data/{}'.format(test_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
            test_loader = create_loader(test, withdrawn_col, conf.batch_size)
            test_probabilities = []
            test_targets = []
            for i in test_loader:
                test_probabilities.append(model.forward(i))
                test_targets.append(i.y)
            test_probabilities = np.array(cat(test_probabilities).detach().cpu().numpy().flatten())
            test_probabilities = 1 / (1 + np.exp(-test_probabilities))

            p_values_approved = []
            p_values_withdrawn = []
            for i in test_probabilities:
                p_values_approved.append(
                    (np.searchsorted(approved_probabilities, i) / (len(approved_probabilities) + 1))
                )
                p_values_withdrawn.append(
                    (np.searchsorted(withdrawn_probabilities, i) / (len(withdrawn_probabilities) + 1))
                )

            class_probabilities.append(test_probabilities)
            cross_conformal_p_withdrawn.append(np.array(p_values_withdrawn))
            cross_conformal_p_approved.append(np.array(p_values_approved))

            #reset model
            model = None


        results_path = Path(root / 'cross_conformal_inference')
        if not results_path.exists():
            results_path.mkdir(exist_ok=True, parents=True)

        #merge targets and predicted values
        targets = []
        for i in test_loader:
            targets.append(i.y)
        targets = np.array(cat(targets).detach().cpu().numpy().flatten())
        cross_conformal_p_approved = np.mean(np.array(cross_conformal_p_approved), axis=0)
        cross_conformal_p_withdrawn = np.mean(np.array(cross_conformal_p_withdrawn), axis=0)
        probabilities = np.mean(np.array(class_probabilities), axis=0)
        test_df = pd.DataFrame({'probabilities': probabilities, 'class': targets,
                                'p_value_approved': cross_conformal_p_approved,
                                'p_value_withdrawn': cross_conformal_p_withdrawn})
        test_df.to_csv(results_path / 'cross_conformal_predictions.csv')


if __name__ == '__main__':
    main()
