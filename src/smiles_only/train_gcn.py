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
from scaffold_cross_val import scaffold_cross_val
from callbacks import early_stop_callback

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/alldata_min_phase_4_train.csv')
@click.option('-dataset', default='all')
@click.option('-bayes_opt', default=False)
@click.option('-conformal', default=False)
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
@click.option('-seed', default=0)
def main(train_data, dataset, withdrawn_col, bayes_opt, conformal, batch_size, gpu, seed):
    if dataset == 'all':
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

            print("Starting training")
            trainer = pl.Trainer(
                max_epochs=conf.epochs,
                gpus=[gpu],  # [0]  # load from checkpoint instead of resume
                weights_summary='top',
                callbacks=[early_stop_callback],
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
        print('Average AUC across folds: {}'.format(np.mean(fold_auc_roc)))
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
            conf = Conf(
                batch_size=batch_size,
                reduce_lr=True,
                hidden_channels=res[0],
                num_layers=res[1],
                num_heads=res[2],
                num_bases=res[3],
            )

        else:
            conf = Conf(
                batch_size=batch_size,
                reduce_lr=True,
                hidden_channels=256,
                num_layers=3,
                num_heads=2,
                num_bases=2,
            )

        for fold in scaffold_cross_val(data, withdrawn_col, batch_size, seed):
            model = EGConvNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
            )
            print("Starting training")
            trainer = pl.Trainer(
                max_epochs=1,
                gpus=[gpu],  # [0]  # load from checkpoint instead of resume
                weights_summary='top',
                callbacks=[early_stop_callback],
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0
            )

            train_loader, val_loader, test_loader = fold
            trainer.fit(model, train_loader, val_loader)

            train_probabilities = []
            for i in train_loader:
                train_probabilities.append(model.forward(i).detach().cpu().numpy())
            train_probabilities = [prediction for sublist in train_probabilities for prediction in sublist]
            train_probabilities = train_probabilities.sort()
            print(train_probabilities)
            print(error)


if __name__ == '__main__':
    main()
