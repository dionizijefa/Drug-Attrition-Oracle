import shutil
from pathlib import Path
from time import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import umap
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from rdkit import Chem, DataStructs
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
from skopt import gp_minimize
from skopt.space import Categorical, Integer
from skopt.utils import use_named_args
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
import torch
from data_utils import smiles2graph
import click
from EGConv_lightning import Conf, EGConvNet

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/train/alldata_min_phase_4_train.csv')
@click.option('-dataset', default='all')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=16)
@click.option('-gpu', default=1)
@click.option('-seed', default=0)
def main(train_data, dataset, withdrawn_col, batch_size, gpu, seed):
    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
        data = data.sample(frac=1, random_state=seed)  # shuffle

    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dimensions = [dim_1, dim_2, dim_3, dim_4]

    @use_named_args(dimensions=dimensions)
    def maximize_ap(hidden_channels, num_layers, num_heads, num_bases):
        # cross val on unique scaffolds -> test is only on unique scaffolds, val only on unique scaffolds
        # append non-unique scaffolds to train at the end
        scaffolds_df = pd.DataFrame(data['scaffolds'].value_counts())
        unique_scaffolds = list(scaffolds_df.loc[scaffolds_df['scaffolds'] == 1].index)
        data_unique_scaffolds = data.loc[data['scaffolds'].isin(unique_scaffolds)]

        cv_splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        fold_ap = []
        fold_auc_roc = []
        cv_fold = []
        for k, (train_index, test_index) in enumerate(
                cv_splitter.split(data_unique_scaffolds, data_unique_scaffolds[withdrawn_col])
        ):

            conf = Conf(
                batch_size=batch_size,
                reduce_lr=True,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                num_heads=num_heads,
                num_bases=num_bases,
            )

            logger = TensorBoardLogger(
                conf.save_dir,
                name='EGConv',
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

            test = data_unique_scaffolds.iloc[test_index]
            test_data_list = []
            for index, row in test.iterrows():
                test_data_list.append(smiles2graph(row, withdrawn_col))
            test_loader = DataLoader(test_data_list, num_workers=0, batch_size=conf.batch_size)

            train_set = data_unique_scaffolds.iloc[train_index]

            train, val = train_test_split(
                train_set,
                test_size=0.15,
                stratify=train_set[withdrawn_col],
                shuffle=True,
                random_state=seed)

            train = pd.concat([train, data.loc[~data['scaffolds'].isin(unique_scaffolds)]])
            train_data_list = []
            for index, row in train.iterrows():
                train_data_list.append(smiles2graph(row, withdrawn_col))

            # balanced sampling of the minority class
            withdrawn = train[withdrawn_col].value_counts()[1]
            approved = train[withdrawn_col].value_counts()[0]
            class_sample_count = [approved, withdrawn]
            weights = 1 / torch.Tensor(class_sample_count)
            samples_weights = weights[train[withdrawn_col].values]
            sampler = WeightedRandomSampler(samples_weights,
                                            num_samples=len(samples_weights),
                                            replacement=True)
            train_loader = DataLoader(train_data_list, num_workers=0, batch_size=conf.batch_size,
                                      sampler=sampler)

            val_data_list = []
            for index, row in val.iterrows():
                val_data_list.append(smiles2graph(row, withdrawn_col))
            val_loader = DataLoader(val_data_list, num_workers=0, batch_size=conf.batch_size)

            model = EGConvNet(
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
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0
            )

            trainer.fit(model, train_loader, val_loader)
            results = trainer.test(model, test_loader)
            test_ap = round(results[0]['test_ap'], 3)
            test_auc = round(results[0]['test_auc'], 3)
            cv_fold.append(k)

            fold_ap.append(test_ap)
            fold_auc_roc.append(test_auc)

        print('Average AP across folds: {}'.format(np.mean(fold_ap)))
        print('Average AUC across folds: {}'.format(np.mean(fold_auc_roc)))
        print('\n')

        for i, result in enumerate(fold_ap):
            print('AP for fold {}= {}'.format(i, result))

        for i, result in enumerate(fold_auc_roc):
            print('AUC for fold {}= {}'.format(i, result))

        return 1 / np.mean(fold_ap)

    start = time()
    res = gp_minimize(maximize_ap,  # the function to minimize
                      dimensions=dimensions,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=25,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=1234)  # the random seed
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


if __name__ == '__main__':
    main()
