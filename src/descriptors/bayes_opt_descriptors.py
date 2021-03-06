import sys
from functools import reduce

from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.append("../..")
from pathlib import Path
from time import time, sleep
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
import click
from descriptors_lightning import Conf, EGConvNet
from src.utils.data_func import cross_val, create_loader, smiles2graph
from pytorch_lightning.callbacks import EarlyStopping
from src.utils.descriptors_list import rdkit_descriptors_len, alvadesc_descriptors_len, padel_descriptors_10pct_len, \
    ozren_selected
from src.utils.descriptors_list import toxprint_descriptors_10pct_len, feature_selected_len, adme_japtox_rdkit_len

root = Path(__file__).resolve().parents[2].absolute()

@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-descriptors', default='feature_selected')
@click.option('-batch_size', default=32)
@click.option('-epochs', default=100)
@click.option('-gpu', default=1)
@click.option('-seed', default=0)
def main(
        train_data,
        test_data,
        withdrawn_col,
        descriptors,
        batch_size,
        epochs,
        gpu,
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
    outer_test_loader = create_loader(test_data, withdrawn_col, batch_size, descriptors=descriptors)

    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dim_5 = Real(1e-5, 1e-3, name='lr')
    #dim_6 = Categorical(["hidden_descriptors", "concat_descriptors", "average_outputs", "concat_early"], name='options')
    dimensions = [dim_1, dim_2, dim_3, dim_4, dim_5,
                  #dim_6
                  ]

    @use_named_args(dimensions=dimensions)
    def inverse_ap(hidden_channels, num_layers, num_heads, num_bases, lr,
        #options
    ):
        fold_ap = []
        fold_auroc = []
        conf = Conf(
            batch_size=batch_size,
            reduce_lr=True,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bases=num_bases,
            lr=lr,
            seed=seed,
        )

        for fold in cross_val(data, withdrawn_col, batch_size, seed, descriptors=descriptors):
            model = EGConvNet(
                conf.to_hparams(),
                reduce_lr=conf.reduce_lr,
                descriptors_len=descriptors_len,
                options='concat_early',
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
                callbacks=[early_stop_callback],
                logger=False,
                deterministic=True,
                auto_lr_find=False,
                num_sanity_val_steps=0,
                checkpoint_callback=False,
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

    print('Starting Bayesian optimization')
    start = time()
    res = gp_minimize(inverse_ap,  # minimize the inverse of average precision
                      dimensions=dimensions,  # hyperparams
                      acq_func="EI",  # the acquisition function
                      n_calls=30,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=seed)  # the random seed
    end = time()
    elapsed = (end - start) / 3600

    print('Value of the minimum: {}'.format(res.fun))
    print('Res space: {}'.format(res.x))
    print('Time elapsed in hrs: {}'.format(elapsed))
    print('\n')
    print('Testing the optimized model on the test set')

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=res.x[0],
        num_layers=res.x[1],
        num_heads=res.x[2],
        num_bases=res.x[3],
        lr=res.x[4],
        seed=seed
    )

    model = EGConvNet(
        conf.to_hparams(),
        descriptors_len=descriptors_len,
        options='concat_early',
        reduce_lr=conf.reduce_lr,
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
        callbacks=[early_stop_callback],
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0,
        checkpoint_callback=False,
        logger=False
    )
    train, val = train_test_split(data, test_size=0.15, stratify=data[withdrawn_col], random_state=seed)
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
                              sampler=sampler, drop_last=True)

    val_loader = create_loader(val, withdrawn_col, batch_size, descriptors=descriptors)

    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(model, outer_test_loader)
    test_ap = round(results[0]['test_ap'], 3)
    test_auc = round(results[0]['test_auc'], 3)

    print('\n')
    print('AP of the outer test set with optimized parameters: {}'.format(test_ap))
    print('AUC of the outer test set with optimized parameters: {}'.format(test_auc))

    results_path = Path(root / 'bayes_opt')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "bayes_opt_descriptors.txt", "w") as file:
            file.write("Bayes opt - EGConv + Descriptors")
            file.write("\n")

    with open(results_path / "bayes_opt_descriptors.txt", "a") as file:
        print('Target label: {}'.format(withdrawn_col), file=file)
        #print('Option: {}'.format(res.x[5]), file=file)
        print('Option: {}'.format('concat_early'), file=file)
        print('Descriptors: {}'.format(descriptors), file=file)
        print('Hidden: {}'.format(res.x[0]), file=file)
        print('Layers: {}'.format(res.x[1]), file=file)
        print('Heads: {}'.format(res.x[2]), file=file)
        print('Bases: {}'.format(res.x[3]), file=file)
        print('Learning rate: {}'.format(res.x[4]), file=file)
        print('Res space: {}'.format(res.space), file=file)
        print('AP on the outer test: {}'.format(test_ap), file=file)
        print('AUC on the outer test: {}'.format(test_auc), file=file)
        file.write("\n")
        file.write("\n")

if __name__ == '__main__':
    main()
