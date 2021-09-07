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
@click.option('-stratify_chemical_space', default=True)
@click.option('-save_model', default=False)
@click.option('-seed', default=0)
def main(train_data, dataset, withdrawn_col, batch_size, gpu, stratify_chemical_space, seed, save_model):
    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col, 'scaffolds']]
        data = data.sample(frac=1, random_state=seed)  # shuffle

    # cross val on unique scaffolds -> test is only on unique scaffolds, val only on unique scaffolds
    # append non-unique scaffolds to train at the end

    stratify_key = withdrawn_col

    if stratify_chemical_space :
        """ generates KDE on UMAP embeddings to stratify train-test splits """
        print('\n')
        print('Performing UMAP and KDE grid search CV to stratify the chemical space across folds')
        #generate morgan fps first
        data_fps = []
        for i in data['standardized_smiles']:
            mol = Chem.MolFromSmiles(i)
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            array = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            data_fps.append(array)

        umapper = umap.UMAP(
            n_components=2,
            metric='jaccard',
            learning_rate=0.5,
            low_memory=True,
            transform_seed=42,
            random_state=42,
        )
        umap_embeddings = umapper.fit_transform(data_fps)
        data['umap_embeddings'] = umap_embeddings
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        withdrawn_grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
        approved_grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
        withdrawn_grid.fit(
            list(data.loc[data[withdrawn_col] == 1]['umap_embeddings'])
        )
        approved_grid.fit(
            list(data.loc[data[withdrawn_col] == 0]['umap_embeddings'])
        )
        data['withdrawn_kde_prob'] = np.exp(withdrawn_grid.best_estimator_.score_samples(umap_embeddings))
        data['approved_kde_prob'] = np.exp(approved_grid.best_estimator_.score_samples(umap_embeddings))

        """
        first_quartile = data['kde_prob'].describe()['25%']
        second_quartile = data['kde_prob'].describe()['50%']
        third_quartile = data['kde_prob'].describe()['75%']
        data['kde_quartile'] = 'first'
        data.loc[data['kde_prob'] < first_quartile, 'kde_quartile'] = 'first'
        data.loc[(data['kde_prob'] > first_quartile) &
                 (data['kde_prob'] <= second_quartile), 'kde_quartile'] = 'second'
        data.loc[(data['kde_prob'] > second_quartile) &
                 (data['kde_prob'] <= third_quartile), 'kde_quartile'] = 'third'
        data.loc[data['kde_prob'] > third_quartile, 'kde_quartile'] = 'fourth'
        data['stratify_label'] = data['wd_consensus_1'].astype(str) + data['kde_quartile']
        stratify_key = 'stratify_label'
        """

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
    predictions_densities = []

    for k, (train_index, test_index) in enumerate(
            cv_splitter.split(data_unique_scaffolds, data_unique_scaffolds[stratify_key])
    ):

        conf = Conf(
            lr=1e-4,
            batch_size=batch_size,
            epochs=100,
            reduce_lr=True,
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
            stratify=train_set[stratify_key],
            shuffle=True,
            random_state=seed)
        # append common scaffolds to train
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

        checkpoint = ModelCheckpoint(
                dirpath=(logger.log_dir + '/checkpoint/'),
                monitor='val_ap_epoch',
                mode='max',
                save_top_k=1,
        )

        print("Starting training")
        trainer = pl.Trainer(
            max_epochs=conf.epochs,
            gpus=[gpu],  # [0]
            logger=logger,  # load from checkpoint instead of resume
            weights_summary='top',
            callbacks=[early_stop_callback] if save_model else [early_stop_callback, checkpoint],
            deterministic=True,
            auto_lr_find=False,
            num_sanity_val_steps=0
        )

        trainer.fit(model, train_loader, val_loader)
        results = trainer.test(model, test_loader)
        results_path = Path(root / "results")
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
        cv_fold.append(k)

        fold_ap.append(test_ap)
        fold_auc_roc.append(test_auc)

        if not results_path.exists():
            results_path.mkdir(exist_ok=True, parents=True)
            with open(results_path / "classification_results.txt", "w") as file:
                file.write("Classification results")
                file.write("\n")

        results = {'Test AP': test_ap,
                   'Test AUC-ROC': test_auc,
                   'CV_fold': cv_fold}
        version = {'version': logger.version}
        results = {logger.name: [results, version]}
        with open(results_path / "classification_results.txt", "a") as file:
            print(results, file=file)
            file.write("\n")

        predictions = []
        for i in test_loader:
            predictions.append(model.forward(i).detach().cpu().numpy())
        predictions = [prediction for sublist in predictions for prediction in sublist]
        test['model_outputs'] = predictions
        predictions_densities.append(test[['model_outputs',
                                           'approved_kde_prob',
                                           'withdrawn_kde_prob',
                                           withdrawn_col]])

    print('Average AP across folds: {}'.format(np.mean(fold_ap)))
    print('Average AUC across folds: {}'.format(np.mean(fold_auc_roc)))
    print('\n')

    for i, result in enumerate(fold_ap):
        print('AP for fold {}= {}'.format(i, result))

    for i, result in enumerate(fold_auc_roc):
        print('AUC for fold {}= {}'.format(i, result))

    output_probs = pd.concat(predictions_densities)
    output_probs.to_csv(results_path / "output_probs.csv")

    results_df = pd.DataFrame({'CV_fold': cv_fold, 'AP': fold_ap, 'AUC': fold_auc_roc}).to_csv(
        results_path / "{}_metrics.csv".format(logger.version))


if __name__ == '__main__':
    main()
