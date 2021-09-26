import sys
sys.path.append("..")
from pathlib import Path
from time import time
import re
import click
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
from src.utils.descriptors_list import padel_descriptors_10pct, rdkit_descriptors, alvadesc_descriptors, \
    toxprint_descriptors_10pct

root = Path(__file__).resolve().parents[1].absolute()

@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-seed', default=0)
def main(
        train_data,
        withdrawn_col,
        seed,
):
    toxprint = pd.read_csv(root / 'data/processing_pipeline/descriptors/toxprint_descriptors.csv')
    padel = pd.read_csv(root / 'data/processing_pipeline/descriptors/padel1560_descriptors.csv', index_col=0)
    alvadesc = pd.read_csv(root / 'data/processing_pipeline/descriptors/alvadesc_descriptors.csv')
    rdkit = pd.read_csv(root / 'data/processing_pipeline/descriptors/rdkit_descriptors.csv')

    descriptors = [toxprint, padel, alvadesc, rdkit]

    # keep original column names later for feature selection
    toxprint_original = toxprint.drop(columns='chembl_id')[toxprint_descriptors_10pct].columns
    padel_original = padel.drop(columns='chembl_id')[padel_descriptors_10pct].columns
    alvadesc_original = alvadesc.drop(columns='chembl_id')[alvadesc_descriptors].columns
    rdkit_original = rdkit.drop(columns='chembl_id')[rdkit_descriptors].columns

    # Take only columns which are nonzero in more that 10% of the molecules (padel and toxprints)
    nonzero = [toxprint_descriptors_10pct, padel_descriptors_10pct, alvadesc_descriptors, rdkit_descriptors]
    for i in range(len(descriptors)):
        chembl_id = descriptors[i]['chembl_id'] #don't forget chembl_id
        descriptors[i] = descriptors[i][nonzero[i]]
        descriptors[i]['chembl_id'] = chembl_id
    columns = [toxprint_original, padel_original, alvadesc_original, rdkit_original]

    # make column names work with XGB
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    for i in descriptors:
        i.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  i.columns.values]

    data = pd.read_csv(root / 'data/{}'.format(train_data))[['chembl_id', withdrawn_col]]
    data = data.sample(frac=1, random_state=seed)  # shuffle

    if withdrawn_col == 'wd_withdrawn':
        data['wd_withdrawn'] = data['wd_withdrawn'].fillna(0) # withdrawn has only withdrawn mols
    else:
        data = data.dropna(subset=[withdrawn_col])  # some molecules don't share all labels

    def inverse_ap(dimensions):
        print('Starting XGBoost bayes-opt on the training set')
        splitter = StratifiedKFold(n_splits=5)
        fold_ap = []
        for (train_index, test_index) in splitter.split(descriptors_X, descriptors_y):
            train_X = descriptors_X.iloc[train_index]
            train_y = descriptors_y.iloc[train_index]
            test_X = descriptors_X.iloc[test_index]
            test_y = descriptors_y.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.15, stratify=train_y)
            eval_set = [(X_val, y_val)]

            zeros = len(np.where(train_y == 0)[0])
            ones = len(np.where(train_y == 1)[0])
            weight = zeros / ones

            model = XGBClassifier(
                max_depth=dimensions[0],
                learning_rate=dimensions[1],
                n_estimators=dimensions[2],
                gamma=dimensions[3],
                min_child_weight=dimensions[4],
                max_delta_step=dimensions[5],
                subsample=dimensions[6],
                colsample_bytree=dimensions[7],
                scale_pos_weight=weight)
            model.fit(X_train, y_train,
                      eval_metric="logloss", eval_set=eval_set,
                      verbose=False, early_stopping_rounds=10)
            y_pred = model.predict(test_X)
            ap = average_precision_score(test_y, y_pred)
            fold_ap.append(ap)

        print('Average AP across folds: {}'.format(np.mean(fold_ap)))
        print('\n')

        for i, result in enumerate(fold_ap):
            print('AP for fold {}= {}'.format(i, result))

        return 1/np.mean(fold_ap)

    res_x = []
    for i in range(len(descriptors)):
        # Bayesian optimization of parameters
        descriptors_set = descriptors[i]
        descriptors_set = descriptors_set.merge(data, how='inner', on='chembl_id')
        descriptors_y = descriptors_set[withdrawn_col]
        descriptors_X = descriptors_set.drop(columns=['chembl_id', withdrawn_col])
        max_depth = Integer(5, 20, name='max_depth')
        learning_rate = Real(0.01, 0.3, name='learning_rate')
        n_estimators = Integer(50, 1000, name='n_estimators')
        gamma = Real(0.01, 1., name='gamma')
        min_child_weight = Integer(2, 10, name='min_child_weight')
        max_delta_step = Real(0.01, 0.1, name='max_delta_step')
        subsample = Real(0.6, 0.9, name='subsample')
        colsample_by_tree = Real(0.5, 0.99, name='colsample_by_tree')
        dimensions = [max_depth, learning_rate, n_estimators, gamma, min_child_weight,
                      max_delta_step, subsample, colsample_by_tree]

        start = time()
        res = gp_minimize(inverse_ap,  # minimize the inverse of average precision
                          dimensions=dimensions,  # hyperparams
                          acq_func="EI",  # the acquisition function
                          n_calls=40,  # the number of evaluations of f
                          n_random_starts=10,  # the number of random initialization points
                          random_state=seed)  # the random seed
        end = time()
        elapsed = (end - start) / 3600

        res_x.append(res.x)
        print('Value of the minimum: {}'.format(res.fun))
        print('Res x: {}'.format(res.x))
        print('Time elapsed in hrs: {}'.format(elapsed))
        print('\n')

        results_path = Path(root / 'bayes_opt')

        if not (results_path / "bayes_opt_XGBoost.txt").exists():
            with open(results_path / "bayes_opt_XGBoost.txt", "w") as file:
                file.write("Bayes opt - XGboost with descriptors")
                file.write("\n")
                
        with open(results_path / "bayes_opt_XGBoost.txt", "a") as file:
            if i == 0:
                print('Descriptors: Toxprint', file=file)
            elif i == 1:
                print('Descriptors: Padel', file=file)
            elif i == 2:
                print('Descriptors: Alvadesc', file=file)
            else:
                print('Descriptors: Rdkit', file=file)
                
            print('Target label: {}'.format(withdrawn_col), file=file)
            print('Max depth: {}'.format(res.x[0]), file=file)
            print('Learning rate: {}'.format(res.x[1]), file=file)
            print('N estimators: {}'.format(res.x[2]), file=file)
            print('Gamma: {}'.format(res.x[3]), file=file)
            print('Min child weight: {}'.format(res.x[4], file=file))
            print('Max delta step: {}'.format(res.x[5], file=file))
            print('Subsample: {}'.format(res.x[6], file=file))
            print('Colsample by tree: {}'.format(res.x[7], file=file))
            print('Res space: {}'.format(res.space), file=file)
            file.write("\n")
            file.write("\n")

    """Now do feature selection on the optimized model"""
    results_path = root / 'feature_selection'
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "Feature_selection_descriptors.txt", "w") as file:
            file.write("XGBoost - Feature importance by gain")
            file.write("\n")

    for i in range(len(descriptors)):
        # Bayesian optimization of parameters
        descriptors_set = descriptors[i]
        descriptors_set = descriptors_set.merge(data, how='inner', on='chembl_id')
        descriptors_y = descriptors_set[withdrawn_col]
        descriptors_X = descriptors_set.drop(columns=['chembl_id', withdrawn_col])
        splitter = StratifiedKFold(n_splits=5)

        descriptors_across_fold = []
        for (train_index, test_index) in splitter.split(descriptors_X, descriptors_y):
            train_X = descriptors_X.iloc[train_index]
            train_y = descriptors_y.iloc[train_index]
            test_X = descriptors_X.iloc[test_index]
            test_y = descriptors_y.iloc[test_index]
            eval_set = [(test_X, test_y)]

            # class weights
            zeros = len(np.where(train_y == 0)[0])
            ones = len(np.where(train_y == 1)[0])
            weight = zeros / ones

            model = XGBClassifier(
                max_depth=res_x[i][0],
                learning_rate=res_x[i][1],
                n_estimators=res_x[i][2],
                gamma=res_x[i][3],
                min_child_weight=res_x[i][4],
                max_delta_step=res_x[i][5],
                subsample=res_x[i][6],
                colsample_bytree=res_x[i][7],
                scale_pos_weight=weight)

            model.fit(train_X, train_y,
                      eval_metric="logloss", eval_set=eval_set,
                      verbose=False, early_stopping_rounds=10)
            sorted_idx = model.feature_importances_.argsort()
            sorted_importances = list(columns[i][sorted_idx])
            descriptors_across_fold.append(sorted_importances[:50])

        if i == 0:
            top_descriptors = set(list([item for sublist in descriptors_across_fold for item in sublist]))
            with open(results_path / "Feature_selection_descriptors.txt", "a") as file:
                print('Toxprint', file=file)
                print(top_descriptors, file=file)
                print('\n')

        if i == 1:
            top_descriptors = set(list([item for sublist in descriptors_across_fold for item in sublist]))
            with open(results_path / "Feature_selection_descriptors.txt", "a") as file:
                print('Padel', file=file)
                print(top_descriptors, file=file)
                print('\n')

        if i == 2:
            top_descriptors = set(list([item for sublist in descriptors_across_fold for item in sublist]))
            with open(results_path / "Feature_selection_descriptors.txt", "a") as file:
                print('Alvadesc', file=file)
                print(top_descriptors, file=file)
                print('\n')

        if i == 3:
            top_descriptors = set(list([item for sublist in descriptors_across_fold for item in sublist]))
            with open(results_path / "Feature_selection_descriptors.txt", "a") as file:
                print('Rdkit', file=file)
                print(top_descriptors, file=file)
                print('\n')
                print('\n')


if __name__ == '__main__':
    main()






