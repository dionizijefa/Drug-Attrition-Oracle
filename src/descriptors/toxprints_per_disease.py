import re
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from pathlib import Path
import click
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
sys.path.append('../..')
from src.utils.descriptors_list import toxprint_descriptors_10pct

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='train/train.csv')
@click.option('-test_data', default='test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, seed):
    train = pd.read_csv(root / 'data/processing_pipeline/{}'.format(train_data))[['chembl_id', withdrawn_col]]
    train = train.sample(frac=1, random_state=seed)  # shuffle
    test = pd.read_csv(root / 'data/processing_pipeline/{}'.format(test_data))[['chembl_id', withdrawn_col]]
    toxprints = pd.read_csv(root / 'data/processing_pipeline/descriptors/toxprint_descriptors.csv')
    chembl_ids = toxprints['chembl_id']
    toxprints = toxprints[toxprint_descriptors_10pct]  # drop mostly 0 zescriptors
    toxprints['chembl_id'] = chembl_ids
    master_atc = pd.read_csv(root / 'data/processing_pipeline/master_atc.csv', index_col=0)

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    toxprints.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                         toxprints.columns.values]

    train = train.merge(master_atc, how='inner', on='chembl_id')
    test = test.merge(master_atc, how='inner', on='chembl_id')

    train = train.merge(toxprints, how='inner', on='chembl_id')
    test = test.merge(toxprints, how='inner', on='chembl_id')

    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'scale_pos_weight': [5, 10, 15, 20, 35],
        'n_estimators': [100, 200, 300, 400, 500],
    }

    train['atc_code'] = train['atc_code'].str.split('0').str[0]
    train['atc_code'] = train['atc_code'].str.split('1').str[0]

    test['atc_code'] = test['atc_code'].str.split('0').str[0]
    test['atc_code'] = test['atc_code'].str.split('1').str[0]

    ap_overall = []
    atc_codes = []
    auroc_overall = []
    results = []
    for i in list(train['atc_code'].unique()):
        try:
            train_subset = train.loc[train['atc_code'] == i]
            test_subset = test.loc[test['atc_code'] == i]

            train_subset = train_subset.drop_duplicates(subset=['chembl_id', 'atc_code'])
            test_subset = test_subset.drop_duplicates(subset=['chembl_id', 'atc_code'])

            cv_splitter = StratifiedKFold(
                n_splits=6,
                shuffle=True,
                random_state=0)

            features_across_fold = []
            ap_fold = []
            auroc_fold = []
            for k, (train_index, test_index) in enumerate(
                    cv_splitter.split(train_subset, train_subset[withdrawn_col])
            ):
                y_train = train_subset.iloc[train_index]['wd_consensus_1']
                y_test = train_subset.iloc[test_index]['wd_consensus_1']
                X_train = train_subset.iloc[train_index].drop(columns=['wd_consensus_1', 'chembl_id', 'atc_code'])
                X_test = train_subset.iloc[test_index].drop(columns=['wd_consensus_1', 'chembl_id', 'atc_code'])

                classifier = XGBClassifier()
                rs_model = RandomizedSearchCV(classifier, param_distributions=params,
                                              n_iter=1, scoring='average_precision',
                                              n_jobs=-1, cv=6, verbose=3)
                rs_model.fit(X_train, y_train)
                predict_proba = rs_model.best_estimator_.predict_proba(X_test)
                ap = average_precision_score(y_test, predict_proba[:, 1])
                auroc = roc_auc_score(y_test, predict_proba[:, 1])
                sorted_idx = rs_model.best_estimator_.feature_importances_.argsort()
                sorted_importances = list(X_train.columns[sorted_idx][-10:])
                features_across_fold.append(sorted_importances)
                ap_fold.append(ap)
                auroc_fold.append(auroc)
            top_features = (
                pd.DataFrame(features_across_fold).melt().groupby('value').sum().sort_values('variable', ascending=False)[
                :10].index)
            results.append(top_features)
            atc_codes.append(i)
            ap_overall.append(np.mean(ap_fold))
            auroc_overall.append(np.mean(auroc_fold))

        except:
            continue

    results_df = pd.DataFrame(results, index=atc_codes)
    results_df['mean_average_precision'] = ap_overall
    results_df['mean_auroc'] = auroc_overall
    results_df.to_csv(root / 'complementary_model_results/toxprint_disease.csv')

if __name__ == '__main__':
    main()
