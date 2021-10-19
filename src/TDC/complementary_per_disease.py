import sys
from xgboost import XGBClassifier
from pathlib import Path
import click
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

sys.path.append('../..')
from src.utils.metrics import table_metrics_trees

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/TDC_predictions/train_subtasks_predictions.csv')
@click.option('-test_data', default='processing_pipeline/TDC_predictions/test_subtasks_predictions.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, seed):
    train = pd.read_csv(root / 'data/{}'.format(train_data), index_col=0)
    train = train.sample(frac=1, random_state=seed)  # shuffle
    test = pd.read_csv(root / 'data/{}'.format(test_data), index_col=0)
    master_atc = pd.read_csv(root / 'data/processing_pipeline/master_atc.csv', index_col=0)

    train = train.merge(master_atc, how='inner', on='chembl_id')
    test = test.merge(master_atc, how='inner', on='chembl_id')

    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'scale_pos_weight': [5, 10, 15, 20, 35],
        'n_estimators': [100, 200, 300, 400, 500],
    }

    results = []
    atc_codes = []
    columns = []
    for i in list(train['atc_code'].unique()):
        train_subset = train.loc[train['atc_code'] == i]
        test_subset = test.loc[test['atc_code'] == i]

        y_train = train_subset['wd_consensus_1']
        y_test = test_subset['wd_consensus_1']
        X_train = train_subset.drop(columns=['wd_consensus_1', 'chembl_id', 'standardized_smiles', 'atc_code'])
        X_test = test_subset.drop(columns=['wd_consensus_1', 'chembl_id', 'standardized_smiles', 'atc_code'])

        classifier = XGBClassifier()
        rs_model = RandomizedSearchCV(classifier, param_distributions=params,
                                      n_iter=100, scoring='average_precision',
                                      n_jobs=-1, cv=6, verbose=3)
        rs_model.fit(X_train, y_train)

        predictions = rs_model.best_estimator_.predict_proba(X_test)
        test_pred_df = pd.DataFrame({'probabilities': predictions[:, 1],
                                     withdrawn_col: y_test,
                                     'predicted_class': rs_model.predict(X_test)})
        results.append(table_metrics_trees(test_pred_df, withdrawn_col).values[0])
        columns.append(table_metrics_trees(test_pred_df, withdrawn_col).columns)
        atc_codes.append(i)

    results_df = pd.DataFrame(results, columns=columns[0], index=atc_codes)
    results_df.to_csv(root / 'complementary_model_results/complementary_disease.csv')

if __name__ == '__main__':
    main()
