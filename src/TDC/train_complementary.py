import sys
from shap.explainers.pytree import TreeExplainer
from xgboost import XGBClassifier
from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import pickle

sys.path.append('../..')
from src.utils.metrics import table_metrics_trees

root = Path(__file__).resolve().parents[2].absolute()


@click.command()
@click.option('-train_data', default='processing_pipeline/TDC_predictions/train_subtasks_predictions.csv')
@click.option('-test_data', default='processing_pipeline/TDC_predictions/test_subtasks_predictions.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, seed):
    data = pd.read_csv(root / 'data/{}'.format(train_data), index_col=0)
    data = data.sample(frac=1, random_state=seed)  # shuffle
    test_data = pd.read_csv(root / 'data/{}'.format(test_data), index_col=0)
    y_train = data[withdrawn_col]
    y_test = test_data[withdrawn_col]
    X_train = data.drop(columns=['chembl_id', 'standardized_smiles', withdrawn_col])
    X_test = test_data.drop(columns=['chembl_id', 'standardized_smiles', withdrawn_col])

    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'scale_pos_weight': [5, 10, 15, 20, 35],
    }

    classifier = XGBClassifier()
    rs_model = RandomizedSearchCV(classifier, param_distributions=params, n_iter=100, scoring='average_precision',
                                  n_jobs=-1, cv=6, verbose=3)
    rs_model.fit(X_train, y_train)

    predictions = rs_model.best_estimator_.predict_proba(X_test)
    test_pred_df = pd.DataFrame({'probabilities': predictions[:, 1],
                                 withdrawn_col: y_test,
                                 'predicted_class': rs_model.predict(X_test)})

    results_path = Path(root / 'complementary_model_results')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)

    results = table_metrics_trees(test_pred_df, withdrawn_col)
    results.to_csv(results_path / 'complementary_results_full.csv')

    # If folder exists - add production complementary models
    optimization_results = pd.DataFrame(rs_model.cv_results_)
    optimization_results.to_csv(results_path / 'complementary_optimization_full.csv')

    # save the predictor
    predictor_path = Path(root / 'production/complementary_model')
    if not predictor_path.exists():
        predictor_path.mkdir(exist_ok=True, parents=True)
    with open(predictor_path / 'xgb_classifier_full.pkl', 'wb') as file:
        pickle.dump(rs_model.best_estimator_, file)

    """ Reduced model """
    # Selected top 5 features by shapely values
    explainer = TreeExplainer(rs_model.best_estimator_)
    SHAP_values = explainer.shap_values(X_train)
    top_5_features = list(
        pd.DataFrame(
            columns=X_train.columns,
            data=abs(np.mean(SHAP_values, axis=0))[np.newaxis]
        ).transpose().sort_values(0, ascending=False)[:5].index)

    rs_model_reduced = RandomizedSearchCV(classifier, param_distributions=params, n_iter=100,
                                          scoring='average_precision', n_jobs=-1, cv=6, verbose=3)
    rs_model_reduced.fit(X_train[top_5_features], y_train)

    predictions = rs_model_reduced.best_estimator_.predict_proba(X_test[top_5_features])
    test_pred_df = pd.DataFrame({'probabilities': predictions[:, 1],
                                 withdrawn_col: y_test,
                                 'predicted_class': rs_model_reduced.predict(X_test[top_5_features])})

    results_path = Path(root / 'complementary_model_results')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)

    results = table_metrics_trees(test_pred_df, withdrawn_col)
    results.to_csv(results_path / 'complementary_results_reduced.csv')

    # save the predictor
    predictor_path = Path(root / 'production/complementary_model')
    if not predictor_path.exists():
        predictor_path.mkdir(exist_ok=True, parents=True)
    with open(predictor_path / 'xgb_classifier_reduced.pkl', 'wb') as file:
        pickle.dump(rs_model_reduced.best_estimator_, file)


if __name__ == '__main__':
    main()
