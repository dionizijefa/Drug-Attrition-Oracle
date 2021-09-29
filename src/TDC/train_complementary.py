from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import pickle

from src.utils.metrics import table_metrics

root = Path(__file__).resolve().parents[2].absolute()

@click.command()
@click.option('-train_data', default='processing_pipeline/TDC_predictions/train_subtasks_predictions.csv')
@click.option('-test_data', default='processing_pipeline/TDC_predictions/test_subtasks_predictions.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, seed):
    data = pd.read_csv(root / 'data/{}'.format(train_data))
    data = data.sample(frac=1, random_state=seed)  # shuffle
    test_data = pd.read_csv(root / 'data/{}'.format(test_data))

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt', "log2"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier(class_weight='balanced')
    print("Running 100 iterations of random search of Random Forest optimization")
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=5, random_state=seed, n_jobs=-1,
                                   scoring='average_precision')
    y_train = data[withdrawn_col]
    X_train = data.drop(columns=['chembl_id', 'standardized_smiles', withdrawn_col])
    y_test = test_data[withdrawn_col]
    X_test = data.drop(columns=['chembl_id', 'standardized_smiles', withdrawn_col])

    rf_random.fit(X_train, y_train)
    predictions = rf_random.best_estimator_.predict_proba(X_test)

    predictions_train = rf_random.best_estimator_.predict_proba(X_train)
    train_pred_df = pd.DataFrame({'probabilities': predictions_train,
                                  'target': y_train})

    #optimal threshold F1 withdrawn class random forest
    optimal_f1_score = []
    optimal_threshold = []
    for threshold in np.arange(0, 1, 0.01):
        predictions_df = train_pred_df.copy()
        predictions_df['predicted_class'] = 0
        predictions_df.loc[predictions_df['probabilities'] > threshold, 'predicted_class'] = 1
        optimal_f1_score.append(f1_score(
            predictions_df['target'], predictions_df['predicted_class'], average='binary'
        ))
        optimal_threshold.append(threshold)

    optimal_f1_index = np.argmax(np.array(optimal_f1_score))
    optimal_threshold = optimal_threshold[optimal_f1_index]

    test_pred_df = pd.DataFrame({'probabilities': predictions,
                                withdrawn_col: y_test})

    results_path = Path(root / 'complementary_model_results')
    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)

    results = table_metrics(test_pred_df, withdrawn_col, optimal_threshold)
    results.to_csv(results_path / 'complementary_results.csv')

    #If folder exists - add production complementary models
    optimization_results = pd.DataFrame(rf_random.cv_results_)
    optimization_results.to_csv(results_path / 'complementary_optimization.csv')

    #save the predictor
    predictor_path = Path(root / 'production/complementary_model')
    if not predictor_path.exists():
        predictor_path.mkdir(exist_ok=True, parents=True)
    with open(predictor_path / 'random_forest_classifier.pkl', 'wb') as file:
        pickle.dump(rf_random.best_estimator_, file)

if __name__ == '__main__':
    main()
