from pathlib import Path
import click
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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
                   'min_samples_leaf': min_samples_leaf}

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
    print(predictions)

if __name__ == '__main__':
    main()
