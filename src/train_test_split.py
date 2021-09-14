from pathlib import Path
import click
import pandas as pd
from sklearn.model_selection import train_test_split as splitter

path = Path(__file__).resolve().parents[1].absolute()


@click.command()
@click.option('-dataset_path', help='Path of the dataset')
@click.option('-seed', help='Random seed of the split', default=0)
def train_test_split(dataset_path, seed):
    data = pd.read_csv(dataset_path, index_col=0)

    scaffolds_df = pd.DataFrame(data['scaffolds'].value_counts())
    unique_scaffolds = list(scaffolds_df.loc[scaffolds_df['scaffolds'] == 1].index)
    data_unique_scaffolds = data.loc[data['scaffolds'].isin(unique_scaffolds)]
    train, test = splitter(data_unique_scaffolds, test_size=0.20,
                           stratify=data_unique_scaffolds['wd_consensus_1'],
                           random_state=seed)
    test.to_csv(path / 'data/processing_pipeline/test/test.csv')
    train = pd.concat([train, data.loc[~data['scaffolds'].isin(unique_scaffolds)]]) # append data not in unique scaffolds
    train.to_csv(path / 'data/processing_pipeline/train/train.csv')

    print('Data split to train and test set')

if __name__ == "__main__":
    train_test_split()
