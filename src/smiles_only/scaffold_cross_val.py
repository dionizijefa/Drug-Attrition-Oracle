from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from data_utils import smiles2graph
from torch import Tensor

def scaffold_cross_val(data, withdrawn_col, batch_size, seed):
    scaffolds_df = pd.DataFrame(data['scaffolds'].value_counts())
    unique_scaffolds = list(scaffolds_df.loc[scaffolds_df['scaffolds'] == 1].index)
    data_unique_scaffolds = data.loc[data['scaffolds'].isin(unique_scaffolds)]

    cv_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    loaders = []
    for k, (train_index, test_index) in enumerate(
            cv_splitter.split(data_unique_scaffolds, data_unique_scaffolds[withdrawn_col])
    ):

        test = data_unique_scaffolds.iloc[test_index]
        test_data_list = []
        for index, row in test.iterrows():
            test_data_list.append(smiles2graph(row, withdrawn_col))
        test_loader = DataLoader(test_data_list, num_workers=0, batch_size=batch_size)

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
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train[withdrawn_col].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph(row, withdrawn_col))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

        loaders.append([train_loader, val_loader, test_loader])

    return loaders

def create_loader(data, withdrawn_col, batch_size):
    data_list = []
    for index, row in data.iterrows():
        data_list.append(smiles2graph(row, withdrawn_col))

    data_loader = DataLoader(data_list, num_workers=0, batch_size=batch_size)





