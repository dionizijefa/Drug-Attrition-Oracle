from abc import ABC
from pathlib import Path
from typing import Optional
import click
import pandas as pd
import pytorch_lightning as pl
import torch
from data_utils import load_data_from_smiles, construct_dataset
from tqdm import tqdm
from transformer import make_model

root = Path(__file__).resolve().parents[2].absolute()

class TransformerNet(pl.LightningModule, ABC):
    def __init__(
            self,
            reduce_lr: Optional[bool] = True,
    ):
        super().__init__()
        self.reduce_lr = reduce_lr
        self.model_params = {
            'd_atom': 28,
            'd_model': 1024,
            'N': 8,
            'h': 16,
            'N_dense': 1,
            'lambda_attention': 0.33,
            'lambda_distance': 0.33,
            'leaky_relu_slope': 0.1,
            'dense_output_nonlinearity': 'relu',
            'distance_matrix_kernel': 'exp',
            'dropout': 0.0,
            'aggregation_type': 'mean'
        }
        self.model = make_model(**self.model_params)

    def forward(self, node_features, batch_mask, adjacency_matrix, distance_matrix):
        out = self.model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        return out


@click.command()
@click.option('-train_data', default='chembl_4_smiles.csv')
@click.option('-dataset', default='all')
@click.option('-withdrawn_col', default='withdrawn')
def main(train_data, dataset, withdrawn_col):
    if dataset == 'all':
        data = pd.read_csv(root / 'data/{}'.format(train_data))[['smiles', withdrawn_col, 'chembl_id']]

    else:
        data = pd.read_csv(root / 'data/{}'.format(train_data))
        data = data.loc[(data['dataset'] == dataset) |
                        (data['dataset'] == 'both') |
                        (data['dataset'] == 'withdrawn')][['smiles', withdrawn_col, 'chembl_id']]

    X_train, y_train = load_data_from_smiles(data['smiles'], data[withdrawn_col],
                                             one_hot_formal_charge=True)

    admes = (root / 'models/TDC/transformer_net')

    results = {}
    task_counter = 0
    num_tasks = 37
    for subdir in admes.iterdir():
        task_name = str(subdir).split('/')[-1]
        print('Doing inference for task {}/{}: {}'.format(task_counter, num_tasks, task_name))
        for file in Path(subdir / 'checkpoint').iterdir():
            checkpoint_file = file

        outputs = []
        model = TransformerNet.load_from_checkpoint(checkpoint_path=str(checkpoint_file), task=task_name)
        model.eval()
        for i in tqdm(X_train):
            node_features = torch.Tensor(i[0]).unsqueeze(0)
            adjacency_matrix = torch.Tensor(i[1]).unsqueeze(0)
            distance_matrix = torch.Tensor(i[1]).unsqueeze(0)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            outputs.append(model.forward(node_features, batch_mask, adjacency_matrix, distance_matrix).squeeze(
                -1).detach().cpu().numpy()[0])

        results[task_name] = outputs
        print('Finished inference for task: {}'.format(task_name))
        task_counter += 1
        print('\n')

    results_df = pd.DataFrame(results)
    results_df['smiles'] = data['smiles'].values
    results_df['chembl_id'] = data['chembl_id'].values
    results_df['withdrawn'] = data[withdrawn_col].values
    results_df.to_csv(root / 'data/subtasks_predictions.csv')


if __name__ == '__main__':
    main()

