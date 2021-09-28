from pathlib import Path
import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.TDC import TDC_EGConv_lightning
from src.utils.data_func import create_loader
from src.smiles_only.EGConv_lightning import EGConvNet

root = Path(__file__).resolve().parents[2].absolute()

@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=32)
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, batch_size, seed):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col,'chembl_id']][:10]
    data = data.sample(frac=1, random_state=seed)  # shuffle
    test_data = pd.read_csv(root / 'data/{}'.format(test_data))[['standardized_smiles', withdrawn_col, 'chembl_id']]

    data_loader = create_loader(data, withdrawn_col, batch_size)
    test_loader = create_loader(test_data, withdrawn_col, batch_size)

    admes = (root / 'production/TDC/production')

    train_results = {}
    test_results = {}

    task_counter = 0
    num_tasks = 38
    for subdir in admes.iterdir():
        task_name = str(subdir).split('/')[-1]
        print('Doing inference for task {}/{}: {}'.format(task_counter, num_tasks, task_name))
        for file in Path(subdir / 'checkpoint').iterdir():
            checkpoint_file = file

        model = TDC_EGConv_lightning.load_from_checkpoint(checkpoint_path=str(checkpoint_file), problem='inference')
        model.eval()

        train_outputs = []
        for i in tqdm(data_loader):
            train_outputs.append(model.forward(i.x, i.edge_index, i.batch))
        train_outputs = 1 / (1 + np.exp(-train_outputs))

        test_outputs = []
        for i in tqdm(test_loader):
            test_outputs.append(model.forward(i.x, i.edge_index, i.batch))
        test_outputs = 1 / (1 + np.exp(-test_outputs))

        train_results[task_name] = train_outputs
        test_results[task_name] = test_outputs
        print('Finished inference for task: {}'.format(task_name))
        task_counter += 1
        print('\n')

    withdrawn_model_path = root / 'production/egconv_production/production/checkpoint/epoch=6-step=398.ckpt'
    withdrawn_model = EGConvNet.load_from_checkpoint(checkpoint_path=withdrawn_model_path)
    withdrawn_model.eval()

    train_outputs = []
    for i in tqdm(data_loader):
        train_outputs.append(withdrawn_model.forward(i.x, i.edge_index, i.batch))
    train_outputs = 1 / (1 + np.exp(-train_outputs))

    test_outputs = []
    for i in tqdm(test_loader):
        test_outputs.append(withdrawn_model.forward(i.x, i.edge_index, i.batch))
    test_outputs = 1 / (1 + np.exp(-test_outputs))

    train_results['predict_withdrawn'] = train_outputs
    test_results['predict_withdrawn'] = test_outputs


    train_results_df = pd.DataFrame(train_results)
    train_results_df['smiles'] = data['smiles'].values
    train_results_df['chembl_id'] = data['chembl_id'].values
    train_results_df['{}'.format(withdrawn_col)] = data[withdrawn_col].values
    train_results_df.to_csv(root / 'data/TDC/train_subtasks_predictions.csv')

    test_results_df = pd.DataFrame(train_results)
    test_results_df['smiles'] = test_data['smiles'].values
    test_results_df['chembl_id'] = test_data['chembl_id'].values
    test_results_df['{}'.format(withdrawn_col)] = test_data[withdrawn_col].values
    test_results_df.to_csv(root / 'data/TDC/train_subtasks_predictions.csv')


if __name__ == '__main__':
    main()

