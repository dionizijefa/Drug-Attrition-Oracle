from pathlib import Path
import click
import numpy as np
import pandas as pd
from torch import cat
from tqdm import tqdm
from TDC_EGConv_lightning import EGConvNet as TDCModel
from utils.data_func import create_loader
from smiles_only.EGConv_lightning import EGConvNet as WithdrawnModel

root = Path(__file__).resolve().parents[2].absolute()

regression_tasks = ['Caco2_Wang', 'Lipophilicity_AstraZeneca','Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo',
                     'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'LD50_Zhu']
classification_tasks = ['HIA_Hou','Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins', 'CYP2C19_Veith',
                    'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith','CYP2C9_Substrate_CarbonMangels',
                     'CYP2D6_Substrate_CarbonMangels','CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI',
                    'Skin Reaction', 'Carcinogens_Languin','ClinTox','nr-ar', 'nr-ar-lbd', 'nr-ahr', 'nr-aromatase',
                     'nr-er','nr-er-lbd', 'nr-ppar-gamma', 'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp','sr-p53']

@click.command()
@click.option('-train_data', default='processing_pipeline/train/train.csv')
@click.option('-test_data', default='processing_pipeline/test/test.csv')
@click.option('-withdrawn_col', default='wd_consensus_1')
@click.option('-batch_size', default=32)
@click.option('-seed', default=0)
def main(train_data, test_data, withdrawn_col, batch_size, seed):
    data = pd.read_csv(root / 'data/{}'.format(train_data))[['standardized_smiles', withdrawn_col,'chembl_id']][:30]
    data = data.sample(frac=1, random_state=seed)  # shuffle
    test_data = pd.read_csv(root / 'data/{}'.format(test_data))[['standardized_smiles', withdrawn_col, 'chembl_id']][:30]

    data_loader = create_loader(data, withdrawn_col, batch_size)
    test_loader = create_loader(test_data, withdrawn_col, batch_size)

    admes = (root / 'production/TDC/')

    train_results = {}
    test_results = {}

    task_counter = 0
    num_tasks = 38
    for subdir in admes.iterdir():
        task_name = str(subdir).split('/')[-1]
        print('Doing inference for task {}/{}: {}'.format(task_counter, num_tasks, task_name))
        for file in Path(subdir / 'checkpoint').iterdir():
            checkpoint_file = file

        model = TDCModel.load_from_checkpoint(checkpoint_path=str(checkpoint_file), problem='inference')
        model.eval()

        train_outputs = []
        for i in tqdm(data_loader):
            train_outputs.append(model.forward(i.x, i.edge_index, i.batch))
        train_outputs = np.array(cat(train_outputs).detach().cpu().numpy().flatten())
        if task_name in classification_tasks:
            train_outputs = 1 / (1 + np.exp(-train_outputs))

        test_outputs = []
        for i in tqdm(test_loader):
            test_outputs.append(model.forward(i.x, i.edge_index, i.batch))
        test_outputs = np.array(cat(test_outputs).detach().cpu().numpy().flatten())
        if task_name in classification_tasks:
            test_outputs = 1 / (1 + np.exp(-test_outputs))


        train_results[task_name] = train_outputs
        test_results[task_name] = test_outputs
        print('Finished inference for task: {}'.format(task_name))
        task_counter += 1
        print('\n')

    withdrawn_model_path = root / 'production/egconv_production/production/checkpoint/epoch=6-step=398.ckpt'
    withdrawn_model = WithdrawnModel.load_from_checkpoint(checkpoint_path=withdrawn_model_path)
    withdrawn_model.eval()

    train_outputs = []
    for i in tqdm(data_loader):
        train_outputs.append(withdrawn_model.forward(i.x, i.edge_index, i.batch))
    train_outputs = np.array(cat(train_outputs).detach().cpu().numpy().flatten())
    train_outputs = 1 / (1 + np.exp(-train_outputs))

    test_outputs = []
    for i in tqdm(test_loader):
        test_outputs.append(withdrawn_model.forward(i.x, i.edge_index, i.batch))
    test_outputs = np.array(cat(test_outputs).detach().cpu().numpy().flatten())
    test_outputs = 1 / (1 + np.exp(-test_outputs))

    train_results['predict_withdrawn'] = train_outputs
    test_results['predict_withdrawn'] = test_outputs


    train_results_df = pd.DataFrame(train_results)
    train_results_df['standardized_smiles'] = data['standardized_smiles'].values
    train_results_df['chembl_id'] = data['chembl_id'].values
    train_results_df['{}'.format(withdrawn_col)] = data[withdrawn_col].values
    train_results_df.to_csv(root / 'data/processing_pipeline/TDC_predictions/train_subtasks_predictions.csv')

    test_results_df = pd.DataFrame(train_results)
    test_results_df['standardized_smiles'] = test_data['standardized_smiles'].values
    test_results_df['chembl_id'] = test_data['chembl_id'].values
    test_results_df['{}'.format(withdrawn_col)] = test_data[withdrawn_col].values
    test_results_df.to_csv(root / 'data/processing_pipeline/TDC_predictions/test_subtasks_predictions.csv')


if __name__ == '__main__':
    main()

