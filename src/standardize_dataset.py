from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from standardiser import standardise
import click

data_path = Path(__file__).resolve().parents[1].absolute()


@click.command()
@click.option('-dataset_path', help='Path of the dataset')
@click.option('-smiles_col', help='Column name of the smiles column', default='original_smiles')
@click.option('-drop_duplicates', help='Drop duplicates from an ID column', default=None)
def standardize_dataset(dataset_path, smiles_col, drop_duplicates):
    output_name = str(dataset_path).split('.')[0]
    data = pd.read_csv(dataset_path, index_col=0)

    standardized_smiles = []
    scaffolds_generic = []
    list_of_dropped = []

    data = data.reset_index()
    for index, smiles in enumerate(data[smiles_col]):
        try:
            new_mol = standardise.run(smiles)
            standardized_smiles.append(new_mol)
        except Exception as e:
            print(e)
            standardized_smiles.append(0)
            list_of_dropped.append(index)
        try:
            # generate generic scaffolds -> all atom types are C and all bonds are single -> more strict split
            scaffolds_generic.append(MolToSmiles(MakeScaffoldGeneric(MolFromSmiles(new_mol))))
        except Exception as e:
            print(e)
            scaffolds_generic.append(0)

    # drop molecule which can't be standardized
    list_of_dropped = data.iloc[list_of_dropped][['chembl_id', 'original_smiles']]
    list_of_dropped.to_csv(data_path / '{}_standardizer_dropped_mols.csv'.format(output_name))
    data['standardized_smiles'] = standardized_smiles
    data['scaffolds'] = scaffolds_generic
    data = data.loc[data['standardized_smiles'] != 0]  # drop molecules that have not passed standardizer

    #post hoc add labels - duplicates are dropped, first value is kept, this fixed the consensus count
    wd_wd = list(data.loc[data['wd_withdrawn'] == 1]['chembl_id'])
    db_wd = list(data.loc[data['wd_drugbank'] == 1]['chembl_id'])
    cb_wd = list(data.loc[data['wd_chembl'] == 1]['chembl_id'])

    data['wd_consensus_1'] = 0
    data['wd_consensus_2'] = 0
    data['wd_consensus_3'] = 0

    for index, row in data.iterrows():
        id = row['chembl_id']
        is_wd_db = 0
        is_wd_cb = 0
        is_wd_wd = 0
        if id in db_wd:
            is_wd_db = 1
        if id in cb_wd:
            is_wd_cb = 1
        if id in wd_wd:
            is_wd_wd = 1

        sum = is_wd_db + is_wd_cb + is_wd_wd

        if sum == 1:
            data.loc[data['chembl_id'] == id, 'wd_consensus_1'] = 1
        if sum == 2:
            data.loc[data['chembl_id'] == id, 'wd_consensus_1'] = 1
            data.loc[data['chembl_id'] == id, 'wd_consensus_2'] = 1
        if sum == 3:
            data.loc[data['chembl_id'] == id, 'wd_consensus_1'] = 1
            data.loc[data['chembl_id'] == id, 'wd_consensus_2'] = 1
            data.loc[data['chembl_id'] == id, 'wd_consensus_3'] = 1

    data.to_csv(data_path / '{}_standardized.csv'.format(output_name))


if __name__ == "__main__":
    standardize_dataset()
