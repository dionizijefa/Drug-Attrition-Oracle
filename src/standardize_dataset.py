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
@click.option('-drop_duplicates', help='Drop duplicates from an ID column', default=None)
def standardize_dataset(dataset_path, drop_duplicates):
    data = pd.read_csv(dataset_path, index_col=0)
    withdrawn = pd.read_csv(data_path / 'data/raw/withdrawn.csv', index_col=0)
    drugbank = pd.read_csv(data_path / 'data/raw/structure links.csv', index_col=0)
    chembl = pd.read_csv(data_path / 'data/raw/chembl.csv', sep=';', error_bad_lines=True)
    standardized_smiles = []
    scaffolds_generic = []

    for smiles in data['smiles']:
        try:
            new_mol = standardise.run(smiles)
            standardized_smiles.append(new_mol)
        except Exception as e:
            print(e)
            standardized_smiles.append(0)
        try:
            # generate generic scaffolds -> all atom types are C and all bonds are single -> more strict split
            scaffolds_generic.append(MolToSmiles(MakeScaffoldGeneric(MolFromSmiles(new_mol))))
        except Exception as e:
            print(e)
            scaffolds_generic.append(0)

    # drop molecule which can't be standardized
    data['standardized_smiles'] = standardized_smiles
    data['scaffolds'] = scaffolds_generic
    data = data.drop_duplicates(subset=drop_duplicates)
    data = data.drop_duplicates(subset='standardized_smiles')# missing values should stay

    data = data.loc[data['standardized_smiles'] != 0]  # drop molecules that have not passed standardizer

    #post hoc add labels - duplicates are dropped, first value is kept, this fixed the consensus count
    wd_wd = list(withdrawn['chembl_id'])
    db_wd = list(drugbank.loc[drugbank['Drug Groups'].str.contains('withdrawn')]['ChEMBL ID'])
    cb_wd = list(chembl.loc[chembl['Availability Type'] == 'Withdrawn']['Parent Molecule'])

    data['withdrawn_chembl'] = 0
    data['withdrawn_drugbank'] = 0
    data['withdrawn_withdrawn'] = 0  # because wd contains only withdrawn, there would be too many NANs

    data.loc[data['chembl_id'].isin(cb_wd), 'withdrawn_chembl'] = 1
    data.loc[data['chembl_id'].isin(db_wd), 'withdrawn_drugbank'] = 1
    data.loc[data['chembl_id'].isin(wd_wd), 'withdrawn_withdrawn'] = 1


    # load original data and add labels, so it's implicitly known where from the data is

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

    output_name = str(dataset_path).split('.')[0]
    data.to_csv(data_path / '{}_standardized.csv'.format(output_name))


if __name__ == "__main__":
    standardize_dataset()
