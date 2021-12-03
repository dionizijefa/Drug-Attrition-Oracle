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
            print(smiles)
            standardized_smiles.append(np.nan)
            list_of_dropped.append(index)
        try:
            # generate generic scaffolds -> all atom types are C and all bonds are single -> more strict split
            scaffolds_generic.append(MolToSmiles(MakeScaffoldGeneric(MolFromSmiles(new_mol))))
        except Exception as e:
            print(e)
            scaffolds_generic.append(0)

    # drop molecule which can't be standardized
    #list_of_dropped = data.iloc[list_of_dropped][['chembl_id', smiles_col]]
    #list_of_dropped.to_csv(data_path / '{}_standardizer_non_standard_mols.csv'.format(output_name))
    data['standardized_smiles'] = standardized_smiles
    data['scaffolds'] = scaffolds_generic
    data['standardized_smiles'] = data['standardized_smiles'].fillna(data[smiles_col])


    data.to_csv(data_path / '{}_standardized.csv'.format(output_name))


if __name__ == "__main__":
    standardize_dataset()
