from pathlib import Path

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
    data = data.drop_duplicates(subset='standardized_smiles')
    data = data.drop_duplicates(subset='parent_inchi_key') # missing values should stay

    data = data.loc[data['standardized_smiles'] != 0]  # drop molecules that have not passed standardizer
    output_name = str(dataset_path).split('.')[0]
    data.to_csv(data_path / '{}_standardized.csv'.format(output_name))


if __name__ == "__main__":
    standardize_dataset()
