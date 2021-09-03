from pathlib import Path

import pandas as pd
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from standardiser import standardise
import click

data_path = Path(__file__).resolve().parents[1].absolute()


@click.command()
@click.option('-dataset_path', help='Path of the dataset')
def standardize_dataset(dataset_path):
    data = pd.read_csv(dataset_path, index_col=0)
    standardized_smiles = []
    scaffolds_generic = []

    for smiles in data['smiles']:
        try:
            new_mol = standardise.run(smiles)
            standardized_smiles.append(new_mol)
            # generate generic scaffolds -> all atom types are C and all bonds are single -> more strict split
            scaffolds_generic.append(MolToSmiles(MakeScaffoldGeneric(MolFromSmiles(new_mol))))
        except Exception as e:
            print(e)
            scaffolds_generic.append(0)
            standardized_smiles.append(0)

    # drop molecule which can't be standardized
    data = data.drop(columns=['smiles'])
    data['smiles'] = standardized_smiles
    data['scaffolds'] = scaffolds_generic

    data = data.loc[data['smiles'] != 0]  # drop molecules that have not passed standardizer
    output_name = str(dataset_path).split('.')[0]
    data.to_csv(data_path / '{}_standardized.csv'.format(output_name))


if __name__ == "__main__":
    standardize_dataset()
