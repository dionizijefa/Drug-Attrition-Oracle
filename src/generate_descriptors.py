from pathlib import Path
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from tqdm import tqdm
import numpy as np
import pandas as pd
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from tqdm import tqdm
import click

path = Path(__file__).resolve().parents[1].absolute()


@click.command()
@click.option('-dataset_path', help='Path of the dataset')
def generate_descriptors(dataset_path):
    """Generates RDKIT descriptors"""
    data = pd.read_csv(dataset_path, index_col=0)
    generator = MakeGenerator(("RDKit2DNormalized",))
    mols = data['chembl_id']
    descriptors = []
    col_names_dtype = generator.GetColumns()[1:]
    col_names = []
    for i in col_names_dtype:
        col_names.append(i[0])
    for i in tqdm(data['standardized_smiles']):
        descriptors.append(generator.process(i)[1:])

    descriptors_df = pd.DataFrame(columns=col_names, data=descriptors)
    descriptors_df = descriptors_df.dropna(axis=1)
    descriptors_df['chembl_id'] = mols
    descriptors_df.to_csv(path / 'data/processing_pipeline/descriptors/rdkit_descriptors.csv')

if __name__ == "__main__":
    generate_descriptors()
