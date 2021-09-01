import pandas as pd
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from standardiser import standardise

def standardise_dataset(data: pd.DataFrame):
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

