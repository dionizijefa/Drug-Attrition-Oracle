from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import pubchempy as pyp
import numpy as np


data_path = Path(__file__).resolve().parents[1].absolute()

def download_data():
    molecule = new_client.molecule
    drug_indication = new_client.drug_indication
    atc_class = new_client.atc_class
    data = pd.read_csv(data_path / 'data/processing_pipeline/MasterDB_15Sep2021_standardized.csv')

    indication = []
    chembl_id = []
    max_phase_for_ind = []
    mesh_heading = []
    mesh_id = []
    for i in drug_indication:
        chembl_id.append(i['molecule_chembl_id'])
        max_phase_for_ind.append(i['max_phase_for_ind'])
        indication.append(i['efo_term'])
        mesh_heading.append(i['mesh_heading'])
        mesh_id.append(i['mesh_id'])

    indications_df = pd.DataFrame({'indication': indication,
                                   'chembl_id': chembl_id,
                                   'max_phase_for_ind': max_phase_for_ind,
                                   'mesh_heading': mesh_heading,
                                   'mesh_id': mesh_id})

    indications_df = indications_df.loc[indications_df['chembl_id'].isin(data['chembl_id'])]
    indications_df.to_csv(data_path / 'data/processing_pipeline/drug_indications.csv')

    chembl_id = []
    atc_classifications = []
    for i in tqdm(data['chembl_id']):
        try:
            for j in molecule.get(i)['atc_classifications']:
                chembl_id.append(i)
                atc_classifications.append(j)
        except:
            print('No mol')

    atc_df = pd.DataFrame({'atc_code': atc_classifications,
                           'chembl_id': chembl_id})
    atc_df.to_csv(data_path / 'data/processing_pipeline/chembl_atc_codes.csv')

    atc_kegg = pd.read_csv(data_path / 'data/raw/drug_atc.list', sep='\t', header=None)
    inchi_kegg = pd.read_csv(data_path / 'data/raw/drug.inchi', sep='\t', header=None)

    atc_kegg[0] = atc_kegg[0].str.split('dr:').str[1]
    inchi_kegg = inchi_kegg.loc[inchi_kegg[0].isin(atc_kegg[0])]

    inchikey_kegg = []
    for i in tqdm(inchi_kegg[1]):
        try:
            inchikey_kegg.append(pyp.get_compounds(i, 'inchi')[0].to_dict()['inchikey'])
        except:
            inchikey_kegg.append(np.NaN)

    inchi_kegg['inchikey'] = inchikey_kegg

    molecule = new_client.molecule

    chembl_ids = []
    for i in tqdm(inchi_kegg['inchikey']):
        try:
            res = molecule.get(i)
            chembl_id = res['molecule_chembl_id']
            chembl_ids.append(chembl_id)
        except:
            chembl_ids.append(0)

    inchi_kegg = inchi_kegg.merge(atc_kegg, how='inner', on=0)
    inchi_kegg.rename(columns={0: 'kegg_id', '1_x': 'inchi', 'inchikey': 'inchi_key', '1_y': 'atc_code'}, inplace=True)
    inchi_kegg['atc_code'] = inchi_kegg['atc_code'].str.split('atc:').str[1]
    atc_kegg = inchi_kegg[['chembl_id', 'atc_code']]

    joined_codes = pd.concat([atc_kegg, atc_df])
    master_db = data[['chembl_id']]
    master_atc = master_db.merge(joined_codes, how='inner', on='chembl_id').drop_duplicates(
        subset=['chembl_id', 'atc_code'])
    master_atc.to_csv(data_path / 'data/processing_pipeline/master_atc.csv')

if __name__ == "__main__":
    download_data()
