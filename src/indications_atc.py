from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm


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

if __name__ == "__main__":
    download_data()
