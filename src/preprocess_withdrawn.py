import pandas as pd
from rdkit.Chem.PandasTools import LoadSDF
import pubchempy as pcp
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client
from pathlib import Path
import click


@click.command()
@click.option('-phase', default=4, help='Minimum phase of the drug to use')
def preprocess(phase):
    data_path = Path(__file__).resolve().parents[1].absolute()
    data = pd.read_csv(data_path / 'data/raw/chembl.csv', sep=';', error_bad_lines=True)

    """
    Preprocess chembl
        1) Drop polymer and inorganic molecules
        2) Keep only phase 1 and two compounds
        3) Try to find pubchem cids for each CHEMBL ID
    """

    data = data.loc[data['Drug Type'] != "10:Polymer"]
    data = data.loc[data['Drug Type'] != "9:Inorganic"]
    data = data.loc[data['Phase'] >= phase]

    data = data[['Parent Molecule', 'Synonyms', 'Phase', 'ATC Codes', 'Level 4 ATC Codes', 'Level 3 ATC Codes',
                 'Level 2 ATC Codes', 'Level 1 ATC Codes',
                 'Drug Type', 'Passes Rule of Five', 'Chirality', 'Prodrug', 'Oral', 'Parenteral', 'Topical',
                 'Black Box', 'Availability Type', 'Withdrawn Year',
                 'Withdrawn Reason', 'Withdrawn Country', 'Withdrawn Class', 'Smiles']]

    data.rename(columns={'Parent Molecule': 'chembl_id',
                         'Synonyms': 'synonyms',
                         'Phase': 'max_phase',
                         'ATC Codes': 'atc_codes',
                         'Level 4 ATC Codes': 'level_4_atc',
                         'Level 3 ATC Codes': 'level_3_atc',
                         'Level 2 ATC Codes': 'level_2_atc',
                         'Level 1 ATC Codes': 'levle_1_atc',
                         'Drug Type': 'drug_type',
                         'Passes Rule of Five': 'passes_ro5',
                         'Chirality': 'chirality',
                         'Prodrug': 'prodrug',
                         'Oral': 'oral',
                         'Parenteral': 'parenteral',
                         'Topical': 'topical',
                         'Black Box': 'black_box',
                         'Availability Type': 'availability_type',
                         'Withdrawn Year': 'withdrawn_year',
                         'Withdrawn Reason': 'withdrawn_reason',
                         'Withdrawn Country': 'withdrawn_country',
                         'Withdrawn Class': 'withdrawn_class',
                         'Smiles': 'smiles'}, inplace=True)

    """
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL121', 'pubchem_cid'] = 77999
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL832', 'pubchem_cid'] = 5342
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL32', 'pubchem_cid'] = 152946
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL191', 'pubchem_cid'] = 3961
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL973', 'pubchem_cid'] = 54684141
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL101', 'pubchem_cid'] = 4781
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1428', 'pubchem_cid'] = 4497
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1450', 'pubchem_cid'] = 74989
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL196', 'pubchem_cid'] = 54670067
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL408', 'pubchem_cid'] = 5591
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1440', 'pubchem_cid'] = 54675776
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL413', 'pubchem_cid'] = 5284616
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL750', 'pubchem_cid'] = 11967800
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL33', 'pubchem_cid'] = 149096
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL752', 'pubchem_cid'] = 6083
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL502835', 'pubchem_cid'] = 135423438
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL595', 'pubchem_cid'] = 4829
    """

    # try search by synonyms
    list_of_features = ['molecule_type', 'structure_type', 'therapeutic_flag', 'indication_class']
    list_of_properties = ['alogp', 'aromatic_rings', 'cx_logd', 'cx_logp', 'cx_most_apka', 'cx_most_bpka', 'full_mwt',
                          'hba', 'hba_lipinski',
                          'hbd', 'hbd_lipinski', 'heavy_atoms', 'molecular_species', 'mw_freebase', 'mw_monoisotopic',
                          'num_lipinski_ro5_violations',
                          'num_ro5_violations', 'psa', 'qed_weighted', 'ro3_pass', 'rtb']

    for i in list_of_features:
        data[i] = 'missing'
    for i in list_of_properties:
        data[i] = 'missing'

    data['parent_chembl_id'] = 'missing'
    data['parent_smiles'] = 'missing'
    data['inchi_key'] = 'missing'
    data['parent_inchi_key'] = 'missing'

    molecule = new_client.molecule

    print('Updating chembl features')
    for mol in tqdm(list(data['chembl_id']), position=0, leave=True):
        try:
            res = molecule.get(mol)

            # add features
            if mol == res['molecule_chembl_id']:
                index = data.loc[data['chembl_id'] == mol].index.values[0]
                for feature in list_of_features:
                    if feature == 'atc_classifications' and not res[feature]:
                        data.at[index, feature] = None
                    elif feature == 'atc_classifications' and res[feature]:
                        data.at[index, feature] = '+'.join(res[feature])
                    else:
                        data.at[index, feature] = res[feature]

                # add properties
                try:
                    for prop in list_of_properties:
                        data.at[index, prop] = res['molecule_properties'][prop]
                except TypeError:
                    print('No properties for molecule {}'.format(mol))

                # try to add smiles if exists
                try:
                    if data.loc[data['chembl_id'] == mol]['smiles'].values[0] == 'missing':
                        smiles = res['molecule_structures']['canonical_smiles']
                        inchi_key = res['molecule_structures']['standard_inchi_key']
                        data.at[index, 'canonical_smiles'] = smiles
                        data.at[index, 'inchi_key'] = inchi_key
                    else:
                        inchi_key = res['molecule_structures']['standard_inchi_key']
                        data.at[index, 'inchi_key'] = inchi_key
                except TypeError:
                    print('No smiles for molecule {}'.format(mol))

                # add parent molecule
                try:
                    parent_mol = res['molecule_hierarchy']['parent_chembl_id']
                    data.at[index, 'parent_chembl_id'] = parent_mol
                except TypeError:
                    continue
                # check if parent has smiles
                try:
                    parent_res = molecule.get(parent_mol)
                    parent_smiles = parent_res['molecule_structures']['canonical_smiles']
                    parent_inchi_key = parent_res['molecule_structures']['standard_inchi_key']
                    data.at[index, 'parent_smiles'] = parent_smiles
                    data.at[index, 'parent_inchi_key'] = parent_inchi_key
                except:
                    print('No parent smiles for molecule {}'.format(mol))

        except Exception as e:
            print(e)
            print("Molecule {} doesn't exist".format(mol))

    print('Finished adding chembl features')

    data = data.loc[~((data['smiles'] == 'missing') & (data['parent_smiles'] == 'missing'))]

    print('Adding toxicity')
    toxic_carcinogenic = ['CHEMBL103', 'CHEMBL1200430', 'CHEMBL1200686', 'CHEMBL1200973',
                          'CHEMBL1201314', 'CHEMBL1201572', 'CHEMBL1201581', 'CHEMBL1201866',
                          'CHEMBL1220', 'CHEMBL135', 'CHEMBL137', 'CHEMBL1393', 'CHEMBL1456',
                          'CHEMBL1479', 'CHEMBL1511', 'CHEMBL152', 'CHEMBL1542', 'CHEMBL1554',
                          'CHEMBL160', 'CHEMBL1643', 'CHEMBL1651906', 'CHEMBL1742990', 'CHEMBL182',
                          'CHEMBL2107841', 'CHEMBL2107857', 'CHEMBL2108027', 'CHEMBL2108078',
                          'CHEMBL2108724', 'CHEMBL221959', 'CHEMBL269732', 'CHEMBL3301581',
                          'CHEMBL34259', 'CHEMBL413', 'CHEMBL414357', 'CHEMBL416', 'CHEMBL417',
                          'CHEMBL467', 'CHEMBL476', 'CHEMBL494753', 'CHEMBL515', 'CHEMBL53463',
                          'CHEMBL671', 'CHEMBL717', 'CHEMBL83', 'CHEMBL866']
    toxic_cardio = ['CHEMBL118', 'CHEMBL121', 'CHEMBL479', 'CHEMBL1020', 'CHEMBL103', 'CHEMBL1070',
                    'CHEMBL1071', 'CHEMBL1098', 'CHEMBL1108', 'CHEMBL1117', 'CHEMBL11359',
                    'CHEMBL1164729', 'CHEMBL1171837', 'CHEMBL1200430', 'CHEMBL1200973',
                    'CHEMBL1201336', 'CHEMBL1201577', 'CHEMBL1201824', 'CHEMBL1276308',
                    'CHEMBL1294', 'CHEMBL1297', 'CHEMBL13', 'CHEMBL134', 'CHEMBL135', 'CHEMBL139',
                    'CHEMBL1431', 'CHEMBL1511', 'CHEMBL154', 'CHEMBL154111', 'CHEMBL157101',
                    'CHEMBL15770', 'CHEMBL16', 'CHEMBL1655', 'CHEMBL1663', 'CHEMBL1743047',
                    'CHEMBL1760', 'CHEMBL178', 'CHEMBL2007641', 'CHEMBL2108676', 'CHEMBL24',
                    'CHEMBL24828', 'CHEMBL27', 'CHEMBL405', 'CHEMBL417', 'CHEMBL42', 'CHEMBL469',
                    'CHEMBL471', 'CHEMBL473', 'CHEMBL494753', 'CHEMBL499', 'CHEMBL517',
                    'CHEMBL521', 'CHEMBL527', 'CHEMBL533', 'CHEMBL53463', 'CHEMBL558', 'CHEMBL563',
                    'CHEMBL571', 'CHEMBL58', 'CHEMBL595', 'CHEMBL599', 'CHEMBL6', 'CHEMBL622',
                    'CHEMBL633', 'CHEMBL640', 'CHEMBL649', 'CHEMBL651', 'CHEMBL652', 'CHEMBL686',
                    'CHEMBL717', 'CHEMBL802', 'CHEMBL898',
                    ]
    toxic_dermis = ['CHEMBL108', 'CHEMBL1200431', 'CHEMBL1201589', 'CHEMBL1201824', 'CHEMBL1366',
                    'CHEMBL2108676', 'CHEMBL34259', 'CHEMBL428647', 'CHEMBL57', 'CHEMBL741',
                    'CHEMBL92',
                    ]
    toxic_gastro = ['CHEMBL118', 'CHEMBL1020', 'CHEMBL1070', 'CHEMBL1071', 'CHEMBL109',
                    'CHEMBL1110', 'CHEMBL11359', 'CHEMBL1201289', 'CHEMBL1297', 'CHEMBL1351',
                    'CHEMBL139', 'CHEMBL1447', 'CHEMBL1460', 'CHEMBL154', 'CHEMBL154111',
                    'CHEMBL15770', 'CHEMBL1651906', 'CHEMBL1742982', 'CHEMBL1753',
                    'CHEMBL2108676', 'CHEMBL2216870', 'CHEMBL3184512', 'CHEMBL34259', 'CHEMBL469',
                    'CHEMBL481', 'CHEMBL521', 'CHEMBL527', 'CHEMBL550348', 'CHEMBL563',
                    'CHEMBL571', 'CHEMBL599', 'CHEMBL6', 'CHEMBL622', 'CHEMBL64', 'CHEMBL686',
                    'CHEMBL803', 'CHEMBL898', 'CHEMBL92',
                    ]
    toxic_hema = ['CHEMBL1024', 'CHEMBL105', 'CHEMBL108', 'CHEMBL1094', 'CHEMBL1096882',
                  'CHEMBL1117', 'CHEMBL11359', 'CHEMBL1161681', 'CHEMBL1201281',
                  'CHEMBL1201314', 'CHEMBL1276308', 'CHEMBL129', 'CHEMBL1351', 'CHEMBL1366',
                  'CHEMBL152', 'CHEMBL1542', 'CHEMBL1643', 'CHEMBL1651906', 'CHEMBL170',
                  'CHEMBL178', 'CHEMBL182', 'CHEMBL34259', 'CHEMBL38', 'CHEMBL417', 'CHEMBL42',
                  'CHEMBL428647', 'CHEMBL44657', 'CHEMBL467', 'CHEMBL476', 'CHEMBL481',
                  'CHEMBL514', 'CHEMBL515', 'CHEMBL53463', 'CHEMBL553025', 'CHEMBL58',
                  'CHEMBL640', 'CHEMBL671', 'CHEMBL70927', 'CHEMBL803', 'CHEMBL820', 'CHEMBL833',
                  'CHEMBL84', 'CHEMBL852', 'CHEMBL92']
    toxic_hepa = ['CHEMBL1077', 'CHEMBL1201506', 'CHEMBL1324', 'CHEMBL109', 'CHEMBL1094',
                  'CHEMBL112', 'CHEMBL1131', 'CHEMBL1171837', 'CHEMBL1200436', 'CHEMBL1201187',
                  'CHEMBL1201288', 'CHEMBL1237044', 'CHEMBL129', 'CHEMBL1380', 'CHEMBL141',
                  'CHEMBL1460', 'CHEMBL1479', 'CHEMBL1518', 'CHEMBL1538', 'CHEMBL157101',
                  'CHEMBL1651906', 'CHEMBL2107825', 'CHEMBL2108611', 'CHEMBL2216870',
                  'CHEMBL222559', 'CHEMBL34259', 'CHEMBL344159', 'CHEMBL354541', 'CHEMBL461101',
                  'CHEMBL471', 'CHEMBL476', 'CHEMBL53463', 'CHEMBL535', 'CHEMBL550348',
                  'CHEMBL558', 'CHEMBL57', 'CHEMBL623', 'CHEMBL633', 'CHEMBL64', 'CHEMBL713',
                  'CHEMBL803', 'CHEMBL806', 'CHEMBL885', 'CHEMBL92', 'CHEMBL922', 'CHEMBL957',
                  'CHEMBL960', 'CHEMBL973', 'CHEMBL991']
    toxic_immune = ['CHEMBL11359', 'CHEMBL1201544', 'CHEMBL1201589', 'CHEMBL1201595',
                    'CHEMBL1201824', 'CHEMBL1201826', 'CHEMBL1201837', 'CHEMBL1237025',
                    'CHEMBL1351', 'CHEMBL1374379', 'CHEMBL1380', 'CHEMBL1456', 'CHEMBL1460',
                    'CHEMBL1542', 'CHEMBL1550', 'CHEMBL160', 'CHEMBL1742990', 'CHEMBL2108676',
                    'CHEMBL269732', 'CHEMBL3137342', 'CHEMBL3544926', 'CHEMBL413', 'CHEMBL414804',
                    'CHEMBL428647', 'CHEMBL469', 'CHEMBL57', 'CHEMBL643', 'CHEMBL852', 'CHEMBL866'
                                                                                       'CHEMBL92',
                    ]
    infections = ['CHEMBL479', 'CHEMBL1201289', 'CHEMBL1201572', 'CHEMBL1201581',
                  'CHEMBL1201607', 'CHEMBL1201828', 'CHEMBL1276308', 'CHEMBL1447', 'CHEMBL1456',
                  'CHEMBL160', 'CHEMBL1742990', 'CHEMBL1753', 'CHEMBL2107857', 'CHEMBL2216870',
                  'CHEMBL221959', 'CHEMBL269732', 'CHEMBL3184512', 'CHEMBL34259', 'CHEMBL413',
                  'CHEMBL553025', 'CHEMBL643', 'CHEMBL866',
                  ]
    toxic_metabolism = ['CHEMBL129', 'CHEMBL1380', 'CHEMBL141', 'CHEMBL1429', 'CHEMBL1431',
                        'CHEMBL1460', 'CHEMBL1743047', 'CHEMBL1760', 'CHEMBL585', 'CHEMBL713',
                        'CHEMBL922', 'CHEMBL945', 'CHEMBL991',
                        ]
    toxic_muscoskelet = ['CHEMBL129', 'CHEMBL32', 'CHEMBL33', 'CHEMBL4', 'CHEMBL640', 'CHEMBL701',
                         'CHEMBL717', 'CHEMBL8']

    toxic_nephro = ['CHEMBL1024', 'CHEMBL105', 'CHEMBL1072', 'CHEMBL11359', 'CHEMBL1200558',
                    'CHEMBL1366', 'CHEMBL152', 'CHEMBL160', 'CHEMBL1651906', 'CHEMBL1747',
                    'CHEMBL177', 'CHEMBL2105720', 'CHEMBL3039597', 'CHEMBL34259', 'CHEMBL3989769',
                    'CHEMBL507870', 'CHEMBL550348', 'CHEMBL666', 'CHEMBL922']

    toxic_neuro = ['CHEMBL118', 'CHEMBL646', 'CHEMBL1020', 'CHEMBL1024', 'CHEMBL103',
                   'CHEMBL1070', 'CHEMBL1071', 'CHEMBL1096882', 'CHEMBL1098', 'CHEMBL11359',
                   'CHEMBL1171837', 'CHEMBL1200430', 'CHEMBL1200973', 'CHEMBL1201112',
                   'CHEMBL1297', 'CHEMBL1342', 'CHEMBL135', 'CHEMBL139', 'CHEMBL1479',
                   'CHEMBL1511', 'CHEMBL154', 'CHEMBL154111', 'CHEMBL15770', 'CHEMBL1619',
                   'CHEMBL1747', 'CHEMBL177', 'CHEMBL1795072', 'CHEMBL3039597', 'CHEMBL3137342',
                   'CHEMBL32', 'CHEMBL33', 'CHEMBL3989738', 'CHEMBL3989769', 'CHEMBL4',
                   'CHEMBL407', 'CHEMBL42', 'CHEMBL469', 'CHEMBL494753', 'CHEMBL521', 'CHEMBL527',
                   'CHEMBL539697', 'CHEMBL563', 'CHEMBL571', 'CHEMBL599', 'CHEMBL6', 'CHEMBL622',
                   'CHEMBL666', 'CHEMBL686', 'CHEMBL701', 'CHEMBL717', 'CHEMBL8', 'CHEMBL81',
                   'CHEMBL83', 'CHEMBL86', 'CHEMBL898'
                   ]
    toxic_psyche = ['CHEMBL646', 'CHEMBL1089', 'CHEMBL11', 'CHEMBL1112', 'CHEMBL1113',
                    'CHEMBL1118', 'CHEMBL1175', 'CHEMBL117785', 'CHEMBL1200854', 'CHEMBL1200986',
                    'CHEMBL1201', 'CHEMBL1201168', 'CHEMBL1237021', 'CHEMBL14376', 'CHEMBL1508',
                    'CHEMBL1621', 'CHEMBL1628227', 'CHEMBL1795072', 'CHEMBL2104993',
                    'CHEMBL2105760', 'CHEMBL21731', 'CHEMBL259209', 'CHEMBL3989843', 'CHEMBL41',
                    'CHEMBL415', 'CHEMBL416956', 'CHEMBL439849', 'CHEMBL445', 'CHEMBL490',
                    'CHEMBL549', 'CHEMBL567', 'CHEMBL621', 'CHEMBL623', 'CHEMBL629', 'CHEMBL637',
                    'CHEMBL644', 'CHEMBL654', 'CHEMBL668', 'CHEMBL701', 'CHEMBL716', 'CHEMBL72',
                    'CHEMBL809', 'CHEMBL814', 'CHEMBL894', 'CHEMBL972', 'CHEMBL99946',
                    ]
    toxic_respiratory = ['CHEMBL656', 'CHEMBL103', 'CHEMBL11359', 'CHEMBL1200973', 'CHEMBL1201589',
                         'CHEMBL1201776', 'CHEMBL1201824', 'CHEMBL1201826', 'CHEMBL1237044',
                         'CHEMBL1263', 'CHEMBL1274', 'CHEMBL1342', 'CHEMBL135', 'CHEMBL1374379',
                         'CHEMBL1431', 'CHEMBL1511', 'CHEMBL1643', 'CHEMBL177', 'CHEMBL2108676',
                         'CHEMBL2216870', 'CHEMBL33986', 'CHEMBL34259', 'CHEMBL398707', 'CHEMBL403664',
                         'CHEMBL469', 'CHEMBL494753', 'CHEMBL511142', 'CHEMBL592', 'CHEMBL596',
                         'CHEMBL607', 'CHEMBL633', 'CHEMBL643', 'CHEMBL651', 'CHEMBL655', 'CHEMBL70',
                         'CHEMBL717', 'CHEMBL780', 'CHEMBL81', 'CHEMBL83', 'CHEMBL895', 'CHEMBL92',
                         'CHEMBL963']
    toxic_teratogen = ['CHEMBL1014', 'CHEMBL1017', 'CHEMBL1023', 'CHEMBL1069', 'CHEMBL109',
                       'CHEMBL1111', 'CHEMBL1131', 'CHEMBL1165', 'CHEMBL1168', 'CHEMBL1200692',
                       'CHEMBL1201314', 'CHEMBL1237', 'CHEMBL125', 'CHEMBL1456', 'CHEMBL1479',
                       'CHEMBL1513', 'CHEMBL1519', 'CHEMBL152', 'CHEMBL1542', 'CHEMBL1554',
                       'CHEMBL1560', 'CHEMBL1581', 'CHEMBL1592', 'CHEMBL1619', 'CHEMBL1639',
                       'CHEMBL1643', 'CHEMBL1747', 'CHEMBL182', 'CHEMBL191', 'CHEMBL2007641',
                       'CHEMBL2028661', 'CHEMBL2103873', 'CHEMBL2105737', 'CHEMBL2107834',
                       'CHEMBL262777', 'CHEMBL3039597', 'CHEMBL3039598', 'CHEMBL34259', 'CHEMBL38',
                       'CHEMBL473417', 'CHEMBL476', 'CHEMBL507870', 'CHEMBL515', 'CHEMBL547',
                       'CHEMBL577', 'CHEMBL578', 'CHEMBL606', 'CHEMBL717', 'CHEMBL813', 'CHEMBL838',
                       'CHEMBL866', 'CHEMBL957', 'CHEMBL960', 'CHEMBL973',
                       ]
    toxic_vascular = ['CHEMBL103', 'CHEMBL11359', 'CHEMBL1161681', 'CHEMBL1171837', 'CHEMBL1200430',
                      'CHEMBL1200973', 'CHEMBL1201202', 'CHEMBL1201212', 'CHEMBL1201336',
                      'CHEMBL1201438', 'CHEMBL1201476', 'CHEMBL1201589', 'CHEMBL1201651',
                      'CHEMBL1201772', 'CHEMBL1201824', 'CHEMBL1201826', 'CHEMBL134', 'CHEMBL135',
                      'CHEMBL136478', 'CHEMBL1374379', 'CHEMBL1431', 'CHEMBL1464', 'CHEMBL1479',
                      'CHEMBL1511', 'CHEMBL1550', 'CHEMBL16', 'CHEMBL160', 'CHEMBL1742982',
                      'CHEMBL1760', 'CHEMBL198362', 'CHEMBL2103827', 'CHEMBL2108676',
                      'CHEMBL221959', 'CHEMBL231779', 'CHEMBL34259', 'CHEMBL42', 'CHEMBL428647',
                      'CHEMBL469', 'CHEMBL481', 'CHEMBL494753', 'CHEMBL512351', 'CHEMBL53463',
                      'CHEMBL539697', 'CHEMBL550348', 'CHEMBL643', 'CHEMBL717', 'CHEMBL81',
                      'CHEMBL83', 'CHEMBL92']

    toxic_dict = {"toxic_carcinogenic": toxic_carcinogenic, "toxic_cardio": toxic_cardio, "toxic_dermis": toxic_dermis,
                  "toxic_gastro": toxic_gastro,
                  "toxic_hema": toxic_hema, "toxic_hepa": toxic_hepa, "toxic_immune": toxic_immune,
                  "infections": infections,
                  "toxic_metabolism": toxic_metabolism, "toxic_muscoskelet": toxic_muscoskelet,
                  "toxic_nephro": toxic_nephro,
                  "toxic_neuro": toxic_neuro, "toxic_psyche": toxic_psyche, "toxic_respiratory": toxic_respiratory,
                  "toxic_teratogen": toxic_teratogen, "toxic_vascular": toxic_vascular}

    data['chembl_tox'] = 'Safe'
    for effect in toxic_dict:
        for molecule in toxic_dict[effect]:
            try:
                old_label = data.loc[data['chembl_id'] == molecule]['chembl_tox'].values[0]
            except IndexError:
                print('Mol {} not in the set'.format(molecule))
            if old_label == 'Safe':
                data.loc[data['chembl_id'] == molecule, 'chembl_tox'] = effect
            else:
                data.loc[data['chembl_id'] == molecule, 'chembl_tox'] = '{}+{}'.format(old_label, effect)
    print('Finished adding toxicity')


    print('Adding mechanisms')
    data['action_type'] = 'missing'
    data['direct_interaction'] = 'missing'
    data['disease_efficacy'] = 'missing'
    data['molecular_mechanism'] = 'missing'
    data['target_chembl_id'] = 'missing'

    mechanism_list = ['action_type', 'direct_interaction', 'disease_efficacy', 'molecular_mechanism',
                      'target_chembl_id']

    for molecule in tqdm(list(data['chembl_id'])):
        for mechanism in new_client.mechanism.filter(molecule_chembl_id=molecule):
            if mechanism['max_phase'] == phase:
                for i in mechanism_list:
                    if data.loc[data['chembl_id'] == molecule][i].values[0] == 'missing':
                        data.loc[data['chembl_id'] == molecule, i] = mechanism[i]
                    else:
                        old_label = data.loc[data['chembl_id'] == molecule][i].values[0]
                        data.loc[data['chembl_id'] == molecule, i] = '{}+{}'.format(old_label, mechanism[i])

    print('Finished adding mechanisms')


    print('Adding pubchem cids by inchikey')
    data['pubchem_cid'] = 'missing'
    for i in tqdm(list(data['inchi_key'])):
        try:
            cid = pcp.get_cids("ZQHFZHPUZXNPMF-UHFFFAOYSA-N", "inchikey")[0]
            data.loc[data['inchi_key'] == i, 'pubchem_cid'] = cid
        except:
            print("Molecule not found")

    data['withdrawn'] = 0
    data.loc[data['availability_type'] == 'Withdrawn', 'withdrawn'] = 1
    data.to_csv(data_path / 'data/chembl_{}_full.csv'.format(phase))
    data[['chembl_id', 'pubchem_cid', 'smiles', 'parent_smiles', 'chembl_tox']].to_csv(
        data_path / 'data/chembl_{}_smiles.csv'.format(phase))

    """
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2109172', 'pubchem_cid'] = 134687958
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110554', 'pubchem_cid'] = 123827
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110563', 'pubchem_cid'] = 5311498
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110565', 'pubchem_cid'] = 123801
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110570', 'pubchem_cid'] = 49800005
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110572', 'pubchem_cid'] = 68726
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1201441', 'pubchem_cid'] = 216258
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110573', 'pubchem_cid'] = 119117
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2108758', 'pubchem_cid'] = 6850741
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4298110', 'pubchem_cid'] = 169535
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4301175', 'pubchem_cid'] = 14457
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4594730', 'pubchem_cid'] = 122706785
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2354773', 'pubchem_cid'] = 454937
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2218860', 'pubchem_cid'] = 6102852
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL508338', 'pubchem_cid'] = 16684434
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2108247', 'pubchem_cid'] = 105755
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2109152', 'pubchem_cid'] = 68245
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3544910', 'pubchem_cid'] = 158385
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4597194', 'pubchem_cid'] = 6328682
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3991035', 'pubchem_cid'] = 16683012
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1727605', 'pubchem_cid'] = 16683095
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3559672', 'pubchem_cid'] = 441975
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110551', 'pubchem_cid'] = 56840886
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110561', 'pubchem_cid'] = 374363820
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110564', 'pubchem_cid'] = 71587286
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2111018', 'pubchem_cid'] = 129626367
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2096629', 'pubchem_cid'] = 2724251
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2096647', 'pubchem_cid'] = 168924
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2107910', 'pubchem_cid'] = 24753271
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110575', 'pubchem_cid'] = 11430828
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2359370', 'pubchem_cid'] = 10898559
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL414804', 'pubchem_cid'] = 9887053
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3991241', 'pubchem_cid'] = 61300
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4594259', 'pubchem_cid'] = 18442052
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2108750', 'pubchem_cid'] = 122222
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4300153', 'pubchem_cid'] = 60143283
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4297207', 'pubchem_cid'] = 23725028
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3991443', 'pubchem_cid'] = 198008
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL3707208', 'pubchem_cid'] = 127024065
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1200571', 'pubchem_cid'] = 6918204
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1201186', 'pubchem_cid'] = 53297469
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL1201660', 'pubchem_cid'] = 8203
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL4594619', 'pubchem_cid'] = 22617237
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2108484', 'pubchem_cid'] = 6549
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2108906', 'pubchem_cid'] = 265028
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2097081', 'pubchem_cid'] = 11953895
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2111129', 'pubchem_cid'] = 23724985
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110914', 'pubchem_cid'] = 11353432
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110576', 'pubchem_cid'] = 56842070
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110566', 'pubchem_cid'] = 72972216
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110555', 'pubchem_cid'] = 11431716
    chembl.loc[chembl['Parent Molecule'] == 'CHEMBL2110560', 'pubchem_cid'] = 131634197
    chembl = chembl.loc[chembl['pubchem_cid'] != 0]
    """

    """
    drugbank = drugbank.loc[~drugbank['InChIKey'].isin(data['inchi_key'])]
    drugbank = drugbank[['DrugBank ID', 'InChIKey', 'Drug Groups', 'SMILES', 'Name']]
    drugbank.rename(columns={'DrugBank ID': 'drugbank_id',
                             'InChIKey': 'inchi_key',
                             'Drug Groups': 'drug_groups',
                             'SMILES': 'smiles',
                             'Name': 'synonyms'}, inplace=True)
    drugbank_list_of_features = ['molecule_type', 'structure_type', 'therapeutic_flag', 'molecule_chembl_id',
                                 'max_phase', 'atc_classifications',
                                 'chirality', 'prodrug', 'oral', 'parenteral', 'topical', 'black_box_warning',
                                 'availability_type', 'withdrawn_year',
                                 'withdrawn_reason', 'withdrawn_country', 'withdrawn_class', 'molecule_type',
                                 'structure_type', 'therapeutic_flag']
    drugbank_list_of_properties = ['alogp', 'aromatic_rings', 'cx_logd', 'cx_logp', 'cx_most_apka', 'cx_most_bpka',
                                   'full_mwt', 'hba', 'hba_lipinski',
                                   'hbd', 'hbd_lipinski', 'heavy_atoms', 'molecular_species', 'mw_freebase',
                                   'mw_monoisotopic', 'num_lipinski_ro5_violations',
                                   'num_ro5_violations', 'psa', 'qed_weighted', 'ro3_pass', 'rtb']
    for i in drugbank_list_of_features:
        drugbank[i] = 'missing'
    for i in list_of_properties:
        drugbank[i] = 'missing'

    drugbank['parent_chembl_id'] = 'missing'
    drugbank['parent_smiles'] = 'missing'
    drugbank['parent_inchi_key'] = 'missing'

    for mol in tqdm(list(drugbank['inchi_key']), position=0, leave=True):
        try:
            res = new_client.molecule.get(mol)

            # add features
            if mol == res['molecule_structures']['standard_inchi_key']:
                index = drugbank.loc[drugbank['inchi_key'] == mol].index.values[0]
                for feature in drugbank_list_of_features:
                    if feature == 'atc_classifications' and not res[feature]:
                        drugbank.at[index, feature] = None
                    elif feature == 'atc_classifications' and res[feature]:
                        drugbank.at[index, feature] = '+'.join(res[feature])
                    else:
                        drugbank.at[index, feature] = res[feature]

                # add properties
                try:
                    for prop in drugbank_list_of_properties:
                        drugbank.at[index, prop] = res['molecule_properties'][prop]
                except TypeError:
                    print('No properties for molecule {}'.format(mol))

                # try to add smiles if exists
                try:
                    if drugbank.loc[drugbank['inchi_key'] == mol]['smiles'].values[0] == 'missing':
                        smiles = res['molecule_structures']['canonical_smiles']
                        drugbank.at[index, 'canonical_smiles'] = smiles
                except TypeError:
                    print('No smiles for molecule {}'.format(mol))

                # add parent molecule
                try:
                    parent_mol = res['molecule_hierarchy']['parent_chembl_id']
                    drugbank.at[index, 'parent_chembl_id'] = parent_mol
                except TypeError:
                    continue
                # check if parent has smiles
                try:
                    parent_res = molecule.get(parent_mol)
                    parent_smiles = parent_res['molecule_structures']['canonical_smiles']
                    parent_inchi_key = parent_res['molecule_structures']['standard_inchi_key']
                    drugbank.at[index, 'parent_smiles'] = parent_smiles
                    drugbank.at[index, 'parent_inchi_key'] = parent_inchi_key
                except:
                    print('No parent smiles for molecule {}'.format(mol))

        except Exception as e:
            print(e)
            print("Molecule {} doesn't exist".format(mol))

    drugbank['action_type'] = 'missing'
    drugbank['direct_interaction'] = 'missing'
    drugbank['disease_efficacy'] = 'missing'
    drugbank['molecular_mechanism'] = 'missing'
    drugbank['target_chembl_id'] = 'missing'

    for molecule in tqdm(list(drugbank['molecule_chembl_id'])):
        for mechanism in new_client.mechanism.filter(molecule_chembl_id=molecule):
            if mechanism['max_phase'] == 4:
                for i in mechanism_list:
                    if drugbank.loc[drugbank['molecule_chembl_id'] == molecule][i].values[0] == 'missing':
                        drugbank.loc[drugbank['molecule_chembl_id'] == molecule, i] = mechanism[i]
                    else:
                        old_label = drugbank.loc[drugbank['molecule_chembl_id'] == molecule][i].values[0]
                        drugbank.loc[drugbank['molecule_chembl_id'] == molecule, i] = '{}+{}'.format(old_label,
                                                                                                     mechanism[i])

    drugank = drugbank.loc[(drugbank['max_phase'] == 'missing') | (drugbank['max_phase'] == 4)]
    drugbank['chembl_tox'] = 'Safe'
    for effect in toxic_dict:
        for molecule in toxic_dict[effect]:
            try:
                old_label = drugbank.loc[drugbank['molecule_chembl_id'] == molecule]['chembl_tox'].values[0]
            except IndexError:
                print('Mol {} not in the set'.format(molecule))
            if old_label == 'Safe':
                drugbank.loc[drugbank['molecule_chembl_id'] == molecule, 'chembl_tox'] = effect
            else:
                drugbank.loc[drugbank['molecule_chembl_id'] == molecule, 'chembl_tox'] = '{}+{}'.format(old_label,
                                                                                                        effect)
    drugbank['withdrawn'] = 0
    drugbank.loc[drugbank['drug_groups'].str.contains('withdrawn'), 'withdrawn'] = 1
    drugbank = drugbank.loc[~drugbank['smiles'].isna()]
    drugbank.to_csv('/home/dionizije/Documents/DAO/data/not_from_script/drugbank_{}_full.csv'.format(phase))
    drugbank[['molecule_chembl_id', 'drug_groups', 'smiles', 'parent_smiles', 'chembl_tox', 'withdrawn']].to_csv(
            '/home/dionizije/Documents/DAO/data/not_from_script/drugbank_{}_smiles.csv'.format(phase))
    """

if __name__ == "__main__":
    preprocess()
