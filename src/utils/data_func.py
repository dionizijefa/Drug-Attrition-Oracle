import sys
sys.path.append("../..")
from pathlib import Path
import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import HybridizationType, ChemicalFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader, Data
from torch import Tensor, cat
from src.utils.descriptors_list import rdkit_descriptors, alvadesc_descriptors, padel_descriptors_10pct, \
    toxprint_descriptors_10pct, feature_selected, alvadesc_100, ozren_selected

fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))


def cross_val(data, withdrawn_col, batch_size, seed, n_splits=5, **kwargs):
    """Don't split rest of the splits on scaffolds"""
    cv_splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    loaders = []
    for k, (train_index, test_index) in enumerate(
            cv_splitter.split(data, data[withdrawn_col])
    ):

        test = data.iloc[test_index]
        test_data_list = []
        for index, row in test.iterrows():
            test_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))
        test_loader = DataLoader(test_data_list, num_workers=0, batch_size=batch_size)

        train_set = data.iloc[train_index]

        train, val = train_test_split(
            train_set,
            test_size=0.15,
            stratify=train_set[withdrawn_col],
            shuffle=True,
            random_state=seed)

        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))

        # balanced sampling of the minority class
        withdrawn = train[withdrawn_col].value_counts()[1]
        approved = train[withdrawn_col].value_counts()[0]
        class_sample_count = [approved, withdrawn]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train[withdrawn_col].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

        loaders.append([train_loader, val_loader, test_loader])

    return loaders


def scaffold_cross_val(data, withdrawn_col, batch_size, seed, **kwargs):
    """Validation split on scaffolds"""
    cv_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    scaffolds_df = pd.DataFrame(data['scaffolds'].value_counts())
    unique_scaffolds = list(scaffolds_df.loc[scaffolds_df['scaffolds'] == 1].index)
    data_unique_scaffolds = data.loc[data['scaffolds'].isin(unique_scaffolds)]

    loaders = []
    for k, (train_index, test_index) in enumerate(
            cv_splitter.split(data_unique_scaffolds, data_unique_scaffolds[withdrawn_col])
    ):

        test = data_unique_scaffolds.iloc[test_index]
        test_data_list = []
        for index, row in test.iterrows():
            test_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))
        test_loader = DataLoader(test_data_list, num_workers=0, batch_size=batch_size)

        train_set = data_unique_scaffolds.iloc[train_index]

        train, val = train_test_split(
            train_set,
            test_size=0.15,
            stratify=train_set[withdrawn_col],
            shuffle=True,
            random_state=seed)

        train = pd.concat([train, data.loc[~data['scaffolds'].isin(unique_scaffolds)]])
        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))

        # balanced sampling of the minority class
        withdrawn = train[withdrawn_col].value_counts()[1]
        approved = train[withdrawn_col].value_counts()[0]
        class_sample_count = [approved, withdrawn]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train[withdrawn_col].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler)

        val_data_list = []
        for index, row in val.iterrows():
            val_data_list.append(smiles2graph(row, withdrawn_col, **kwargs))
        val_loader = DataLoader(val_data_list, num_workers=0, batch_size=batch_size)

        loaders.append([train_loader, val_loader, test_loader])

    return loaders


def create_loader(data, withdrawn_col, batch_size, **kwargs):
    data_list = []
    for index, row in data.iterrows():
        data_list.append(smiles2graph(row, withdrawn_col, **kwargs))

    data_loader = DataLoader(data_list, num_workers=0, batch_size=batch_size)

    return data_loader


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


def smiles2graph(data, withdrawn_col, **kwargs):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # smiles = smiles
    # y = withdrawn_col

    y = data[withdrawn_col]
    smiles = data['standardized_smiles']
    if 'descriptors' in kwargs:
        descriptors = data
        if kwargs['descriptors'] == 'alvadesc':
            descriptors = descriptors[alvadesc_descriptors].iloc[:100]  # keep only descriptors from the list
        elif kwargs['descriptors'] == 'padel1560':
            descriptors = descriptors[padel_descriptors_10pct]
        elif kwargs['descriptors'] == 'toxprint':
            descriptors = descriptors[toxprint_descriptors_10pct]
        elif kwargs['descriptors'] == 'rdkit':
            descriptors = descriptors[rdkit_descriptors]
        elif kwargs['descriptors'] == 'ozren_selected':
            descriptors = descriptors[ozren_selected]
        else:
            descriptors = descriptors[feature_selected]

    mol = Chem.MolFromSmiles(smiles)

    # atoms
    donor = []
    acceptor = []
    features = []
    names = []
    donor_string = []

    for atom in mol.GetAtoms():
        atom_feature_names = []
        atom_features = []
        atom_features += one_hot_vector(
            atom.GetAtomicNum(),
            [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
        )

        atom_feature_names.append(atom.GetSymbol())
        atom_features += one_hot_vector(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )
        atom_feature_names.append(atom.GetTotalNumHs())
        atom_features += one_hot_vector(
            atom.GetHybridization(),
            [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
             HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED]
        )
        atom_feature_names.append(atom.GetHybridization().__str__())

        atom_features.append(atom.IsInRing())
        atom_features.append(atom.GetIsAromatic())

        if atom.GetIsAromatic() == 1:
            atom_feature_names.append('Aromatic')
        else:
            atom_feature_names.append('Non-aromatic')

        if atom.IsInRing() == 1:
            atom_feature_names.append('Is in ring')
        else:
            atom_feature_names.append('Not in ring')

        donor.append(0)
        acceptor.append(0)

        donor_string.append('Not a donor or acceptor')

        atom_features = np.array(atom_features, dtype=int)
        atom_feature_names = np.array(atom_feature_names, dtype=object)
        features.append(atom_features)
        names.append(atom_feature_names)

    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
        if feats[j].GetFamily() == 'Donor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                donor[k] = 0
                donor_string[k] = 'Donor'
        elif feats[j].GetFamily() == 'Acceptor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                acceptor[k] = 1
                donor_string[k] = 'Acceptor'

    features = np.array(features, dtype=int)
    donor = np.array(donor, dtype=int)
    donor = donor[..., np.newaxis]
    acceptor = np.array(acceptor, dtype=int).transpose()
    acceptor = acceptor[..., np.newaxis]
    x = np.append(features, donor, axis=1)
    x = np.append(x, acceptor, axis=1)

    donor_string = np.array(donor_string, dtype=object)
    donor_string = donor_string[..., np.newaxis]

    names = np.array(names, dtype=object)
    names = np.append(names, donor_string, axis=1)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # add edges in both directions
            edges_list.append((i, j))
            edges_list.append((j, i))

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = Tensor(edge_index).long()
    graph['node_feat'] = Tensor(x)
    graph['y'] = Tensor([y])
    graph['feature_names'] = names

    if 'descriptors' in kwargs:
        graph['descriptors'] = Tensor([descriptors.astype(float)])
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=graph['y'], feature_names=names,
                    descriptors=graph['descriptors'])
    else:
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=graph['y'], feature_names=names)


def calibrate(model, calib_loader, descriptors=False):

    calib_probabilities = []
    targets = []
    if descriptors:
        for i in calib_loader:
            calib_probabilities.append(model.forward(i.x, i.edge_index, i.batch, i.descriptors))
            targets.append(i.y)
    else:
        for i in calib_loader:
            calib_probabilities.append(model.forward(i.x, i.edge_index, i.batch))
            targets.append(i.y)
    calib_probabilities = np.array(cat(calib_probabilities).detach().cpu().numpy().flatten())
    calib_probabilities = 1 / (1 + np.exp(-calib_probabilities))
    targets = np.array(cat(targets).detach().cpu().numpy().flatten())
    calibration_df = pd.DataFrame({'probabilities': calib_probabilities, 'class': targets})
    approved_probabilities = calibration_df.loc[calibration_df['class'] == 0]['probabilities'].values
    approved_probabilities = 1 - approved_probabilities
    withdrawn_probabilities = calibration_df.loc[calibration_df['class'] == 1]['probabilities'].values
    approved_probabilities = np.sort(approved_probabilities)
    withdrawn_probabilities = np.sort(withdrawn_probabilities)

    return approved_probabilities, withdrawn_probabilities


def conformal_prediction(test_loader, model, approved_probabilities, withdrawn_probabilities, descriptors=False):
    test_probabilities = []
    test_targets = []
    if descriptors:
        for i in test_loader:
            test_probabilities.append(model.forward(i.x, i.edge_index, i.batch, i.descriptors))
            test_targets.append(i.y)
    else:
        for i in test_loader:
            test_probabilities.append(model.forward(i.x, i.edge_index, i.batch))
            test_targets.append(i.y)
    test_probabilities = np.array(cat(test_probabilities).detach().cpu().numpy().flatten())
    test_probabilities = 1 / (1 + np.exp(-test_probabilities))

    p_values_approved = []
    p_values_withdrawn = []
    for i in test_probabilities:
        p_values_approved.append(
            (np.searchsorted(approved_probabilities, i) / (len(approved_probabilities) + 1))
        )
        p_values_withdrawn.append(
            (np.searchsorted(withdrawn_probabilities, i) / (len(withdrawn_probabilities) + 1))
        )

    return p_values_approved, p_values_withdrawn, test_probabilities

def smiles2graph_inference(data, **kwargs):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(data)
    descriptors = kwargs['descriptors']
    descriptors = descriptors[alvadesc_100].values[0]

    # atoms
    donor = []
    acceptor = []
    features = []
    names = []
    donor_string = []

    for atom in mol.GetAtoms():
        atom_feature_names = []
        atom_features = []
        atom_features += one_hot_vector(
            atom.GetAtomicNum(),
            [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
        )

        atom_feature_names.append(atom.GetSymbol())
        atom_features += one_hot_vector(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4]
        )
        atom_feature_names.append(atom.GetTotalNumHs())
        atom_features += one_hot_vector(
            atom.GetHybridization(),
            [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
             HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.UNSPECIFIED]
        )
        atom_feature_names.append(atom.GetHybridization().__str__())

        atom_features.append(atom.IsInRing())
        atom_features.append(atom.GetIsAromatic())

        if atom.GetIsAromatic() == 1:
            atom_feature_names.append('Aromatic')
        else:
            atom_feature_names.append('Non-aromatic')

        if atom.IsInRing() == 1:
            atom_feature_names.append('Is in ring')
        else:
            atom_feature_names.append('Not in ring')

        donor.append(0)
        acceptor.append(0)

        donor_string.append('Not a donor or acceptor')

        atom_features = np.array(atom_features, dtype=int)
        atom_feature_names = np.array(atom_feature_names, dtype=object)
        features.append(atom_features)
        names.append(atom_feature_names)

    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
        if feats[j].GetFamily() == 'Donor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                donor[k] = 0
                donor_string[k] = 'Donor'
        elif feats[j].GetFamily() == 'Acceptor':
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                acceptor[k] = 1
                donor_string[k] = 'Acceptor'

    features = np.array(features, dtype=int)
    donor = np.array(donor, dtype=int)
    donor = donor[..., np.newaxis]
    acceptor = np.array(acceptor, dtype=int).transpose()
    acceptor = acceptor[..., np.newaxis]
    x = np.append(features, donor, axis=1)
    x = np.append(x, acceptor, axis=1)

    donor_string = np.array(donor_string, dtype=object)
    donor_string = donor_string[..., np.newaxis]

    names = np.array(names, dtype=object)
    names = np.append(names, donor_string, axis=1)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # add edges in both directions
            edges_list.append((i, j))
            edges_list.append((j, i))

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = Tensor(edge_index).long()
    graph['node_feat'] = Tensor(x)
    graph['feature_names'] = names

    if 'descriptors' in kwargs:
        graph['descriptors'] = Tensor([descriptors.astype(float)])
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], feature_names=names,
                    descriptors=graph['descriptors'])
    else:
        return Data(x=graph['node_feat'], edge_index=graph['edge_index'], feature_names=names)
