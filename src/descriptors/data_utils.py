"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondDir as BD
from rdkit.Chem.rdchem import HybridizationType

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))


def load_data_from_df(dataset_path, add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'
    feature_path = dataset_path.replace('.csv', f'{feat_stamp}.p')
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x_all, y_all = pickle.load(open(feature_path, "rb"))
        return x_all, y_all

    data_df = pd.read_csv(dataset_path)

    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    x_all, y_all = load_data_from_smiles(data_x, data_y, add_dummy_node=add_dummy_node,
                                         one_hot_formal_charge=one_hot_formal_charge)
    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x_all, y_all), open(feature_path, "wb"))

    return x_all, y_all


def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all, y_all


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list = [], [], []
    labels = []

    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))

    return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list, labels)]


def construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)

def construct_dataset_multilabel(x_all, y_all, y2_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.
        y2_all (list): Second label

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], data[2], i)
              for i, data in enumerate(zip(x_all, y_all, y2_all))]
    return MolDataset(output)


def construct_loader(x, y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        y (list): A list of the corresponding labels.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=shuffle)
    return loader

def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

def smiles2graph(smiles, withdrawn_col):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    #smiles = smiles
    #y = withdrawn_col

    y = smiles[withdrawn_col]
    smiles = smiles['smiles']
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

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = torch.Tensor(edge_index).long()
    graph['node_feat'] = torch.Tensor(x)
    graph['y'] = torch.Tensor([y])
    graph['feature_names'] = names

    return Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=graph['y'], feature_names=names)


def smiles2graph_descriptors(data, withdrawn_col, descriptors_from):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    smiles = data['smiles']
    y = data[withdrawn_col]
    descriptors = data[descriptors_from:]
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

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = torch.Tensor(edge_index).long()
    graph['node_feat'] = torch.Tensor(x)
    graph['y'] = torch.Tensor([y])
    graph['feature_names'] = names
    graph['descriptors'] = torch.Tensor([descriptors])

    return Data(x=graph['node_feat'], edge_index=graph['edge_index'], y=graph['y'], descriptors=graph['descriptors'],
                feature_names=names)
