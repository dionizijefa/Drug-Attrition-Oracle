from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo as BS
from rdkit.Chem.rdchem import BondDir as BD
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from pathlib import Path
import torch
from rdkit.Chem import AllChem
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

class MolDataset(Dataset):
    def __init__(self, pt_file):
        self.data = torch.load(pt_file)
        self.labels = [x[4] for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class MoleculesDataset(Dataset):
    def __init__(self, csv_file, withdrawn_col, descriptors_from_col):
        self.data = pd.read_csv(csv_file, index_col=0)
        self.labels = self.data[withdrawn_col]
        self.descriptors_from_col = descriptors_from_col

        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.stereo = {BS.STEREONONE: 0, BS.STEREOANY: 1, BS.STEREOZ: 2,
                  BS.STEREOE: 3, BS.STEREOCIS: 4, BS.STEREOTRANS: 5}
        self.direction = {BD.NONE: 0, BD.BEGINWEDGE: 1, BD.BEGINDASH: 2,
                     BD.ENDDOWNRIGHT: 3, BD.ENDUPRIGHT: 4, BD.EITHERDOUBLE: 5,
                     BD.UNKNOWN: 6}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def process(self):
        data_list = []
        for i, molecule in tqdm(self.data.iterrows()):
            name = molecule['DrugBank ID']
            smiles = molecule['full_smiles']
            mol = Chem.MolFromSmiles(smiles)
            label = molecule['label']

            """ Features """
            atomic_number = []
            aromatic = []
            donor = []
            acceptor = []
            s = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []

            for atom in mol.GetAtoms():
                # type_idx.append(self.types[atom.GetSymbol()])
                fdef_name = Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'
                factory = ChemicalFeatures.BuildFeatureFactory(str(fdef_name))
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                donor.append(0)
                acceptor.append(0)
                s.append(1 if hybridization == HybridizationType.S
                         else 0)
                sp.append(1 if hybridization == HybridizationType.SP
                          else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2
                           else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3
                           else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D
                            else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2
                             else 0)

                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

            feats = factory.GetFeaturesForMol(mol)
            for j in range(0, len(feats)):
                if feats[j].GetFamily() == 'Donor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        donor[k] = 1
                elif feats[j].GetFamily() == 'Acceptor':
                    node_list = feats[j].GetAtomIds()
                    for k in node_list:
                        acceptor[k] = 1

            x = torch.tensor([atomic_number,
                              acceptor,
                              donor,
                              aromatic,
                              s, sp, sp2, sp3, sp3d, sp3d2,
                              num_hs],
                             dtype=torch.float).t().contiguous()

            row, col, bond_idx, bond_stereo, bond_dir = [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                bond_stereo += 2 * [self.stereo[bond.GetStereo()]]
                bond_dir += 2 * [self.direction[bond.GetBondDir()]]
                # 2* list, because the bonds are defined 2 times, start -> end,
                # and end -> start
            edge_index = torch.tensor([row, col], dtype=torch.long)

            """ Create distance matrix """
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            conf = mol.GetConformer()
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())])
            dist_matrix = pairwise_distances(pos_matrix)

            adj_matrix = np.eye(mol.GetNumAtoms())
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom().GetIdx()
                end_atom = bond.GetEndAtom().GetIdx()
                adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

            descriptors = molecule[self.descriptors_from_col].values.astype(float)
            data_list.append([x, adj_matrix, dist_matrix, descriptors, label, name])

        return data_list

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
    adjacency_list, distance_list, features_list, descriptors_list = [], [], [], []
    labels = []
    names = []

    max_size = 0
    for molecule in batch:
        if molecule[1].shape[0] > max_size:
            max_size = molecule[1].shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule[1], (max_size, max_size)))
        distance_list.append(pad_array(molecule[2], (max_size, max_size)))
        features_list.append(pad_array(molecule[0], (max_size, molecule[0].shape[1])))
        descriptors_list.append(molecule[3])
        labels.append(molecule[4])
        names.append(molecule[5])

    data = [torch.Tensor(features) for features in (
        adjacency_list, features_list, distance_list, descriptors_list, labels
    )]

    return [data, names]





