import sys

sys.path.append("..")
import seaborn as sns
import networkx as nx
from matplotlib import pyplot, pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GNNExplainer
import pandas as pd
from torch import zeros, long
from IPython.core.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.utils import to_networkx
from pathlib import Path
import numpy as np
from src.smiles_only.EGConv_lightning import EGConvNet
from src.utils.data_func import smiles2graph_inference
from standardiser import standardise
from scipy.stats import zscore

root = Path(__file__).resolve().parents[1].absolute()


class DrugAttritionOracle:
    def __init__(self):
        for file in Path(root / 'production/egconv_production/production/checkpoint/').iterdir():
            checkpoint_file = file
        self.model = EGConvNet.load_from_checkpoint(
            str(checkpoint_file)
        )
        self.model.eval()
        self.approved_calibration = np.loadtxt(root / 'production/egconv_production/approved_calibration.csv') * 100
        self.withdrawn_calibration = np.loadtxt(root / 'production/egconv_production/withdrawn_calibration.csv') * 100

    def standardize_molecule(self, smiles):
        try:
            standardized = standardise.run(smiles)
        except Exception as e:
            print(e)
        return standardized

    def predict_probability(self, smiles):
        data = smiles2graph_inference(smiles)
        data.batch = zeros(data.num_nodes, dtype=long)
        output = self.model.forward(data.x, data.edge_index, data.batch).detach().cpu().numpy()[0][0]
        output = round((1 / (1 + np.exp(-output)) * 100), 2)
        return output

    def predict_class(self, smiles, threshold=53):
        data = smiles2graph_inference(smiles)
        data.batch = zeros(data.num_nodes, dtype=long)
        output = self.model.forward(data.x, data.edge_index, data.batch).detach().cpu().numpy()[0][0]
        output = round((1 / (1 + np.exp(-output)) * 100), 2)
        if output < threshold:
            return 'Approved'
        else:
            return 'Withdrawn'

    def conformal(self, smiles, significance=None):
        probability = self.predict_probability(smiles)
        approved_p_value = (np.searchsorted(self.approved_calibration, (100 - probability))) \
                           / (len(self.approved_calibration) + 1)
        withdrawn_p_value = (np.searchsorted(self.withdrawn_calibration, probability)) \
                            / (len(self.withdrawn_calibration) + 1)

        if significance != None:
            withdrawn_class = True if withdrawn_p_value > (1 - significance) else False
            approved_class = True if approved_p_value > (1 - significance) else False

            return pd.DataFrame({'Withdrawn class': withdrawn_class,
                                 'Approved class': approved_class}, index=[0])

        else:
            return pd.DataFrame({'Withdrawn class p-value': round(withdrawn_p_value, 2),
                                 'Approved class p-value': round(approved_p_value, 2)}, index=[0])

    def draw_molecule(self, smiles, bonds=None):
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 200)
        drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=bonds)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        return SVG(svg)

    def explain_molecule(self, smiles, epochs=300):
        features = ["is_boron", "is_carbon", "is_nitrogen", "is_oxygen", "is_flourine", "is_phosporus", "is_sulfur",
                    "is_chlorine", "is_bromine", "is_iodine", "is_other", "zero_Hs", "one_H", "two_Hs", "three_Hs",
                    "four_Hs", "is_s", "is_sp", "is_sp2", "is_sp3", "is_sp3d", "is_sp3d2", "unspecified_hybr",
                    "is_inring", "is_aromatic", "is_donor", "is_acceptor"]

        explainer = GNNExplainer(self.model, epochs=epochs)
        data = smiles2graph_inference(smiles)
        data.batch = zeros(data.num_nodes, dtype=long)
        node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)

        # node importance
        node_feat_mask = node_feat_mask.detach().numpy()
        node_feat_importance = pd.DataFrame(data=node_feat_mask[np.newaxis], columns=features, index=[0])


        # edge importance
        edge_mask = edge_mask.detach().numpy()
        edge_mask = abs(zscore(edge_mask))
        highlighted_edges = list((np.where(edge_mask > 1)[0]).astype(object))
        edge_index = data.edge_index.detach().cpu().numpy()

        # edge indices contain both direction so we need to drop one and save a copy
        final_edge = edge_index[:, ::2]
        normal = []  # first direction in edge mask
        reverse = []  # second direction
        for high in highlighted_edges:
            normal.append(list(edge_index[:, high]))
            reverse.append(list(edge_index[:, high][::-1]))

        # find bonds to higlight
        bonds_to_highlight = []
        for i in range(len(final_edge[0])):
            atom_1 = final_edge[0][i]
            atom_2 = final_edge[1][i]
            bond = [atom_1, atom_2]
            if bond in normal:
                bonds_to_highlight.append(i)
                continue
            if bond in reverse:
                bonds_to_highlight.append(i)

        """Atom symbols in feature importance
        mol = Chem.MolFromSmiles(r'{}'.format(smiles))
        atoms = []
        for i in bonds_to_highlight:
            atoms.append(mol.GetBonds()[i].GetBeginAtomIdx())
            atoms.append(mol.GetBonds()[i].GetEndAtomIdx())

        symbols = []
        for i in atoms:
            symbols.append(mol.GetAtoms()[i].GetSymbol())
        """

        return bonds_to_highlight, node_feat_importance

    def plot_feature_importance(self, node_feat_importance):
        sns.set(rc={"figure.figsize": (7, 8)})
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        return sns.barplot(data=node_feat_importance, orient='horizontal', ax=ax)

        """
        Old code for nx graphs
        viridis = pyplot.cm.get_cmap('YlOrRd')
        color_map = []
        for i in edge_mask:
            if i != 0:
                color_map.append('red')
            else:
                color_map.append('white')
        graph = Data(x=data.x, edge_index=data.edge_index, edge_attrs=color_map, node_labels=data.feature_names)
        g = to_networkx(graph, to_undirected=True, edge_attrs=['edge_attrs'], node_attrs=['node_labels'])
        for i in g.nodes:
            g.nodes[i]['atom'] = g.nodes[i]['node_labels'][0]
        pos = nx.planar_layout(g)
        pos = nx.spring_layout(g, pos=pos)
        widths = [x * 10 for x in edge_mask]
        labels = nx.get_node_attributes(g, 'atom')
        return nx.draw(g, pos=pos, labels=labels, width=widths, edge_color=color_map)
        """
