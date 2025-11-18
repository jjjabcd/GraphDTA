import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
import networkx as nx

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ALLOWED_ATOMS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
    'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 
    'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
    'Pb', 'Unknown'
]

ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ALLOWED_HS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ALLOWED_VALENCE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def atom_features(atom):
    """
    RDKit atom → feature vector
    """
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ALLOWED_ATOMS) +
        one_of_k_encoding(atom.GetDegree(), ALLOWED_DEGREES) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), ALLOWED_HS) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), ALLOWED_VALENCE) +
        [atom.GetIsAromatic()],
        dtype=np.float32
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def protein_sequence_to_index(seq, max_len=max_seq_len):
    """
    Convert protein sequence to index sequence using seq_dict.
    Unknown characters → 0
    Pad to max_len.
    """
    seq = str(seq).upper()
    encoded = []

    for ch in seq[:max_len]:
        encoded.append(seq_dict.get(ch, 0))  # unknown → 0

    # padding
    while len(encoded) < max_len:
        encoded.append(0)

    return np.array(encoded, dtype=np.int64)


# ============================================================
# Dataset (UPDATED PROTEIN ENCODING)
# ============================================================

class DTADataset(Dataset):
    """
    CSV → graph + protein index input → torch Dataset
    """
    def __init__(self, df, task_name="Kd", max_prot_len=max_seq_len):
        self.df = df
        self.task = task_name
        self.max_len = max_prot_len

        self.smiles = df["SMILES"].values
        self.proteins = df["FASTA"].values

        self.labels = df["pKd"].values if task_name == "Kd" else df["pKi"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        seq = self.proteins[idx]
        label = float(self.labels[idx])

        # drug graph (original)
        c_size, features, edge_index = smile_to_graph(smi)

        # protein encoding (NEW)
        prot_idx = protein_sequence_to_index(seq, self.max_len)

        return {
            "c_size": torch.tensor(c_size, dtype=torch.long),
            "node_feat": torch.tensor(features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "protein": torch.tensor(prot_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }

def load_datasets_from_csv(train_csv, val_csv, test_csv, task="Kd", max_prot_len=1000):
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    train_dataset = DTADataset(df_tr, task, max_prot_len)
    val_dataset = DTADataset(df_va, task, max_prot_len)
    test_dataset = DTADataset(df_te, task, max_prot_len)

    return train_dataset, val_dataset, test_dataset