import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as GeomData
from torch_geometric.data import DataLoader

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# internal modules
from src.create_data import smile_to_graph, protein_sequence_to_index
from src.models.ginconv import GINConvNet
from src.models.gcn import GCNNet
from src.models.gat import GATNet
from src.models.gat_gcn import GAT_GCN

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ======================================================
# Utility
# ======================================================

def get_device(cuda_index: int):
    if cuda_index < 0 or not torch.cuda.is_available():
        print("[Info] Using CPU")
        return torch.device("cpu")
    print(f"[Info] Using GPU: cuda:{cuda_index}")
    return torch.device(f"cuda:{cuda_index}")


MODEL_FACTORY = {
    "GINConvNet": GINConvNet,
    "GCNNet": GCNNet,
    "GATNet": GATNet,
    "GAT_GCN": GAT_GCN,
}


# ======================================================
# Dataset conversion (CSV → PyG)
# ======================================================

def build_smile_graph(smiles):
    """
    Build SMILES → graph dictionary (unique only)
    """
    unique = sorted(set(smiles))
    table = {}
    print(f"[Info] Building SMILES graph for {len(unique)} unique SMILES...")

    for smi in unique:
        c_size, features, edge_index = smile_to_graph(smi)
        table[smi] = (c_size, features, edge_index)

    return table


def build_test_dataset(csv_path, task_name, smile_graph):
    df = pd.read_csv(csv_path)
    data_list = []

    for _, row in df.iterrows():
        smi = row["SMILES"]
        seq = row["FASTA"]

        c_size, features, edge_index = smile_graph[smi]
        x = torch.tensor(features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        prot_idx = protein_sequence_to_index(seq)
        target = torch.LongTensor([prot_idx])
        c_size_tensor = torch.LongTensor([c_size])

        g = GeomData(x=x, edge_index=edge_index)
        g.target = target
        g.c_size = c_size_tensor
        data_list.append(g)

    return df, data_list


# ======================================================
# Prediction logic
# ======================================================

def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)                 # shape: [N,1]
            preds.append(out.view(-1).cpu())

    return torch.cat(preds).numpy()


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="GraphDTA Prediction")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_name", default="GINConvNet",
                        choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--model_path", required=True,
                        help="best_model.pt path")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cuda", type=int, default=0)

    args = parser.parse_args()

    device = get_device(args.cuda)

    # Load test set
    df_test = pd.read_csv(args.test_csv)
    smiles = list(df_test["SMILES"])

    # Build SMILES graph
    smile_graph = build_smile_graph(smiles)

    # Convert test CSV → PyG dataset
    df_test, data_list = build_test_dataset(
        args.test_csv, args.task_name, smile_graph
    )

    loader = DataLoader(data_list, batch_size=512, shuffle=False)

    # Load model
    ModelClass = MODEL_FACTORY[args.model_name]
    model = ModelClass().to(device)

    print(f"[Info] Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Prediction
    preds = predict(model, loader, device)

    df_test["predicted_value"] = preds
    df_test.to_csv(args.out_csv, index=False)

    print(f"[Info] Saved predictions → {args.out_csv}")


if __name__ == "__main__":
    main()
