import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data as GeomData
from torch_geometric.data import DataLoader

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# project modules
from src.metric import get_mse, get_rmse, get_pcc, get_cindex, get_rm2
from src.create_data import smile_to_graph, protein_sequence_to_index

from src.models.ginconv import GINConvNet
from src.models.gcn import GCNNet
from src.models.gat import GATNet
from src.models.gat_gcn import GAT_GCN

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ============================================================
# utils
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cuda_index: int):
    if cuda_index < 0 or not torch.cuda.is_available():
        print("[Info] Using CPU")
        return torch.device("cpu")
    device = torch.device(f"cuda:{cuda_index}")
    print(f"[Info] Using GPU: cuda:{cuda_index}")
    return device


MODEL_FACTORY = {
    "GINConvNet": GINConvNet,
    "GCNNet": GCNNet,
    "GATNet": GATNet,
    "GAT_GCN": GAT_GCN,
}


# ============================================================
# dataset builders
# ============================================================

def build_smile_graph(smiles_list):
    unique_smiles = sorted(set(smiles_list))
    print(f"[Info] Building smile_graph for {len(unique_smiles)} unique SMILES...")

    smile_graph = {}
    for smi in tqdm(unique_smiles, desc="Processing SMILES", ncols=100):
        c_size, features, edge_index = smile_to_graph(smi)
        smile_graph[smi] = (c_size, features, edge_index)

    return smile_graph


def build_dataset_from_csv(csv_path, task_name, smile_graph):
    df = pd.read_csv(csv_path)
    print(f"[Info] Loading CSV: {csv_path} (n={len(df)})")

    label_col = "pKd" if task_name == "Kd" else "pKi"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    dataset = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset", ncols=100):
        smi = row["SMILES"]
        seq = row["FASTA"]
        y_val = float(row[label_col])

        c_size, features, edge_index = smile_graph[smi]
        x = torch.tensor(features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        prot_idx = protein_sequence_to_index(seq)
        target = torch.LongTensor([prot_idx])
        y = torch.tensor([y_val], dtype=torch.float32)
        c_size_tensor = torch.LongTensor([c_size])

        g = GeomData(x=x, edge_index=edge_index, y=y)
        g.target = target
        g.c_size = c_size_tensor

        dataset.append(g)

    return dataset


def build_dataloaders(train_csv, val_csv, task_name, batch_size):
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    all_smiles = list(df_tr["SMILES"]) + list(df_va["SMILES"])
    smile_graph = build_smile_graph(all_smiles)

    train_data = build_dataset_from_csv(train_csv, task_name, smile_graph)
    val_data = build_dataset_from_csv(val_csv, task_name, smile_graph)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ============================================================
# evaluation
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y

            preds.append(out.view(-1).cpu())
            trues.append(y.view(-1).cpu())

    if len(preds) == 0:
        return {"MSE": 0.0}

    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()

    mse = float(get_mse(y_true, y_pred))
    rmse = float(get_rmse(y_true, y_pred))
    pcc = float(get_pcc(y_true, y_pred))
    ci = float(get_cindex(y_true, y_pred))
    rm2 = float(get_rm2(y_true, y_pred))

    return {"MSE": mse, "RMSE": rmse, "PCC": pcc, "CI": ci, "RM2": rm2}


# ============================================================
# main training loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train GraphDTA from CSV (train + val only)")
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--model_name", default="GINConvNet",
                        choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    set_seed(args.seed)
    device = get_device(args.cuda)

    # =====================================
    # Build dataloaders
    # =====================================
    train_loader, val_loader = build_dataloaders(
        args.train_csv, args.val_csv, args.task_name, args.batch_size
    )

    # =====================================
    # Build model
    # =====================================
    ModelClass = MODEL_FACTORY[args.model_name]
    model = ModelClass().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mse = float("inf")
    patience_cnt = 0
    history = []

    print(f"[Info] Training {args.model_name}")

    # tqdm progress bar
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Training", ncols=100)

    for epoch in epoch_iter:
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, val_loader, device)

        epoch_iter.set_postfix({
            "train_loss": avg_train_loss,
            "val_mse": val_metrics["MSE"]
        })

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics
        })

        # early stopping
        if val_metrics["MSE"] < best_val_mse:
            best_val_mse = val_metrics["MSE"]
            patience_cnt = 0

            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

            with open(os.path.join(args.out_dir, "best_metrics.txt"), "w") as f:
                for k, v in val_metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"epoch: {epoch}\n")

        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f"[Info] Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    print("[Info] Training completed.")


if __name__ == "__main__":
    main()