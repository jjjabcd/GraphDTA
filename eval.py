import argparse
import pandas as pd
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from src.metric import (
    get_rmse,
    get_pcc,
    get_cindex,
    get_rm2,
    get_mse
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphDTA results")
    parser.add_argument("--task_name", required=True, choices=["Kd", "Ki"])
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    print(f"[Info] Loading predictions from {args.pred_csv}")
    df = pd.read_csv(args.pred_csv)

    # Target column
    label_col = "pKd" if args.task_name == "Kd" else "pKi"

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {args.pred_csv}")

    y_true = df[label_col].values
    y_pred = df["predicted_value"].values

    # Compute metrics
    rmse = get_rmse(y_true, y_pred)
    mse = get_mse(y_true, y_pred)
    pcc = get_pcc(y_true, y_pred)
    ci = get_cindex(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)

    # Save metrics
    metrics = pd.DataFrame([{
        "RMSE": rmse,
        "MSE": mse,
        "PCC": pcc,
        "CI": ci,
        "RM2": rm2
    }])

    metrics.to_csv(args.out_csv, index=False)
    print(f"[Info] Saved metrics → {args.out_csv}")


if __name__ == "__main__":
    main()
