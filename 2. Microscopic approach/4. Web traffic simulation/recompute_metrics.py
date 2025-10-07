#!/usr/bin/env python
"""
Recompute streaming metrics for a single (model, alpha, beta) run, with:
 - final-frame checkpoint
 - ARI on non-omitted frames
 - normalized overclustering (%)
 - counts of true and predicted clusters
 - count of omitted frames
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    v_measure_score,
    pair_confusion_matrix,
)

def _safe_pair_f1(y_true, y_pred):
    cm = pair_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def compute_progress_metrics(df: pd.DataFrame, step: int = 100) -> pd.DataFrame:
    """
    Build a DataFrame with core metrics every <step> frames + final frame:
      - ARI_global
      - ARI_non_omitted
      - V_measure_global
      - pair_f1
      - accuracy_global
      - accuracy_non_omitted
      - f1_weighted
      - n_omitted_frames
      - pct_omitted_frames
      - n_true_clusters
      - n_pred_clusters
      - overclustering_pct
    Requires columns: 'frame.time', 'true_label', 'dataset_id'
    """
    # sort by time
    df_sorted = df.sort_values("frame.time").reset_index(drop=True)
    y_true_full = df_sorted["true_label"].to_numpy()
    y_pred_full = df_sorted["dataset_id"].to_numpy()
    omitted_id = y_pred_full.max()

    # prepare checkpoints: multiples of step + final index
    n_total = len(df_sorted)
    checkpoints = list(range(step, n_total + 1, step))
    if checkpoints[-1] != n_total:
        checkpoints.append(n_total)

    records = []
    for n in checkpoints:
        y_t = y_true_full[:n]
        y_p = y_pred_full[:n]

        # basic global metrics
        rec = {
            "frames": n,
            "ARI_global": adjusted_rand_score(y_t, y_p),
            "V_measure_global": v_measure_score(y_t, y_p),
            "pair_f1": _safe_pair_f1(y_t, y_p),
        }

        # mask out omitted for certain metrics
        mask_no_om = y_p != omitted_id
        y_t_no = y_t[mask_no_om]
        y_p_no = y_p[mask_no_om]

        # ARI on non-omitted frames
        rec["ARI_non_omitted"] = (
            adjusted_rand_score(y_t_no, y_p_no) if len(y_t_no) else 0.0
        )

        # accuracy & weighted F1
        rec["accuracy_global"] = (y_p == y_t).mean()
        rec["accuracy_non_omitted"] = (
            accuracy_score(y_t_no, y_p_no) if len(y_t_no) else 0.0
        )
        rec["f1_weighted"] = (
            f1_score(y_t_no, y_p_no, average="weighted") if len(y_t_no) else 0.0
        )

        # count omitted frames
        omitted_count = int(np.sum(y_p == omitted_id))
        rec["n_omitted_frames"] = omitted_count
        rec["pct_omitted_frames"] = (omitted_count / n * 100.0) if n > 0 else 0.0

        # cluster counts
        n_true = len(np.unique(y_t))
        n_pred = len(np.unique(y_p_no))
        rec["n_true_clusters"] = n_true
        rec["n_pred_clusters"] = n_pred

        # normalized overclustering %: pred non-omitted vs true
        rec["overclustering_pct"] = (
            100.0 * (n_pred - n_true) / n_true if n_true > 0 else 0.0
        )

        records.append(rec)

    return pd.DataFrame(records)


def smart_read(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        return pd.read_pickle(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")


def main():
    ap = argparse.ArgumentParser(
        description="Re-generate metrics_progress.csv with extra stats."
    )
    ap.add_argument("run_dir",
                    help="Folder containing assignments_pred.(pkl|csv|...) and true_label")
    ap.add_argument("--step", type=int, default=100,
                    help="Frames per checkpoint (default: 100)")
    ap.add_argument("--outfile", default="metrics_progress.csv",
                    help="Output CSV filename (inside run_dir)")
    args = ap.parse_args()

    # locate prediction file
    candidates = [f"assignments_pred{x}" for x in [".pkl", ".csv", ".feather", ".parquet"]]
    pred_path = next(
        (os.path.join(args.run_dir, f) for f in candidates if os.path.isfile(os.path.join(args.run_dir, f))),
        None
    )
    if pred_path is None:
        raise FileNotFoundError(f"No assignments_pred.* in {args.run_dir}")

    df = smart_read(pred_path)
    for col in ("dataset_id", "true_label", "frame.time"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {pred_path}")

    prog = compute_progress_metrics(df, step=args.step)

    out_csv = os.path.join(args.run_dir, args.outfile)
    prog.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(prog)} checkpoints")

if __name__ == "__main__":
    main()
