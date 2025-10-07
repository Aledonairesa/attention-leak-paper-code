from pathlib import Path
import re
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate          

# ---------------------------------------------------------------------------
# 1. Gather CSVs
# ---------------------------------------------------------------------------

ROOT = Path("metrics")
INTERVAL_RE = re.compile(r"(\d+(?:\.\d+)?)s")          # e.g. "10.0s"
FILE_RE     = re.compile(r"^(.*?)_features_(.*?)_metrics\.csv$",
                         re.IGNORECASE)                # model & features

records = []

for interval_dir in ROOT.iterdir():
    if not interval_dir.is_dir():
        continue

    m_int = INTERVAL_RE.fullmatch(interval_dir.name)
    if not m_int:
        continue                                      # skip unrelated dirs

    interval_val = float(m_int.group(1))              # 10.0, 5.0 ...

    for csv_path in interval_dir.glob("*.csv"):
        m_file = FILE_RE.fullmatch(csv_path.name)
        if not m_file:
            continue

        model, features = m_file.group(1), m_file.group(2)

        df_csv = pd.read_csv(csv_path)

        # Handle two possible layouts:
        #   a) 1-row CSV already containing Model / Features columns
        #   b) metrics only – then we insert them
        if "Model" not in df_csv.columns:
            df_csv.insert(0, "Model",    model)
        if "Features" not in df_csv.columns:
            df_csv.insert(1, "Features", features)

        df_csv.insert(2, "Interval", interval_val)
        records.append(df_csv)

# One big DataFrame
if not records:
    raise RuntimeError("No matching CSV files were found!")

df_all = pd.concat(records, ignore_index=True)

# Consistent ordering of columns
metric_cols = [c for c in df_all.columns
               if c not in ("Model", "Features", "Interval")]
df_all = df_all[["Model", "Features", "Interval", *metric_cols]]

# ---------------------------------------------------------------------------
# 2. Pretty-print table -> txt
# ---------------------------------------------------------------------------

out_dir = ROOT / "combined_metrics"
out_dir.mkdir(exist_ok=True)

table_txt = tabulate(
    df_all,
    headers="keys",
    floatfmt=".5g",
    tablefmt="github"        
)

(out_dir / "all_metrics_table.txt").write_text(table_txt, encoding="utf-8")

print(f"Wrote pretty table with {len(df_all)} rows → {out_dir/'all_metrics_table.txt'}")

# ---------------------------------------------------------------------------
# 3. Per-metric plots  (REPLACED / UPDATED)
# ---------------------------------------------------------------------------

# The four feature-set names
FEATURE_SETS = [
    "features_all",
    "features_corr_var",
    "features_corr_var_PCA40",
    "features_RFE",
]

# Draw left-to-right: 1 s -> 10 s
interval_order = [1.0, 2.0, 5.0, 10.0]
tick_labels    = ["1", "2", "5", "10"]    

# Metric-specific y-axis limits
ylims = {
    "MAE":  (None, 80),     # (bottom, top)  None = let Matplotlib pick
    "MBE":  (None, 80),
    "R2":   (-3,   None),   # only a lower bound
    "RMSE": (None, 100),
}

for metric in metric_cols:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(FEATURE_SETS),
        figsize=(4 * len(FEATURE_SETS), 4),
        sharex=True,
        sharey=True,
    )

    # --- Plot each feature set ------------------------------------------------
    for ax, feat in zip(axes, FEATURE_SETS):
        sub = df_all[df_all["Features"] == feat]

        for model, g in sub.groupby("Model"):
            # Re-index so every interval is present 
            y = (
                g.set_index("Interval")
                 .reindex(interval_order)[metric]
            )
            ax.plot(interval_order, y, marker="o", label=model)

        ax.set_title(feat.replace("features_", ""))   # cleaner subplot titles
        ax.set_xlabel("Interval (s)")
        ax.grid(True, ls=":")

    # --- Common y-label & y-limits -------------------------------------------
    axes[0].set_ylabel(metric)
    if metric in ylims:
        bottom, top = ylims[metric]
        for ax in axes:
            ax.set_ylim(bottom=bottom, top=top)

    # --- X-axis ticks ---------------------------------------------------------
    for ax in axes:
        ax.set_xticks(interval_order)
        ax.set_xticklabels(tick_labels)

    # --- Legend: put it *below* the panels and leave space --------------------
    handles, labels = axes[-1].get_legend_handles_labels()
    ncols = min(len(labels), 4) if labels else 1

    # Add extra bottom margin before tight_layout so nothing overlaps
    fig.subplots_adjust(bottom=0.25, top=0.82)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),   # centred below the sub-plots
        ncol=ncols,
        frameon=False,
    )

    # --- Title ­+ save --------------------------------------------------------
    fig.suptitle(f"{metric} vs. interval per model (separated by feature set)")
    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved {out_path}")
