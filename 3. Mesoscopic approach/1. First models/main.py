#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate evaluation and optional metrics/plots generation"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to run (e.g. KNN_50 RF XGB RF_CLS TCN)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="List of feature labels to use (e.g. features_all features_corr_var features_RFE)"
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        type=float,
        required=True,
        help="One or more split intervals (floats), e.g. 5.0 2.0"
    )
    parser.add_argument(
        "--metrics",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Whether to run metrics/plots (True/False)"
    )
    return parser.parse_args()

def run_evaluate(models, features, interval):
    cmd = [
        sys.executable, "evaluate.py",
        "--models", *models,
        "--features", *features,
        "--interval", str(interval)
    ]
    print(f"\n Running evaluation: interval={interval}s")
    subprocess.run(cmd, check=True)

def run_metrics_and_plots(model, feature, interval):
    cmd = [
        sys.executable, "metrics_and_plots.py",
        "--model", model,
        "--features", feature,
        "--interval", str(interval)
    ]
    print(f"   Generating metrics & plots for {model} + {feature} @ {interval}s")
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()

    # Ensure scripts exist
    for script in ("evaluate.py", "metrics_and_plots.py"):
        if not Path(script).exists():
            print(f"Error: {script} not found in current directory.", file=sys.stderr)
            sys.exit(1)

    # Loop over each interval
    for interval in args.intervals:
        # 1) run evaluation
        run_evaluate(args.models, args.features, interval)

        # 2) optionally run metrics & plots
        if args.metrics:
            for model in args.models:
                for feat in args.features:
                    # base feature
                    run_metrics_and_plots(model, feat, interval)
                    # special PCA variant
                    if feat == "features_corr_var":
                        pca_feat = f"{feat}_PCA40"
                        run_metrics_and_plots(model, pca_feat, interval)

    print("\n All done.")

if __name__ == "__main__":
    main()
