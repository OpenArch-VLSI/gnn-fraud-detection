"""Run controlled test-label leakage experiments for all three GNN models.

IMPORTANT: outputs are deliberately contaminated diagnostics, not unbiased
performance estimates on unseen transactions.
Run from the repository root:
    python run_experiments.py --fractions 0.25 --seeds 42
For a larger sweep, use --fractions 0.1 0.25 0.5 0.75 1.0.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CONFIGS = [
    "configs/sage_baseline.yaml",
    "configs/gat_baseline.yaml",
    "configs/camo_gat.yaml",
]


def run_experiment(config_file, seed, fraction):
    config_stem = Path(config_file).stem
    fraction_tag = f"leak{int(round(fraction * 100)):03d}"
    run_name = f"{config_stem}_{fraction_tag}_seed{seed}"
    cmd = [
        sys.executable, "src/train.py",
        "--config", config_file,
        "--seed", str(seed),
        "--run_name", run_name,
        "--leakage_test_fraction", str(fraction),
    ]
    print("\n--- Training: " + " ".join(cmd) + " ---", flush=True)
    subprocess.run(cmd, check=True)

    exp_dir = os.path.join("experiments", run_name)
    eval_cmd = [sys.executable, "src/evaluate.py", "--exp_dir", exp_dir]
    print("\n--- Evaluating: " + " ".join(eval_cmd) + " ---", flush=True)
    subprocess.run(eval_cmd, check=True)

    metrics_path = os.path.join(exp_dir, "test_results.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "model": config_stem,
        "fraction": fraction,
        "seed": seed,
        "run_name": run_name,
        "test_pr_auc": metrics.get("pr_auc"),
        "test_roc_auc": metrics.get("roc_auc"),
        "val_pr_auc": (metrics.get("val") or {}).get("pr_auc"),
        "exposed_pr_auc": ((metrics.get("leakage_experiment") or {}).get(
            "exposed_subset_metrics") or {}).get("pr_auc"),
        "unexposed_pr_auc": ((metrics.get("leakage_experiment") or {}).get(
            "unexposed_subset_metrics") or {}).get("pr_auc"),
        "exposed_count": (metrics.get("leakage_experiment") or {}).get(
            "exposed_test_nodes", 0),
        "test_count": (metrics.get("leakage_experiment") or {}).get(
            "total_test_nodes", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25],
                        help="Fractions of test labels to expose, e.g. 0.1 0.25 0.5 1.0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--models", nargs="+", choices=["sage_baseline", "gat_baseline", "camo_gat"],
                        default=["sage_baseline", "gat_baseline", "camo_gat"])
    args = parser.parse_args()

    if any(f < 0 or f > 1 for f in args.fractions):
        parser.error("Every fraction must be between 0 and 1 inclusive.")

    chosen_configs = [c for c in CONFIGS if Path(c).stem in args.models]
    rows = []
    for fraction in args.fractions:
        for config in chosen_configs:
            for seed in args.seeds:
                rows.append(run_experiment(config, seed, fraction))

    os.makedirs("report", exist_ok=True)
    json_path = os.path.join("report", "leakage_experiment_results.json")
    md_path = os.path.join("report", "leakage_experiment_results.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Test-label leakage experiment results\n\n")
        f.write("> WARNING: Selected test labels were used during training. These numbers are "
                "contaminated diagnostics and must not be presented as unseen-data performance.\n\n")
        f.write("| Model config | Test-label fraction exposed | Seed | Validation PR-AUC | All-test PR-AUC | Exposed-subset PR-AUC | Unexposed-subset PR-AUC |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            def fmt(v):
                return "N/A" if v is None else f"{v:.4f}"
            f.write(f"| {r['model']} | {r['fraction']:.0%} | {r['seed']} | "
                    f"{fmt(r['val_pr_auc'])} | {fmt(r['test_pr_auc'])} | "
                    f"{fmt(r['exposed_pr_auc'])} | {fmt(r['unexposed_pr_auc'])} |\n")

    print(f"\nFinished. Results saved to {md_path} and {json_path}.")
    print("Remember: leakage results are not valid estimates of generalization.")


if __name__ == "__main__":
    main()
