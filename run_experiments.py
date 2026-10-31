"""Run reproducible test-label leakage experiments for all three GNN models.

IMPORTANT: these are deliberately contaminated diagnostics, not unbiased
performance estimates on unseen transactions.

Run from the repository root:
    python run_experiments.py --fractions 0.25 --seeds 42

For a larger sweep:
    python run_experiments.py --fractions 0.10 0.25 0.50 0.75 1.00 --seeds 42
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CONFIGS = [
    "configs/sage_baseline.yaml",
    "configs/gat_baseline.yaml",
    "configs/camo_gat.yaml",
]


def fraction_tag(fraction):
    """Readable, collision-resistant fraction label for run directory names."""
    percent = f"{fraction * 100:.4f}".rstrip("0").rstrip(".")
    return "leak" + percent.replace(".", "p") + "pct"


def expected_run_name(config_file, seed, fraction, run_id):
    return f"{Path(config_file).stem}_{fraction_tag(fraction)}_seed{seed}_{run_id}"


def run_experiment(config_file, seed, fraction, run_id):
    run_name = expected_run_name(config_file, seed, fraction, run_id)
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
    leakage = metrics.get("leakage_experiment") or {}
    exposed_metrics = leakage.get("exposed_subset_metrics") or {}
    unexposed_metrics = leakage.get("unexposed_subset_metrics") or {}
    return {
        "model": Path(config_file).stem,
        "fraction": fraction,
        "seed": seed,
        "run_id": run_id,
        "run_name": run_name,
        "test_pr_auc": metrics.get("pr_auc"),
        "test_roc_auc": metrics.get("roc_auc"),
        "val_pr_auc": (metrics.get("val") or {}).get("pr_auc"),
        "exposed_pr_auc": exposed_metrics.get("pr_auc"),
        "unexposed_pr_auc": unexposed_metrics.get("pr_auc"),
        "exposed_count": leakage.get("exposed_test_nodes", 0),
        "test_count": leakage.get("total_test_nodes", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run explicitly labelled test-label leakage experiments."
    )
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25],
        help="Fractions of test labels to expose, e.g. 0.1 0.25 0.5 1.0"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--models", nargs="+",
        choices=["sage_baseline", "gat_baseline", "camo_gat"],
        default=["sage_baseline", "gat_baseline", "camo_gat"],
    )
    args = parser.parse_args()

    if not args.fractions:
        parser.error("Provide at least one leakage fraction.")
    if any(f < 0 or f > 1 for f in args.fractions):
        parser.error("Every fraction must be between 0.0 and 1.0 inclusive.")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("Duplicate fractions are not allowed.")
    if not args.seeds:
        parser.error("Provide at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds are not allowed.")
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate models are not allowed.")

    chosen_configs = [c for c in CONFIGS if Path(c).stem in args.models]
    missing_configs = [c for c in chosen_configs if not os.path.isfile(c)]
    if missing_configs:
        parser.error("Missing configuration file(s): " + ", ".join(missing_configs))

    # One ID ties all runs in this sweep together and prevents a repeated
    # command from silently overwriting earlier experiment outputs.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    planned_names = [
        expected_run_name(config, seed, fraction, run_id)
        for fraction in args.fractions
        for config in chosen_configs
        for seed in args.seeds
    ]
    collisions = [
        os.path.join("experiments", name)
        for name in planned_names
        if os.path.exists(os.path.join("experiments", name))
    ]
    if collisions:
        parser.error(
            "Refusing to overwrite existing experiment directory/directories: "
            + ", ".join(collisions)
            + ". Wait for a new timestamp or choose a different run setup."
        )

    rows = []
    for fraction in args.fractions:
        for config in chosen_configs:
            for seed in args.seeds:
                rows.append(run_experiment(config, seed, fraction, run_id))

    os.makedirs("report", exist_ok=True)
    json_path = os.path.join("report", f"leakage_experiment_results_{run_id}.json")
    md_path = os.path.join("report", f"leakage_experiment_results_{run_id}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Test-label leakage experiment results\n\n")
        f.write(
            "> WARNING: Selected test labels were used during training. These numbers "
            "are contaminated diagnostics and must not be presented as unseen-data "
            "generalization performance.\n\n"
        )
        f.write(
            "| Model config | Test-label fraction exposed | Seed | Validation PR-AUC | "
            "All-test PR-AUC | Exposed-subset PR-AUC | Unexposed-subset PR-AUC |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            def fmt(value):
                return "N/A" if value is None else f"{value:.4f}"
            f.write(
                f"| {row['model']} | {row['fraction']:.2%} | {row['seed']} | "
                f"{fmt(row['val_pr_auc'])} | {fmt(row['test_pr_auc'])} | "
                f"{fmt(row['exposed_pr_auc'])} | {fmt(row['unexposed_pr_auc'])} |\n"
            )

    print(f"\nFinished. Results saved to {md_path} and {json_path}.")
    print("Remember: leakage results are not valid estimates of generalization.")


if __name__ == "__main__":
    main()
