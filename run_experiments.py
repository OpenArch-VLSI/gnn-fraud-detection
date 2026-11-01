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


def expected_run_name(config_file, seed, fraction, run_id, contaminated=False):
    if contaminated:
        return f"CONTAMINATED_{Path(config_file).stem}_mix_train_test_seed{seed}_{run_id}"
    return f"{Path(config_file).stem}_{fraction_tag(fraction)}_seed{seed}_{run_id}"


def run_experiment(config_file, seed, fraction, run_id, contaminated=False):
    run_name = expected_run_name(config_file, seed, fraction, run_id, contaminated)
    cmd = [
        sys.executable, "src/train.py",
        "--config", config_file,
        "--seed", str(seed),
        "--run_name", run_name,
        "--leakage_test_fraction", str(fraction),
    ]
    if contaminated:
        cmd += ["--contaminated_same_label_eval"]
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
        "epoch_selected": (metrics.get("best_training_epoch") or {}).get("epoch"),
        "contaminated_validation_pr_auc": (metrics.get("val") or {}).get("pr_auc"),
        "contaminated_same_test_pr_auc": metrics.get("pr_auc"),
        "warning_status": metrics.get("evaluation_status"),
    }


def write_reports(rows, report_dir, run_id, contaminated):
    """Write reports whose warning language matches the selected experiment mode."""
    os.makedirs(report_dir, exist_ok=True)
    prefix = "contaminated_same_label_benchmark" if contaminated else "leakage_experiment_results"
    json_path = os.path.join(report_dir, f"{prefix}_{run_id}.json")
    md_path = os.path.join(report_dir, f"{prefix}_{run_id}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        payload = ({
            "title": "Contaminated train+test / same-test benchmark -- NOT GENERALIZATION",
            "evaluation_status": "CONTAMINATED_SAME_LABEL_EVALUATION",
            "rows": rows,
        } if contaminated else rows)
        json.dump(payload, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        if contaminated:
            f.write("# Contaminated train+test / same-test benchmark -- NOT GENERALIZATION\n\n")
            f.write("**Evaluation status:** `CONTAMINATED_SAME_LABEL_EVALUATION`\n\n")
            f.write(
                "> WARNING: CONTAMINATED SAME-LABEL EVALUATION. All original test nodes and "
                "labels were included in training, and evaluation was performed on that same "
                "test split. These metrics are not valid estimates of generalization or "
                "deployment performance.\n\n"
            )
        else:
            f.write("# Test-label leakage experiment results\n\n")
            f.write(
                "> Leakage diagnostic: each row states the configured fraction of original test "
                "labels exposed during training. A fraction of 0% is clean; nonzero fractions "
                "are partial leakage diagnostics and are not unseen-data generalization estimates.\n\n"
            )
        if contaminated:
            f.write("| Model | Seed | Epoch selected | Validation AP (validation labels not used as training targets) | Contaminated same-test PR-AUC | ROC-AUC | Warning status |\n")
            f.write("|---|---:|---:|---:|---:|---:|---|\n")
        else:
            f.write("| Model config | Test-label fraction exposed | Seed | Validation PR-AUC | "
                    "All-test PR-AUC | Exposed-subset PR-AUC | Unexposed-subset PR-AUC |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            def fmt(value):
                return "N/A" if value is None else f"{value:.4f}"
            if contaminated:
                f.write(f"| {row['model']} | {row['seed']} | {row['epoch_selected']} | "
                        f"{fmt(row['contaminated_validation_pr_auc'])} | "
                        f"{fmt(row['contaminated_same_test_pr_auc'])} | "
                        f"{fmt(row['test_roc_auc'])} | {row['warning_status']} |\n")
            else:
                f.write(f"| {row['model']} | {row['fraction']:.2%} | {row['seed']} | "
                        f"{fmt(row['val_pr_auc'])} | {fmt(row['test_pr_auc'])} | "
                        f"{fmt(row['exposed_pr_auc'])} | {fmt(row['unexposed_pr_auc'])} |\n")
    return md_path, json_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run explicitly labelled test-label leakage experiments."
    )
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25],
        help="Fractions of test labels to expose, e.g. 0.1 0.25 0.5 1.0"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--contaminated_same_label_eval", action="store_true",
                        help="Run the explicit train+test / same-test contaminated benchmark.")
    parser.add_argument(
        "--models", nargs="+",
        choices=["sage_baseline", "gat_baseline", "camo_gat"],
        default=["sage_baseline", "gat_baseline", "camo_gat"],
    )
    return parser


def validate_args(parser, args):

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
    if args.contaminated_same_label_eval:
        args.fractions = [1.0]
    elif any(f == 1.0 for f in args.fractions):
        parser.error(
            "--fractions 1.0 requires --contaminated_same_label_eval because it "
            "exposes every original test label to training.")
    return args


def main():
    parser = build_parser()
    args = validate_args(parser, parser.parse_args())

    chosen_configs = [c for c in CONFIGS if Path(c).stem in args.models]
    missing_configs = [c for c in chosen_configs if not os.path.isfile(c)]
    if missing_configs:
        parser.error("Missing configuration file(s): " + ", ".join(missing_configs))

    # One ID ties all runs in this sweep together and prevents a repeated
    # command from silently overwriting earlier experiment outputs.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    planned_names = [
        expected_run_name(config, seed, fraction, run_id, args.contaminated_same_label_eval)
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
                rows.append(run_experiment(config, seed, fraction, run_id,
                                           args.contaminated_same_label_eval))

    md_path, json_path = write_reports(
        rows, "report", run_id, args.contaminated_same_label_eval)

    print(f"\nFinished. Results saved to {md_path} and {json_path}.")
    print("Remember: leakage results are not valid estimates of generalization.")


if __name__ == "__main__":
    main()
