import os
import subprocess
import json
import numpy as np

def run_experiment(config_file, seed):
    """Runs a single training experiment as a subprocess."""
    run_name = f"{os.path.basename(config_file).replace('.yaml', '')}_seed{seed}"
    cmd = [
        "python", "src/train.py",
        "--config", config_file,
        "--seed", str(seed),
        "--run_name", run_name
    ]
    print(f"\n--- Running: {' '.join(cmd)} ---")
    subprocess.run(cmd, check=True)
    
    # Evaluate model
    exp_dir = os.path.join("experiments", run_name)
    eval_cmd = [
        "python", "src/evaluate.py",
        "--exp_dir", exp_dir
    ]
    print(f"\n--- Evaluating: {' '.join(eval_cmd)} ---")
    subprocess.run(eval_cmd, check=True)
    
    # Read test metrics
    metrics_path = os.path.join(exp_dir, "test_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

def main():
    configs = [
        "configs/sage_baseline.yaml",
        "configs/gat_baseline.yaml",
        "configs/camo_gat.yaml"
    ]
    seeds = [42, 123, 456]
    
    results = {}
    
    for config in configs:
        config_name = os.path.basename(config).replace('.yaml', '')
        results[config_name] = {'pr_auc': [], 'roc_auc': []}
        for seed in seeds:
            metrics = run_experiment(config, seed)
            if metrics:
                results[config_name]['pr_auc'].append(metrics.get("pr_auc", 0))
                results[config_name]['roc_auc'].append(metrics.get("roc_auc", 0))
                
    # Generate Markdown Table
    os.makedirs("report", exist_ok=True)
    report_path = os.path.join("report", "results.md")
    
    with open(report_path, "w") as f:
        f.write("# Phase 8 Experiment Results\n\n")
        f.write("| Model Configuration | PR-AUC (Mean ± Std) | ROC-AUC (Mean ± Std) |\n")
        f.write("|---------------------|---------------------|----------------------|\n")
        
        for config_name, metrics in results.items():
            pr_aucs = metrics['pr_auc']
            roc_aucs = metrics['roc_auc']
            
            if not pr_aucs:
                f.write(f"| {config_name} | N/A | N/A |\n")
                continue
                
            pr_mean = np.mean(pr_aucs)
            # ddof=1: sample standard deviation. With only 3 seeds, the
            # population std (ddof=0) noticeably understates the spread.
            pr_std = np.std(pr_aucs, ddof=1)
            roc_mean = np.mean(roc_aucs)
            roc_std = np.std(roc_aucs, ddof=1)
            
            f.write(f"| {config_name} | {pr_mean:.4f} ± {pr_std:.4f} | {roc_mean:.4f} ± {roc_std:.4f} |\n")

    print(f"\nAll experiments complete! Results saved to {report_path}.")

if __name__ == "__main__":
    main()
