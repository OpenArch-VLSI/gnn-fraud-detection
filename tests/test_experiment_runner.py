import json

import pytest

from run_experiments import (build_parser, expected_run_name, fraction_tag,
                             validate_args, write_reports)


def test_fraction_tags_preserve_fraction_precision():
    assert fraction_tag(0.25) == "leak25pct"
    assert fraction_tag(0.125) == "leak12p5pct"
    assert fraction_tag(1.0) == "leak100pct"
    assert fraction_tag(0.125) != fraction_tag(0.13)


def test_run_names_include_fraction_seed_and_sweep_id():
    name = expected_run_name("configs/sage_baseline.yaml", 42, 0.25, "20261031_190500")
    assert name == "sage_baseline_leak25pct_seed42_20261031_190500"


def test_runner_rejects_unmarked_full_test_label_exposure(capsys):
    parser = build_parser()
    args = parser.parse_args(['--fractions', '1.0'])
    with pytest.raises(SystemExit) as error:
        validate_args(parser, args)
    assert error.value.code == 2
    assert '--contaminated_same_label_eval' in capsys.readouterr().err


def test_runner_accepts_partial_leakage_and_explicit_contaminated_full_mix():
    parser = build_parser()
    partial = validate_args(parser, parser.parse_args(['--fractions', '0.10', '0.25', '0.75']))
    assert partial.fractions == [0.10, 0.25, 0.75]
    contaminated = validate_args(parser, parser.parse_args([
        '--fractions', '1.0', '--contaminated_same_label_eval']))
    assert contaminated.fractions == [1.0]


def test_clean_report_describes_fraction_without_full_contamination_claim(tmp_path):
    rows = [{
        'model': 'sage_baseline', 'fraction': 0.25, 'seed': 42,
        'val_pr_auc': 0.1, 'test_pr_auc': 0.2,
        'exposed_pr_auc': 0.3, 'unexposed_pr_auc': 0.15,
    }]
    md_path, json_path = write_reports(rows, str(tmp_path), 'clean', contaminated=False)
    report = (tmp_path / 'leakage_experiment_results_clean.md').read_text(encoding='utf-8')
    assert md_path.endswith('leakage_experiment_results_clean.md')
    assert 'configured fraction of original test labels' in report
    assert '25.00%' in report
    assert 'All original test nodes and labels were included in training' not in report
    assert 'CONTAMINATED SAME-LABEL EVALUATION' not in report
    assert json.loads((tmp_path / 'leakage_experiment_results_clean.json').read_text()) == rows
    assert json_path.endswith('leakage_experiment_results_clean.json')


def test_contaminated_report_has_status_and_same_label_warning(tmp_path):
    rows = [{
        'model': 'sage_baseline', 'seed': 42, 'epoch_selected': 2,
        'contaminated_validation_pr_auc': 0.1,
        'contaminated_same_test_pr_auc': 0.2,
        'test_roc_auc': 0.8,
        'warning_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
    }]
    md_path, json_path = write_reports(rows, str(tmp_path), 'contam', contaminated=True)
    report = (tmp_path / 'contaminated_same_label_benchmark_contam.md').read_text(encoding='utf-8')
    payload = json.loads((tmp_path / 'contaminated_same_label_benchmark_contam.json').read_text())
    assert md_path.endswith('contaminated_same_label_benchmark_contam.md')
    assert 'CONTAMINATED_SAME_LABEL_EVALUATION' in report
    assert 'All original test nodes and labels were included in training' in report
    assert 'not valid estimates of generalization or deployment performance' in report
    assert 'Validation AP (validation labels not used as training targets)' in report
    assert payload['evaluation_status'] == 'CONTAMINATED_SAME_LABEL_EVALUATION'
    assert payload['rows'] == rows
    assert json_path.endswith('contaminated_same_label_benchmark_contam.json')
