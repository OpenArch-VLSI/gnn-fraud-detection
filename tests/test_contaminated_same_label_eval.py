import json
import os
from types import SimpleNamespace

import pytest
import torch

from src.evaluate import (FANOUT as EVAL_FANOUT, SAMPLE_SEED,
                          is_contaminated_same_label_config,
                          load_leakage_indices,
                          load_and_validate_smoke_metadata, validate_smoke_checkpoint,
                          validate_contamination_metadata, validate_evaluation_outputs,
                          finalize_smoke_evaluation, mark_smoke_evaluation_failure,
                          write_smoke_report)
from src.train import (build_training_mask, ensure_resumed_best_model,
                       build_parser, should_save_checkpoint,
                       validate_contaminated_config, selection_metric_name,
                       checkpoint_is_due,
                       validate_three_way_smoke_options,
                       FANOUT as TRAIN_FANOUT)
from src.utils.metrics import selection_score
from run_experiments import expected_run_name


def graph_masks():
    return SimpleNamespace(
        train_mask=torch.tensor([True, True, False, False, False, False]),
        val_mask=torch.tensor([False, False, True, True, False, False]),
        test_mask=torch.tensor([False, False, False, False, True, True]),
    )


def contaminated_config():
    return {
        'contaminated_same_label_eval': True,
        'leakage_test_fraction': 1.0,
        'seed': 42,
    }


def test_full_mix_rejects_any_fraction_other_than_one():
    with pytest.raises(ValueError, match='leakage_test_fraction 1.0'):
        validate_contaminated_config({
            'leakage_test_fraction': 0.25,
            'graph': 'causal',
            'run_name': 'CONTAMINATED_example',
        })


def test_clean_defaults_and_sampling_configuration_remain_safe():
    args = build_parser().parse_args([])
    assert args.contaminated_same_label_eval is False
    assert args.leakage_test_fraction == 0.0
    assert TRAIN_FANOUT == EVAL_FANOUT == [25, 10]
    assert SAMPLE_SEED == 0


def test_full_mix_mask_is_train_or_test_and_excludes_validation():
    data = graph_masks()
    mask, exposed = build_training_mask(data, contaminated_config())
    assert torch.equal(mask, data.train_mask | data.test_mask)
    assert not torch.any(mask & data.val_mask)
    assert torch.equal(exposed, torch.where(data.test_mask)[0])


def test_three_way_smoke_mask_includes_all_original_splits_without_mutation():
    data = graph_masks()
    original_train = data.train_mask.clone()
    original_val = data.val_mask.clone()
    original_test = data.test_mask.clone()
    mask, exposed = build_training_mask(data, {
        'contaminated_three_way_same_label_eval': True,
        'leakage_test_fraction': 1.0,
        'seed': 42,
    })
    assert torch.equal(mask, original_train | original_val | original_test)
    assert torch.equal(data.train_mask, original_train)
    assert torch.equal(data.val_mask, original_val)
    assert torch.equal(data.test_mask, original_test)
    assert int(mask.sum()) == 6
    assert torch.equal(exposed, torch.where(original_test)[0])


def test_three_way_smoke_rejects_resume_directory():
    with pytest.raises(ValueError, match='does not support --resume_dir'):
        validate_three_way_smoke_options({'resume_dir': 'experiments/previous'})


def test_clean_mask_remains_unchanged_at_zero_fraction():
    data = graph_masks()
    mask, exposed = build_training_mask(data, {'leakage_test_fraction': 0.0, 'seed': 42})
    assert torch.equal(mask, data.train_mask)
    assert len(exposed) == 0


def test_contamination_metadata_requires_all_test_ids():
    config = {
        'contaminated_same_label_eval': True,
        'evaluation_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
        'leakage_test_fraction': 1.0,
        'training_splits': ['train', 'test'],
        'evaluation_split': 'test',
        'valid_generalization_estimate': False,
        'graph': 'causal',
    }
    all_test_ids = torch.tensor([4, 5])
    validate_contamination_metadata(config, all_test_ids, all_test_ids)
    with pytest.raises(ValueError):
        validate_contamination_metadata(config, torch.tensor([4]), all_test_ids)
    broken = dict(config, evaluation_split='val')
    with pytest.raises(ValueError):
        validate_contamination_metadata(broken, all_test_ids, all_test_ids)


def test_evaluator_rejects_ambiguous_or_contradictory_full_mix_metadata():
    with pytest.raises(ValueError, match='requires explicit'):
        is_contaminated_same_label_config({'leakage_test_fraction': 1.0})
    with pytest.raises(ValueError, match='incomplete'):
        is_contaminated_same_label_config({
            'leakage_test_fraction': 1.0,
            'contaminated_same_label_eval': True,
        })
    assert not is_contaminated_same_label_config({'leakage_test_fraction': 0.25})


def test_contaminated_evaluation_requires_leakage_index_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match='CONTAMINATED_SAME_LABEL_EVALUATION'):
        load_leakage_indices(str(tmp_path), 1.0, contaminated=True)
    # An ordinary clean run retains its empty leakage-ID behavior.
    assert len(load_leakage_indices(str(tmp_path), 0.0, contaminated=False)) == 0


def test_contaminated_configuration_requires_causal_graph_and_marked_run_name():
    with pytest.raises(ValueError, match='graph causal'):
        validate_contaminated_config({
            'leakage_test_fraction': 1.0,
            'graph': 'full',
            'run_name': 'CONTAMINATED_example',
        })
    with pytest.raises(ValueError, match='run_name'):
        validate_contaminated_config({
            'leakage_test_fraction': 1.0,
            'graph': 'causal',
            'run_name': 'ordinary_run',
        })


def test_metadata_rejects_missing_flag_bad_splits_and_duplicate_ids():
    base = {
        'contaminated_same_label_eval': True,
        'evaluation_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
        'leakage_test_fraction': 1.0,
        'training_splits': ['train', 'test'],
        'evaluation_split': 'test',
        'valid_generalization_estimate': False,
        'graph': 'causal',
    }
    test_ids = torch.tensor([4, 5])
    with pytest.raises(ValueError):
        validate_contamination_metadata(dict(base, training_splits=['train']), test_ids, test_ids)
    with pytest.raises(ValueError):
        validate_contamination_metadata(dict(base, valid_generalization_estimate=True), test_ids, test_ids)
    with pytest.raises(ValueError):
        validate_contamination_metadata(base, torch.tensor([4, 4]), test_ids)


def test_three_way_metadata_requires_validation_training_declaration():
    config = {
        'contaminated_same_label_eval': True,
        'contaminated_three_way_same_label_eval': True,
        'evaluation_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
        'leakage_test_fraction': 1.0,
        'training_splits': ['train', 'val', 'test'],
        'evaluation_split': 'test',
        'valid_generalization_estimate': False,
        'validation_labels_used_as_training_targets': True,
        'graph': 'causal',
    }
    test_ids = torch.tensor([4, 5])
    validate_contamination_metadata(config, test_ids, test_ids)
    with pytest.raises(ValueError):
        validate_contamination_metadata(
            dict(config, validation_labels_used_as_training_targets=False), test_ids, test_ids)


def test_smoke_report_contains_artifact_backed_training_and_evaluation_details(tmp_path):
    path = tmp_path / 'smoke_report.md'
    metadata = {
        'experiment_type': 'CONTAMINATED_THREE_WAY_SAME_LABEL_SMOKE_TEST',
        'training_target_splits': ['train', 'val', 'test'],
        'split_counts': {'train': 2, 'val': 2, 'test': 2, 'combined_training_targets': 6},
        'test_evaluation_targets': 2, 'configured_epochs': 3, 'completed_epochs': 3,
        'checkpoint_path': 'experiments/x/best_model.pt', 'checkpoint_epoch': 3,
        'training_duration_seconds': 12.0,
    }
    log = [{'epoch': 1, 'train_loss': 1.0, 'epoch_duration_seconds': 2.0},
           {'epoch': 2, 'train_loss': 0.8, 'epoch_duration_seconds': 2.0},
           {'epoch': 3, 'train_loss': 0.6, 'epoch_duration_seconds': 2.0}]
    results = {'pr_auc': 0.5, 'roc_auc': 0.7, 'total_evaluation_duration_seconds': 3.0}
    write_smoke_report(str(path), 'CONTAMINATED_expA', {'model': 'sage', 'seed': 42},
                       'causal', metadata, log, results, 0.25)
    report = path.read_text(encoding='utf-8')
    assert 'Original train, validation, and test labels were training targets' in report
    assert 'Configured epochs: 3' in report
    assert 'Epoch 3: training loss=0.600000' in report
    assert 'Final-epoch checkpoint' in report
    assert 'Average Precision' in report and 'ROC-AUC' in report


def smoke_config():
    return {
        # These are read by write_smoke_report and must describe the fixed
        # causal three-way smoke evaluation, not a generic test configuration.
        'model': 'sage',
        'seed': 42,
        'graph': 'causal',
        'contaminated_three_way_same_label_eval': True,
        'contaminated_same_label_eval': True,
        'evaluation_status': 'CONTAMINATED_SAME_LABEL_EVALUATION',
        'training_splits': ['train', 'val', 'test'],
        'evaluation_split': 'test',
        'valid_generalization_estimate': False,
        'validation_labels_used_as_training_targets': True,
        'epochs': 3, 'leakage_test_fraction': 1.0,
    }


def smoke_metadata():
    return {
        'experiment_type': 'CONTAMINATED_THREE_WAY_SAME_LABEL_SMOKE_TEST',
        'training_status': 'completed', 'evaluation_stage_status': 'pending',
        'overall_status': 'evaluation_pending', 'configured_epochs': 3,
        'completed_epochs': 3, 'checkpoint_policy': 'final_epoch',
        'checkpoint_epoch': 3, 'training_target_splits': ['train', 'val', 'test'],
        'model': 'sage', 'graph': 'causal',
    }


def test_smoke_evaluation_rejects_missing_or_inconsistent_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match='smoke_metadata'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config())
    (tmp_path / 'smoke_metadata.json').write_text(json.dumps(
        dict(smoke_metadata(), completed_epochs=2)), encoding='utf-8')
    with pytest.raises(ValueError, match='completed_epochs'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config())


def test_smoke_evaluation_retry_and_completed_result_protection(tmp_path):
    path = tmp_path / 'smoke_metadata.json'
    path.write_text(json.dumps(dict(smoke_metadata(), evaluation_stage_status='failed',
                                    overall_status='failed')), encoding='utf-8')
    with pytest.raises(RuntimeError, match='pass --force'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config())
    assert load_and_validate_smoke_metadata(str(tmp_path), smoke_config(), allow_retry=True)['overall_status'] == 'failed'
    path.write_text(json.dumps(dict(smoke_metadata(), evaluation_stage_status='completed',
                                    overall_status='completed')), encoding='utf-8')
    with pytest.raises(FileExistsError, match='already completed'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config(), allow_retry=True)


def test_partial_evaluation_outputs_require_force_for_a_failed_retry(tmp_path):
    out_path = tmp_path / 'test_results.json'
    scores_path = tmp_path / 'test_scores.npz'
    out_path.write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='partial/failed'):
        validate_evaluation_outputs(str(out_path), str(scores_path), force=False)
    validate_evaluation_outputs(str(out_path), str(scores_path), force=True)


def test_smoke_finalization_does_not_complete_or_publish_on_report_failure(tmp_path):
    metadata = dict(smoke_metadata(), split_counts={'train': 1, 'val': 1, 'test': 2,
                    'combined_training_targets': 4}, training_duration_seconds=1.0,
                    checkpoint_path='best_model.pt', test_evaluation_targets=2)
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

    def fail_report(*_args, **_kwargs):
        raise RuntimeError('report write failed')

    with pytest.raises(RuntimeError, match='report write failed'):
        finalize_smoke_evaluation(
            str(tmp_path), smoke_config(), 'causal', metadata,
            [{'epoch': 3, 'train_loss': 0.1, 'epoch_duration_seconds': 1.0}],
            {'pr_auc': 0.5, 'roc_auc': 0.6, 'total_evaluation_duration_seconds': 1.0},
            torch.tensor([0.1, 0.2]), torch.tensor([0, 1]), torch.tensor([4, 5]),
            torch.tensor([True, True]), report_writer=fail_report)
    assert not (tmp_path / 'test_results.json').exists()
    assert not (tmp_path / 'test_scores.npz').exists()
    assert not (tmp_path / 'smoke_report.md').exists()
    mark_smoke_evaluation_failure(str(metadata_path), RuntimeError('report write failed'))
    failed = json.loads(metadata_path.read_text())
    assert failed['evaluation_stage_status'] == 'failed'
    assert failed['overall_status'] == 'failed'


def test_completed_smoke_metadata_is_never_downgraded_by_later_failure(tmp_path):
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata_path.write_text(json.dumps(dict(smoke_metadata(), evaluation_stage_status='completed',
                                             overall_status='completed')), encoding='utf-8')
    mark_smoke_evaluation_failure(str(metadata_path), RuntimeError('late failure'))
    completed = json.loads(metadata_path.read_text())
    assert completed['evaluation_stage_status'] == 'completed'
    assert completed['overall_status'] == 'completed'


def smoke_finalization_inputs(tmp_path):
    metadata = dict(smoke_metadata(), split_counts={'train': 1, 'val': 1, 'test': 2,
                    'combined_training_targets': 4}, training_duration_seconds=1.0,
                    checkpoint_path='best_model.pt', test_evaluation_targets=2)
    return (
        str(tmp_path), smoke_config(), 'causal', metadata,
        [{'epoch': 3, 'train_loss': 0.1, 'epoch_duration_seconds': 1.0}],
        {'pr_auc': 0.5, 'roc_auc': 0.6, 'total_evaluation_duration_seconds': 1.0},
        torch.tensor([0.1, 0.2]), torch.tensor([0, 1]), torch.tensor([4, 5]),
        torch.tensor([True, True]),
    )


@pytest.mark.parametrize('failure_call', [2, 3])
def test_publication_failure_restores_existing_partial_artifacts(tmp_path, failure_call):
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata = smoke_finalization_inputs(tmp_path)[3]
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    old_contents = {
        'test_results.json': 'old results',
        'test_scores.npz': b'old scores',
        'smoke_report.md': 'old report',
    }
    (tmp_path / 'test_results.json').write_text(old_contents['test_results.json'], encoding='utf-8')
    (tmp_path / 'test_scores.npz').write_bytes(old_contents['test_scores.npz'])
    (tmp_path / 'smoke_report.md').write_text(old_contents['smoke_report.md'], encoding='utf-8')
    original_metadata = metadata_path.read_text(encoding='utf-8')
    calls = {'count': 0}

    def fail_on_requested_replace(source, destination):
        calls['count'] += 1
        if calls['count'] == failure_call:
            raise OSError('publication failed')
        os.replace(source, destination)

    with pytest.raises(OSError, match='publication failed'):
        finalize_smoke_evaluation(*smoke_finalization_inputs(tmp_path),
                                  replace_func=fail_on_requested_replace)
    assert (tmp_path / 'test_results.json').read_text(encoding='utf-8') == old_contents['test_results.json']
    assert (tmp_path / 'test_scores.npz').read_bytes() == old_contents['test_scores.npz']
    assert (tmp_path / 'smoke_report.md').read_text(encoding='utf-8') == old_contents['smoke_report.md']
    assert metadata_path.read_text(encoding='utf-8') == original_metadata


def test_metadata_write_failure_restores_published_artifacts(tmp_path):
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata = smoke_finalization_inputs(tmp_path)[3]
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / 'test_results.json').write_text('old results', encoding='utf-8')
    (tmp_path / 'test_scores.npz').write_bytes(b'old scores')
    (tmp_path / 'smoke_report.md').write_text('old report', encoding='utf-8')
    original_metadata = metadata_path.read_text(encoding='utf-8')

    def fail_metadata_write(_path, _payload):
        raise OSError('metadata write failed')

    with pytest.raises(OSError, match='metadata write failed'):
        finalize_smoke_evaluation(*smoke_finalization_inputs(tmp_path),
                                  metadata_writer=fail_metadata_write)
    assert (tmp_path / 'test_results.json').read_text(encoding='utf-8') == 'old results'
    assert (tmp_path / 'test_scores.npz').read_bytes() == b'old scores'
    assert (tmp_path / 'smoke_report.md').read_text(encoding='utf-8') == 'old report'
    assert metadata_path.read_text(encoding='utf-8') == original_metadata


def test_metadata_writer_that_modifies_then_fails_is_rolled_back(tmp_path):
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata = smoke_finalization_inputs(tmp_path)[3]
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / 'test_results.json').write_text('old results', encoding='utf-8')
    (tmp_path / 'test_scores.npz').write_bytes(b'old scores')
    (tmp_path / 'smoke_report.md').write_text('old report', encoding='utf-8')
    original_metadata = metadata_path.read_text(encoding='utf-8')

    def write_then_fail(path, payload):
        with open(path, 'w') as f:
            json.dump(payload, f)
        raise OSError('metadata changed then failed')

    with pytest.raises(OSError, match='metadata changed then failed'):
        finalize_smoke_evaluation(*smoke_finalization_inputs(tmp_path),
                                  metadata_writer=write_then_fail)
    assert (tmp_path / 'test_results.json').read_text(encoding='utf-8') == 'old results'
    assert (tmp_path / 'test_scores.npz').read_bytes() == b'old scores'
    assert (tmp_path / 'smoke_report.md').read_text(encoding='utf-8') == 'old report'
    assert metadata_path.read_text(encoding='utf-8') == original_metadata


def test_rollback_failure_retains_backups_and_reports_both_errors(tmp_path):
    metadata_path = tmp_path / 'smoke_metadata.json'
    metadata = smoke_finalization_inputs(tmp_path)[3]
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    (tmp_path / 'test_results.json').write_text('old results', encoding='utf-8')
    (tmp_path / 'test_scores.npz').write_bytes(b'old scores')
    (tmp_path / 'smoke_report.md').write_text('old report', encoding='utf-8')
    calls = {'count': 0}

    def fail_second_publication(source, destination):
        calls['count'] += 1
        if calls['count'] == 2:
            raise OSError('second publish failed')
        os.replace(source, destination)

    def fail_result_restore(source, destination):
        if destination.endswith('test_results.json'):
            raise OSError('result rollback failed')
        os.replace(source, destination)

    with pytest.raises(RuntimeError, match='Rollback also failed') as error:
        finalize_smoke_evaluation(*smoke_finalization_inputs(tmp_path),
                                  replace_func=fail_second_publication,
                                  restore_func=fail_result_restore)
    assert 'second publish failed' in str(error.value)
    assert 'result rollback failed' in str(error.value)
    assert list(tmp_path.glob('*.backup'))


def test_smoke_evaluation_rejects_mismatched_model_or_graph(tmp_path):
    path = tmp_path / 'smoke_metadata.json'
    # Keep the config valid so this reaches the recorded-model comparison.
    path.write_text(json.dumps(dict(smoke_metadata(), model='gat')), encoding='utf-8')
    with pytest.raises(ValueError, match='recorded model'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config())
    # Likewise, this is a recorded-graph mismatch, distinct from an invalid
    # graph in config.json below.
    path.write_text(json.dumps(dict(smoke_metadata(), graph='full')), encoding='utf-8')
    with pytest.raises(ValueError, match='recorded graph'):
        load_and_validate_smoke_metadata(str(tmp_path), smoke_config())
    with pytest.raises(ValueError, match='config: graph'):
        load_and_validate_smoke_metadata(str(tmp_path), dict(smoke_config(), graph='full'))


def test_smoke_checkpoint_must_match_final_epoch_metadata():
    metadata = smoke_metadata()
    validate_smoke_checkpoint({'checkpoint_policy': 'final_epoch', 'epoch': 3}, smoke_config(), metadata)
    with pytest.raises(ValueError, match='checkpoint_policy'):
        validate_smoke_checkpoint({'checkpoint_policy': 'metric_selected', 'epoch': 3}, smoke_config(), metadata)
    with pytest.raises(ValueError, match='epoch'):
        validate_smoke_checkpoint({'checkpoint_policy': 'final_epoch', 'epoch': 2}, smoke_config(), metadata)


def test_checkpoint_selection_saves_first_zero_score_then_only_improvements():
    assert selection_score({'pr_auc': 0.37, 'roc_auc': 0.99}) == 0.37
    assert selection_metric_name({}) == 'val_pr_auc'
    assert selection_metric_name({'contaminated_same_label_eval': True}) == 'contaminated_test_pr_auc'
    assert selection_metric_name({'contaminated_three_way_same_label_eval': True}) == 'final_epoch'
    assert should_save_checkpoint(None, 0.0)
    assert not should_save_checkpoint(0.0, 0.0)
    assert should_save_checkpoint(0.0, 0.1)


def test_checkpoint_selection_uses_the_mode_specific_recorded_metric():
    rows = [
        {'epoch': 1, 'pr_auc': 0.70, 'contaminated_test_pr_auc': 0.10},
        {'epoch': 2, 'pr_auc': 0.20, 'contaminated_test_pr_auc': 0.90},
        {'epoch': 3, 'pr_auc': 0.60, 'contaminated_test_pr_auc': 0.40},
    ]

    def selected_epoch(config):
        best_score, epoch = None, None
        for row in rows:
            metric = (row['contaminated_test_pr_auc']
                      if selection_metric_name(config) == 'contaminated_test_pr_auc'
                      else row['pr_auc'])
            if should_save_checkpoint(best_score, metric):
                best_score, epoch = metric, row['epoch']
        return epoch, best_score

    assert selected_epoch({}) == (1, 0.70)
    assert selected_epoch({'contaminated_same_label_eval': True}) == (2, 0.90)


def test_three_way_smoke_checkpoint_is_final_epoch_and_ignores_test_ap():
    config = {'contaminated_three_way_same_label_eval': True, 'epochs': 3}
    # Deliberately make an earlier hypothetical test AP higher than epoch three.
    test_ap = [0.99, 0.80, 0.10]
    saved_epochs = [epoch for epoch, score in enumerate(test_ap, start=1)
                    if checkpoint_is_due(config, epoch, None, score)]
    assert saved_epochs == [3]


def test_contaminated_runner_name_is_unambiguous():
    name = expected_run_name('configs/sage_baseline.yaml', 42, 1.0, 'stamp', contaminated=True)
    assert name == 'CONTAMINATED_sage_baseline_mix_train_test_seed42_stamp'


def test_resume_immediately_writes_evaluable_best_checkpoint(tmp_path):
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    checkpoint = {'model': {'weight': torch.tensor([1.0])}, 'epoch': 2}
    recovery = ensure_resumed_best_model(str(old_dir), str(new_dir), checkpoint)
    assert (new_dir / 'best_model.pt').exists()
    assert torch.load(new_dir / 'best_model.pt', weights_only=False)['epoch'] == 2
    assert recovery['recovered'] is False
    assert 'checkpoint_recovery' in torch.load(new_dir / 'best_model.pt', weights_only=False)


def test_resume_preserves_existing_best_checkpoint(tmp_path):
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    torch.save({'epoch': 1}, old_dir / 'best_model.pt')
    recovery = ensure_resumed_best_model(str(old_dir), str(new_dir), {'epoch': 2})
    assert recovery == {'recovered': True, 'method': 'copied_prior_best_model'}
    assert torch.load(new_dir / 'best_model.pt', weights_only=False)['epoch'] == 1
