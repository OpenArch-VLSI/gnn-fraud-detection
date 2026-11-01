from types import SimpleNamespace

import pytest
import torch

from src.evaluate import (FANOUT as EVAL_FANOUT, SAMPLE_SEED,
                          is_contaminated_same_label_config,
                          load_leakage_indices,
                          validate_contamination_metadata)
from src.train import (build_training_mask, ensure_resumed_best_model,
                       build_parser, should_save_checkpoint,
                       validate_contaminated_config, selection_metric_name,
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


def test_checkpoint_selection_saves_first_zero_score_then_only_improvements():
    assert selection_score({'pr_auc': 0.37, 'roc_auc': 0.99}) == 0.37
    assert selection_metric_name({}) == 'val_pr_auc'
    assert selection_metric_name({'contaminated_same_label_eval': True}) == 'contaminated_test_pr_auc'
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
