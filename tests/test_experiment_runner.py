from run_experiments import expected_run_name, fraction_tag


def test_fraction_tags_preserve_fraction_precision():
    assert fraction_tag(0.25) == "leak25pct"
    assert fraction_tag(0.125) == "leak12p5pct"
    assert fraction_tag(1.0) == "leak100pct"
    assert fraction_tag(0.125) != fraction_tag(0.13)


def test_run_names_include_fraction_seed_and_sweep_id():
    name = expected_run_name("configs/sage_baseline.yaml", 42, 0.25, "20261031_190500")
    assert name == "sage_baseline_leak25pct_seed42_20261031_190500"
