import csv
from pathlib import Path

import numpy as np
import pytest

from script.analysis.fpct_confirmatory_statistics import (
    CELLS,
    SEEDS,
    TASKS,
    IntegrityError,
    exact_sign_flip,
    exact_sign_test,
    group_cube,
    hierarchical_bootstrap,
    seed_level_estimands,
    seed_task_cells,
)


def synthetic_rows(effect=0.02):
    rows = []
    for seed in SEEDS:
        for task in TASKS:
            for group in range(5):
                base = 0.4 + group * 0.01
                values = {
                    "Y_P": base - 0.005,
                    "Y_CC": base,
                    "Y_CF": base + effect / 2,
                    "Y_FC": base + effect / 3,
                    "Y_FF": base + effect,
                }
                for cell in CELLS:
                    rows.append({
                        "seed": seed, "task": task,
                        "content_group_hash": f"{task}-{group}",
                        "cell": cell, "correct": values[cell],
                    })
    return rows


def test_estimands_are_frozen_formulas():
    values = seed_level_estimands(seed_task_cells(group_cube(synthetic_rows())))[45]["estimands"]
    assert values["T"] == pytest.approx(0.02)
    assert values["D_C"] == pytest.approx(0.01)
    assert values["D_F"] == pytest.approx(0.02 - 0.02 / 3)
    assert values["O"] == pytest.approx((values["D_C"] + values["D_F"]) / 2)
    assert values["I"] == pytest.approx(values["D_F"] - values["D_C"])
    assert values["N"] == pytest.approx(0.005)


def test_exact_sign_flip_enumerates_all_4096_assignments():
    result = exact_sign_flip(np.ones(12))
    assert result["one_sided_p"] == pytest.approx(1 / 4096)
    assert result["two_sided_p"] == pytest.approx(2 / 4096)


def test_exact_sign_test_is_separate_sensitivity():
    result = exact_sign_test([1] * 9 + [-1] * 3)
    assert result["n_nonzero"] == 12
    assert result["positive"] == 9
    assert 0 < result["one_sided_p"] < 0.1


def test_hierarchical_bootstrap_is_deterministic_and_joint():
    cube = group_cube(synthetic_rows())
    first = hierarchical_bootstrap(cube, replicates=100, rng_seed=7)
    second = hierarchical_bootstrap(cube, replicates=100, rng_seed=7)
    assert first == second
    assert first["T"]["lower_95"] == pytest.approx(0.02)


def test_missing_cell_is_integrity_failure():
    rows = synthetic_rows()
    rows.pop()
    with pytest.raises(IntegrityError, match="incomplete cells"):
        group_cube(rows)
