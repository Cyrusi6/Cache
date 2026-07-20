from __future__ import annotations

"""Frozen FPCT confirmatory reductions and paired inference.

Input rows are one row per seed/task/content-group/cell.  Correctness must
already be group-level and lie in [0, 1].  All cells for a group are reduced
jointly; missing or duplicate cells are an integrity failure.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


SEEDS = tuple(range(45, 57))
TASKS = ("arc_challenge", "openbookqa", "mmlu_redux")
CELLS = ("Y_P", "Y_CC", "Y_CF", "Y_FC", "Y_FF")
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20_260_719


class IntegrityError(RuntimeError):
    pass


def read_group_cells(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"seed", "task", "content_group_hash", "cell", "correct"}
    if not rows or not required.issubset(rows[0]):
        raise IntegrityError(f"missing required columns: {sorted(required)}")
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str]] = set()
    for row in rows:
        item = {
            "seed": int(row["seed"]),
            "task": row["task"],
            "content_group_hash": row["content_group_hash"],
            "cell": row["cell"],
            "correct": float(row["correct"]),
        }
        key = (item["seed"], item["task"], item["content_group_hash"], item["cell"])
        if key in seen:
            raise IntegrityError(f"duplicate cell row: {key}")
        seen.add(key)
        if item["seed"] not in SEEDS or item["task"] not in TASKS or item["cell"] not in CELLS:
            raise IntegrityError(f"out-of-contract row: {key}")
        if not math.isfinite(item["correct"]) or not 0 <= item["correct"] <= 1:
            raise IntegrityError(f"invalid correctness: {key}")
        parsed.append(item)
    return parsed


def group_cube(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["task"]))
        group = grouped.setdefault(key, {}).setdefault(str(row["content_group_hash"]), {})
        cell = str(row["cell"])
        if cell in group:
            raise IntegrityError(f"duplicate cell in group: {key}/{cell}")
        group[cell] = float(row["correct"])
    cube: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        for task in TASKS:
            groups = grouped.get((seed, task), {})
            if not groups:
                raise IntegrityError(f"missing seed/task: {seed}/{task}")
            ordered = sorted(groups)
            for group_id in ordered:
                if set(groups[group_id]) != set(CELLS):
                    raise IntegrityError(f"incomplete cells: {seed}/{task}/{group_id}")
            cube[(seed, task)] = {
                cell: np.asarray([groups[group_id][cell] for group_id in ordered], dtype=np.float64)
                for cell in CELLS
            }
            cube[(seed, task)]["group_ids"] = np.asarray(ordered, dtype=object)
    return cube


def seed_task_cells(cube: dict[tuple[int, str], dict[str, np.ndarray]]) -> dict[int, dict[str, dict[str, float]]]:
    result: dict[int, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        result[seed] = {}
        for task in TASKS:
            result[seed][task] = {
                cell: float(cube[(seed, task)][cell].mean()) for cell in CELLS
            }
    return result


def _deltas(cells: dict[str, float]) -> dict[str, float]:
    dc = cells["Y_CF"] - cells["Y_CC"]
    df = cells["Y_FF"] - cells["Y_FC"]
    return {
        "T": cells["Y_FF"] - cells["Y_CC"],
        "D_C": dc,
        "D_F": df,
        "O": (dc + df) / 2.0,
        "I": df - dc,
        "N": cells["Y_CC"] - cells["Y_P"],
    }


def seed_level_estimands(task_cells: dict[int, dict[str, dict[str, float]]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        macro_cells = {
            cell: float(np.mean([task_cells[seed][task][cell] for task in TASKS]))
            for cell in CELLS
        }
        output[seed] = {
            "cells": macro_cells,
            "estimands": _deltas(macro_cells),
            "tasks": {
                task: {
                    "cells": task_cells[seed][task],
                    "estimands": _deltas(task_cells[seed][task]),
                }
                for task in TASKS
            },
        }
    return output


def exact_sign_flip(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise IntegrityError("exact sign flip requires 12 finite matched deltas")
    observed = float(values.mean())
    means = np.empty(2 ** len(values), dtype=np.float64)
    for index, signs in enumerate(itertools.product((-1.0, 1.0), repeat=len(values))):
        means[index] = float(np.mean(values * np.asarray(signs)))
    eps = 1e-15
    one = float(np.mean(means >= observed - eps))
    two = float(np.mean(np.abs(means) >= abs(observed) - eps))
    return {"observed_mean": observed, "one_sided_p": one, "two_sided_p": min(1.0, two)}


def exact_sign_test(values: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(values), dtype=np.float64)
    nonzero = values[values != 0]
    positives = int((nonzero > 0).sum())
    n = int(nonzero.size)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "one_sided_p": 1.0, "two_sided_p": 1.0}
    one = float(stats.binomtest(positives, n, 0.5, alternative="greater").pvalue)
    two = float(stats.binomtest(positives, n, 0.5, alternative="two-sided").pvalue)
    return {"n_nonzero": n, "positive": positives, "one_sided_p": one, "two_sided_p": two}


def paired_t_sensitivity(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise IntegrityError("paired t sensitivity requires 12 finite deltas")
    result = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
    return {"t": float(result.statistic), "one_sided_p": float(result.pvalue)}


def hierarchical_bootstrap(
    cube: dict[tuple[int, str], dict[str, np.ndarray]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(rng_seed)
    estimates = {name: np.empty(replicates, dtype=np.float64) for name in ("T", "D_C", "D_F", "O", "I", "N")}
    for replicate in range(replicates):
        sampled_seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
        replicate_estimands: list[dict[str, float]] = []
        for seed_index in sampled_seed_indices:
            seed = SEEDS[int(seed_index)]
            macro_cells = {cell: 0.0 for cell in CELLS}
            for task in TASKS:
                task_cube = cube[(seed, task)]
                n = len(task_cube["group_ids"])
                group_draw = rng.integers(0, n, size=n)
                for cell in CELLS:
                    macro_cells[cell] += float(task_cube[cell][group_draw].mean()) / len(TASKS)
            replicate_estimands.append(_deltas(macro_cells))
        for name in estimates:
            estimates[name][replicate] = float(np.mean([item[name] for item in replicate_estimands]))
    return {
        name: {
            "mean": float(values.mean()),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "lower_90": float(np.quantile(values, 0.05)),
            "upper_90": float(np.quantile(values, 0.95)),
        }
        for name, values in estimates.items()
    }


def classify_held_out(seed_values: dict[int, dict[str, Any]], bootstrap: dict[str, dict[str, float]]) -> str:
    t = np.asarray([seed_values[seed]["estimands"]["T"] for seed in SEEDS])
    o = np.asarray([seed_values[seed]["estimands"]["O"] for seed in SEEDS])
    df = np.asarray([seed_values[seed]["estimands"]["D_F"] for seed in SEEDS])
    sign_t = exact_sign_flip(t)
    performance = (
        t.mean() >= 0.01
        and sign_t["one_sided_p"] <= 0.05
        and bootstrap["T"]["lower_95"] > 0
        and int((t > 0).sum()) >= 9
    )
    task_harm = False
    severe_task_harm = False
    for task in TASKS:
        task_t = np.asarray([seed_values[seed]["tasks"][task]["estimands"]["T"] for seed in SEEDS])
        task_harm |= bool(task_t.mean() < -0.01 and int((task_t < 0).sum()) >= 8)
        severe_task_harm |= bool(task_t.mean() <= -0.02 and int((task_t < 0).sum()) >= 9)
    performance &= not task_harm
    if performance:
        sign_o = exact_sign_flip(o)
        mechanism = (
            sign_o["one_sided_p"] <= 0.05
            and bootstrap["O"]["lower_95"] > 0
            and df.mean() > 0
            and int((o > 0).sum()) >= 9
        )
        return "MECHANISM_SUPPORTED_GO" if mechanism else "PERFORMANCE_GO_MECHANISM_UNRESOLVED"
    if bootstrap["T"]["upper_95"] < 0 or severe_task_harm:
        return "HARM_NO_GO"
    if bootstrap["T"]["upper_95"] < 0.005:
        return "FUTILITY_NO_GO"
    return "INCONCLUSIVE"


def analyse(path: Path, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    cube = group_cube(read_group_cells(path))
    seeds = seed_level_estimands(seed_task_cells(cube))
    bootstrap = hierarchical_bootstrap(cube, replicates=replicates)
    inference = {}
    for name in ("T", "O", "I", "N"):
        values = [seeds[seed]["estimands"][name] for seed in SEEDS]
        inference[name] = {
            "exact_sign_flip": exact_sign_flip(values),
            "paired_t_sensitivity": paired_t_sensitivity(values),
            "exact_sign_test_sensitivity": exact_sign_test(values),
        }
    return {
        "schema_version": 1,
        "input": str(path.resolve()),
        "seeds": {str(seed): value for seed, value in seeds.items()},
        "bootstrap": bootstrap,
        "inference": inference,
        "classification_without_external_control_gates": classify_held_out(seeds, bootstrap),
        "assumption_boundary": {
            "sign_flip": "sharp/symmetric null with sign-exchangeability",
            "paired_t": "independent seed deltas and approximate normality of the sample mean",
            "sign_test": "independent exchangeable signs under a continuous no-tie null"
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    payload = analyse(args.input, replicates=args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
