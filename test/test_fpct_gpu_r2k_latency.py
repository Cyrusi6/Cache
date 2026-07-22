import json
from argparse import Namespace
from pathlib import Path

from script.experiment import fpct_gpu_r2k_latency as latency


def _block(index: int, factorized_ratio: float):
    order = latency.BLOCK_ORDER[index]
    canaries = {}
    for label in ("checkpoint_native", "forced_on"):
        arms = []
        for operator in order:
            ratio = factorized_ratio if operator == "f" else 1.0
            arms.append(
                {
                    "operator": operator,
                    "timing": {
                        "cuda_median_seconds": ratio,
                        "wall_median_seconds": ratio,
                    },
                }
            )
        canaries[label] = {"order": list(order), "arms": arms}
    return {
        "block_index": index,
        "process_seed": latency.PROCESS_SEEDS[index],
        "order": list(order),
        "canaries": canaries,
        "accuracy_or_correctness_accessed": False,
    }


def test_block_order_is_balanced_abba():
    assert latency.BLOCK_ORDER.count(("c_post", "f")) == 4
    assert latency.BLOCK_ORDER.count(("f", "c_post")) == 4
    assert latency.BLOCK_ORDER[:4] == (
        ("c_post", "f"),
        ("f", "c_post"),
        ("f", "c_post"),
        ("c_post", "f"),
    )


def test_aggregate_is_diagnostic_only_and_uses_frozen_thresholds(tmp_path: Path):
    for index in range(8):
        (tmp_path / f"block_{index:02d}.json").write_text(
            json.dumps(_block(index, 1.2))
        )
    output = tmp_path / "aggregate.json"
    result = latency.run_aggregate(Namespace(root=str(tmp_path), output=str(output)))
    assert result["status"] == "DIAGNOSTIC_QUALIFIED"
    assert result["may_produce_r2k_go"] is False
    assert result["canaries"]["checkpoint_native"]["cuda_balanced_median_ratio"] == 1.2
    assert result["canaries"]["forced_on"]["cuda_block_bootstrap_one_sided_95_ucb"] == 1.2
    assert result["accuracy_or_correctness_accessed"] is False


def test_slow_diagnostic_never_becomes_go(tmp_path: Path):
    for index in range(8):
        (tmp_path / f"block_{index:02d}.json").write_text(
            json.dumps(_block(index, 1.6))
        )
    result = latency.run_aggregate(
        Namespace(root=str(tmp_path), output=str(tmp_path / "aggregate.json"))
    )
    assert result["status"] == "DIAGNOSTIC_NOT_QUALIFIED"
    assert result["classification"] == "DIAGNOSTIC_ONLY"
