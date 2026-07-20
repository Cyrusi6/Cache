from __future__ import annotations

"""Machine-verifiable zero-scientific-change check for FPCT-3.5P/3.7-R1."""

import argparse
import json
from pathlib import Path
from typing import Any

from fpct_bootstrap import require_active


REPO_ROOT = Path(__file__).resolve().parents[2]
require_active(target=Path(__file__))


class ProtocolDiffError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolDiffError(f"expected JSON object: {path}")
    return value


def verify() -> dict[str, Any]:
    old35 = _read(
        REPO_ROOT / "recipe/eval_recipe/fpct_3_5/alignment_correctness_manifest.json"
    )
    old37 = _read(
        REPO_ROOT / "recipe/eval_recipe/fpct_3_7/certified_support_manifest.json"
    )
    new35 = _read(
        REPO_ROOT / "recipe/eval_recipe/fpct_3_5p/provenance_replay_manifest.json"
    )
    new37 = _read(
        REPO_ROOT / "recipe/eval_recipe/fpct_3_7r/import_provenance_manifest.json"
    )
    diff = _read(REPO_ROOT / "recipe/eval_recipe/fpct_3_7r/protocol_diff.json")
    failures: list[str] = []
    if diff.get("scientific_change_count") != 0:
        failures.append("protocol diff scientific_change_count")
    if new35["inherited_scientific_contract"]["top_k"] != old35[
        "certified_one_to_many"
    ]["top_k"]:
        failures.append("FPCT-3.5P top_k")
    for key in (
        "minimum_positive_groups_each_task",
        "minimum_pooled_positive_groups",
    ):
        if new35["inherited_scientific_contract"][key] != old35["readiness"][key]:
            failures.append(f"FPCT-3.5P readiness/{key}")
        if new37["readiness"][key] != old37["readiness"][key]:
            failures.append(f"FPCT-3.7-R1 readiness/{key}")
    alignment = dict(new37["alignment"])
    alignment.pop("sanitizer")
    if alignment != old37["alignment"]:
        failures.append("FPCT-3.7-R1 alignment contract")
    if new37["readiness"]["primary"] != old37["readiness"]["primary"]:
        failures.append("FPCT-3.7-R1 primary readiness")
    if new37["readiness"]["same_tokenizer_control_ranked"] != old37[
        "readiness"
    ]["same_tokenizer_control_ranked"]:
        failures.append("FPCT-3.7-R1 same-tokenizer ranking exclusion")
    for section, source in (("fpct_3_5", old35), ("fpct_3_7", old37)):
        for field in diff["preserved_fields"][section]:
            if field not in source:
                failures.append(f"missing frozen source field {section}/{field}")
    if failures:
        raise ProtocolDiffError(
            "scientific protocol drift detected: " + ", ".join(failures)
        )
    return {
        "schema_version": 1,
        "status": "ZERO_SCIENTIFIC_CHANGES_VERIFIED",
        "scientific_change_count": 0,
        "checked": {
            "fpct_3_5_exact_identity": old35["tokenizer_identity"],
            "fpct_3_5_certifier": old35["certified_one_to_many"],
            "fpct_3_5_sanitizer": old35["conditional_sanitizer"],
            "fpct_3_7_alignment": old37["alignment"],
            "fpct_3_7_readiness": old37["readiness"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify()
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            raise ProtocolDiffError("output must be absolute")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
