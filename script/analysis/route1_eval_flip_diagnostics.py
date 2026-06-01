from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


Row = Dict[str, str]
Stats = MutableMapping[str, Any]


def _read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_cot_csv(run_dir: Path, dataset: str) -> Path:
    dataset_dir = run_dir / dataset
    matches = sorted(dataset_dir.glob("*_cot.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_cot.csv under {dataset_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple *_cot.csv files under {dataset_dir}: {matches}")
    return matches[0]


def _dataset_names(*run_dirs: Path) -> List[str]:
    common: set[str] | None = None
    for run_dir in run_dirs:
        names = {path.name for path in run_dir.iterdir() if path.is_dir()}
        common = names if common is None else common & names
    return sorted(common or set())


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _number(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: Iterable[float | None]) -> float:
    clean = [value for value in values if value is not None]
    return float(statistics.fmean(clean)) if clean else 0.0


def _median(values: Iterable[float | None]) -> float:
    clean = [value for value in values if value is not None]
    return float(statistics.median(clean)) if clean else 0.0


def _normalized_pred(row: Row) -> str:
    return (row.get("extracted_normalized") or row.get("pred") or "").strip().upper()


def _choice_labels(row: Row) -> set[str]:
    return {label for label in "ABCDEFGHIJ" if row.get(label, "").strip()}


def _is_valid_pred(row: Row) -> bool:
    pred = _normalized_pred(row)
    return bool(pred) and pred in _choice_labels(row)


def _key_rows(rows: Sequence[Row]) -> Dict[Tuple[str, str], Row]:
    by_question_id: Dict[str, Row] = {}
    duplicate_ids: set[str] = set()
    for row in rows:
        question_id = row.get("question_id", "")
        if question_id in by_question_id:
            duplicate_ids.add(question_id)
        by_question_id[question_id] = row
    if not duplicate_ids:
        return {(row.get("question_id", ""), ""): row for row in rows}
    return {(row.get("question_id", ""), row.get("question", "")): row for row in rows}


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _category(base_correct: bool, cand_correct: bool, same_pred: bool) -> str:
    if base_correct and cand_correct:
        return "both_correct"
    if base_correct and not cand_correct:
        return "regression"
    if not base_correct and cand_correct:
        return "improvement"
    if same_pred:
        return "both_wrong_same_pred"
    return "both_wrong_diff_pred"


def _counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _empty_dataset_stats(dataset: str) -> Stats:
    return {
        "dataset": dataset,
        "total": 0,
        "baseline_correct": 0,
        "candidate_correct": 0,
        "both_correct": 0,
        "regression": 0,
        "improvement": 0,
        "both_wrong_same_pred": 0,
        "both_wrong_diff_pred": 0,
        "pred_changed": 0,
        "baseline_blank_pred": 0,
        "candidate_blank_pred": 0,
        "baseline_invalid_pred": 0,
        "candidate_invalid_pred": 0,
        "missing_in_baseline": 0,
        "missing_in_candidate": 0,
    }


def _summarize_dataset(
    dataset: str,
    baseline_rows: Sequence[Row],
    candidate_rows: Sequence[Row],
    baseline_name: str,
    candidate_name: str,
    max_examples_per_category: int,
    output_truncate: int,
) -> Tuple[Stats, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    baseline_by_key = _key_rows(baseline_rows)
    candidate_by_key = _key_rows(candidate_rows)
    baseline_keys = set(baseline_by_key)
    candidate_keys = set(candidate_by_key)
    common_keys = sorted(baseline_keys & candidate_keys)

    stats = _empty_dataset_stats(dataset)
    stats["missing_in_baseline"] = len(candidate_keys - baseline_keys)
    stats["missing_in_candidate"] = len(baseline_keys - candidate_keys)

    subject_stats: Dict[str, Stats] = {}
    examples: List[Dict[str, Any]] = []
    example_counts: Counter[str] = Counter()
    baseline_extractors: Counter[str] = Counter()
    candidate_extractors: Counter[str] = Counter()
    baseline_pred_dist: Counter[str] = Counter()
    candidate_pred_dist: Counter[str] = Counter()
    true_answer_dist: Counter[str] = Counter()
    baseline_lengths: List[float | None] = []
    candidate_lengths: List[float | None] = []
    baseline_input_lengths: List[float | None] = []
    candidate_input_lengths: List[float | None] = []

    for key in common_keys:
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        subject = base.get("subject", "")
        if subject not in subject_stats:
            subject_stats[subject] = _empty_dataset_stats(dataset)
            subject_stats[subject]["subject"] = subject

        base_correct = _bool(base.get("is_correct", ""))
        cand_correct = _bool(cand.get("is_correct", ""))
        base_pred = _normalized_pred(base)
        cand_pred = _normalized_pred(cand)
        same_pred = base_pred == cand_pred
        category = _category(base_correct, cand_correct, same_pred)

        stats["total"] += 1
        stats["baseline_correct"] += int(base_correct)
        stats["candidate_correct"] += int(cand_correct)
        stats[category] += 1
        stats["pred_changed"] += int(not same_pred)
        stats["baseline_blank_pred"] += int(not base_pred)
        stats["candidate_blank_pred"] += int(not cand_pred)
        stats["baseline_invalid_pred"] += int(not _is_valid_pred(base))
        stats["candidate_invalid_pred"] += int(not _is_valid_pred(cand))

        subject_row = subject_stats[subject]
        subject_row["total"] += 1
        subject_row["baseline_correct"] += int(base_correct)
        subject_row["candidate_correct"] += int(cand_correct)
        subject_row[category] += 1
        subject_row["pred_changed"] += int(not same_pred)
        subject_row["baseline_blank_pred"] += int(not base_pred)
        subject_row["candidate_blank_pred"] += int(not cand_pred)
        subject_row["baseline_invalid_pred"] += int(not _is_valid_pred(base))
        subject_row["candidate_invalid_pred"] += int(not _is_valid_pred(cand))

        baseline_extractors[base.get("extraction_method_used", "")] += 1
        candidate_extractors[cand.get("extraction_method_used", "")] += 1
        baseline_pred_dist[base_pred] += 1
        candidate_pred_dist[cand_pred] += 1
        true_answer_dist[
            (base.get("ground_truth_normalized") or base.get("true_answer") or "")
            .strip()
            .upper()
        ] += 1
        baseline_lengths.append(_number(base.get("cot_gen_length", "")))
        candidate_lengths.append(_number(cand.get("cot_gen_length", "")))
        baseline_input_lengths.append(_number(base.get("cot_input_length", "")))
        candidate_input_lengths.append(_number(cand.get("cot_input_length", "")))

        if category in {"regression", "improvement", "both_wrong_diff_pred"}:
            example_key = f"{dataset}:{category}"
            if example_counts[example_key] < max_examples_per_category:
                examples.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "subject": subject,
                        "question_id": base.get("question_id", ""),
                        "question": _truncate(
                            base.get("question", ""), output_truncate
                        ),
                        "true_answer": base.get("true_answer", ""),
                        "baseline_name": baseline_name,
                        "baseline_pred": base_pred,
                        "baseline_correct": base_correct,
                        "candidate_name": candidate_name,
                        "candidate_pred": cand_pred,
                        "candidate_correct": cand_correct,
                        "A": _truncate(base.get("A", ""), output_truncate),
                        "B": _truncate(base.get("B", ""), output_truncate),
                        "C": _truncate(base.get("C", ""), output_truncate),
                        "D": _truncate(base.get("D", ""), output_truncate),
                        "baseline_output": _truncate(
                            base.get("cot_output", ""), output_truncate
                        ),
                        "candidate_output": _truncate(
                            cand.get("cot_output", ""), output_truncate
                        ),
                    }
                )
                example_counts[example_key] += 1

    def finalize(row: Stats) -> Stats:
        total = row["total"]
        baseline_correct = row["baseline_correct"]
        candidate_correct = row["candidate_correct"]
        row["baseline_accuracy"] = baseline_correct / total if total else 0.0
        row["candidate_accuracy"] = candidate_correct / total if total else 0.0
        row["delta_accuracy"] = row["candidate_accuracy"] - row["baseline_accuracy"]
        row["net_flip"] = row["improvement"] - row["regression"]
        row["pred_changed_rate"] = row["pred_changed"] / total if total else 0.0
        row["regression_rate"] = row["regression"] / total if total else 0.0
        row["improvement_rate"] = row["improvement"] / total if total else 0.0
        return row

    dataset_stats = finalize(stats)
    subject_rows = [finalize(row) for row in subject_stats.values()]
    subject_rows.sort(
        key=lambda row: (row["delta_accuracy"], row["total"], row["subject"])
    )

    output_stats = {
        "dataset": dataset,
        "baseline_extraction_methods": _counter_to_dict(baseline_extractors),
        "candidate_extraction_methods": _counter_to_dict(candidate_extractors),
        "baseline_pred_distribution": _counter_to_dict(baseline_pred_dist),
        "candidate_pred_distribution": _counter_to_dict(candidate_pred_dist),
        "true_answer_distribution": _counter_to_dict(true_answer_dist),
        "baseline_gen_length_mean": _mean(baseline_lengths),
        "candidate_gen_length_mean": _mean(candidate_lengths),
        "baseline_gen_length_median": _median(baseline_lengths),
        "candidate_gen_length_median": _median(candidate_lengths),
        "baseline_input_length_mean": _mean(baseline_input_lengths),
        "candidate_input_length_mean": _mean(candidate_input_lengths),
    }
    return dataset_stats, subject_rows, examples, output_stats


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    baseline_name: str,
    candidate_name: str,
    baseline_dir: Path,
    candidate_dir: Path,
    dataset_rows: Sequence[Stats],
    output_rows: Sequence[Mapping[str, Any]],
    subject_rows: Sequence[Stats],
) -> None:
    benchmark_mean_base = statistics.fmean(
        row["baseline_accuracy"] for row in dataset_rows
    )
    benchmark_mean_candidate = statistics.fmean(
        row["candidate_accuracy"] for row in dataset_rows
    )
    total = sum(row["total"] for row in dataset_rows)
    weighted_base = (
        sum(row["baseline_correct"] for row in dataset_rows) / total if total else 0.0
    )
    weighted_candidate = (
        sum(row["candidate_correct"] for row in dataset_rows) / total if total else 0.0
    )
    regressions = sum(row["regression"] for row in dataset_rows)
    improvements = sum(row["improvement"] for row in dataset_rows)

    dataset_table = _markdown_table(
        [
            "Dataset",
            "Base Acc",
            "Cand Acc",
            "Delta",
            "Regression",
            "Improvement",
            "Net",
            "Pred Changed",
        ],
        [
            [
                row["dataset"],
                _format_percent(row["baseline_accuracy"]),
                _format_percent(row["candidate_accuracy"]),
                f"{row['delta_accuracy'] * 100:+.2f}",
                row["regression"],
                row["improvement"],
                row["net_flip"],
                _format_percent(row["pred_changed_rate"]),
            ]
            for row in dataset_rows
        ],
    )

    format_table = _markdown_table(
        [
            "Dataset",
            "Base Blank",
            "Cand Blank",
            "Base Invalid",
            "Cand Invalid",
            "Base Gen Len",
            "Cand Gen Len",
        ],
        [
            [
                row["dataset"],
                row["baseline_blank_pred"],
                row["candidate_blank_pred"],
                row["baseline_invalid_pred"],
                row["candidate_invalid_pred"],
                f"{out['baseline_gen_length_mean']:.2f}",
                f"{out['candidate_gen_length_mean']:.2f}",
            ]
            for row, out in zip(dataset_rows, output_rows)
        ],
    )

    answer_distribution_table = _markdown_table(
        [
            "Dataset",
            "Dist",
            "A",
            "B",
            "C",
            "D",
        ],
        [
            [
                out["dataset"],
                label,
                *[
                    f"{dist.get(option, 0) / max(sum(dist.values()), 1) * 100:.1f}"
                    for option in "ABCD"
                ],
            ]
            for out in output_rows
            for label, dist in [
                ("true", out["true_answer_distribution"]),
                ("base_pred", out["baseline_pred_distribution"]),
                ("cand_pred", out["candidate_pred_distribution"]),
            ]
        ],
    )

    top_negative_subjects = [
        row
        for row in subject_rows
        if row["dataset"] == "mmlu-redux" and row["total"] >= 20
    ][:12]
    subject_table = _markdown_table(
        ["Subject", "Base Acc", "Cand Acc", "Delta", "Reg", "Imp", "Net"],
        [
            [
                row["subject"],
                _format_percent(row["baseline_accuracy"]),
                _format_percent(row["candidate_accuracy"]),
                f"{row['delta_accuracy'] * 100:+.2f}",
                row["regression"],
                row["improvement"],
                row["net_flip"],
            ]
            for row in top_negative_subjects
        ],
    )

    candidate_option_totals = {
        option: sum(
            out["candidate_pred_distribution"].get(option, 0) for out in output_rows
        )
        for option in "ABCD"
    }
    candidate_option_total = sum(candidate_option_totals.values())
    candidate_dominant_option = max(
        candidate_option_totals,
        key=lambda option: candidate_option_totals[option],
    )
    candidate_dominant_rate = (
        candidate_option_totals[candidate_dominant_option] / candidate_option_total
        if candidate_option_total
        else 0.0
    )
    has_candidate_format_issue = any(
        row["candidate_blank_pred"] or row["candidate_invalid_pred"]
        for row in dataset_rows
    )
    format_interpretation = (
        f"- `{candidate_name}` has blank or invalid predictions in this comparison;"
        " inspect the output format checks before interpreting accuracy deltas."
        if has_candidate_format_issue
        else f"- `{candidate_name}` does not show a broad output-format failure:"
        " blank and invalid predictions remain zero in this comparison."
    )
    if regressions > improvements:
        flip_interpretation = (
            "- The downstream drop is driven by answer-level flip balance:"
            f" regressions outnumber improvements by {regressions - improvements}."
        )
    else:
        flip_interpretation = (
            "- Answer-level flip balance is non-negative in this comparison:"
            f" improvements outnumber regressions by {improvements - regressions}."
        )
    prior_interpretation = (
        f"- `{candidate_name}` shifts its aggregate answer prior toward option"
        f" `{candidate_dominant_option}`"
        f" ({candidate_dominant_rate * 100:.1f}% of parsed predictions across"
        " the compared benchmarks)."
    )

    lines = [
        f"# Route-1 {baseline_name} vs {candidate_name} Flip Diagnostics",
        "",
        "## Compared Runs",
        "",
        f"- Baseline: `{baseline_name}`",
        f"- Candidate: `{candidate_name}`",
        f"- Baseline dir: `{baseline_dir}`",
        f"- Candidate dir: `{candidate_dir}`",
        "",
        "## Aggregate",
        "",
        f"- Benchmark mean baseline accuracy: `{_format_percent(benchmark_mean_base)}`",
        f"- Benchmark mean candidate accuracy: `{_format_percent(benchmark_mean_candidate)}`",
        f"- Benchmark mean delta: `{(benchmark_mean_candidate - benchmark_mean_base) * 100:+.2f}`",
        f"- Question-weighted baseline accuracy: `{_format_percent(weighted_base)}`",
        f"- Question-weighted candidate accuracy: `{_format_percent(weighted_candidate)}`",
        f"- Total regressions: `{regressions}`",
        f"- Total improvements: `{improvements}`",
        f"- Net flip count: `{improvements - regressions}`",
        "",
        "## Dataset Summary",
        "",
        dataset_table,
        "",
        "## Output Format Checks",
        "",
        format_table,
        "",
        "## Answer Distribution",
        "",
        answer_distribution_table,
        "",
        "## Worst MMLU Subject Deltas",
        "",
        subject_table,
        "",
        "## Interpretation",
        "",
        format_interpretation,
        flip_interpretation,
        prior_interpretation,
        "- Treat this diagnostic as answer-level evidence; causal interpretation"
        " still depends on the training objective and ablation setup.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two completed eval runs and report answer flips."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Run directory containing dataset subdirectories with *_cot.csv files.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Run directory containing dataset subdirectories with *_cot.csv files.",
    )
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary.json/csv/markdown files are written.",
    )
    parser.add_argument("--max-examples-per-category", type=int, default=50)
    parser.add_argument("--output-truncate", type=int, default=360)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_names(args.baseline_dir, args.candidate_dir)
    if not datasets:
        raise ValueError(
            f"No common dataset directories: {args.baseline_dir}, {args.candidate_dir}"
        )

    dataset_rows: List[Stats] = []
    subject_rows: List[Stats] = []
    example_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        baseline_csv = _find_cot_csv(args.baseline_dir, dataset)
        candidate_csv = _find_cot_csv(args.candidate_dir, dataset)
        dataset_stats, dataset_subjects, examples, output_stats = _summarize_dataset(
            dataset=dataset,
            baseline_rows=_read_csv(baseline_csv),
            candidate_rows=_read_csv(candidate_csv),
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            max_examples_per_category=args.max_examples_per_category,
            output_truncate=args.output_truncate,
        )
        dataset_rows.append(dataset_stats)
        subject_rows.extend(dataset_subjects)
        example_rows.extend(examples)
        output_rows.append(output_stats)

    dataset_rows.sort(key=lambda row: row["dataset"])
    subject_rows.sort(
        key=lambda row: (row["dataset"], row["delta_accuracy"], row["subject"])
    )

    summary = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "datasets": dataset_rows,
        "output_checks": output_rows,
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)
    _write_csv(args.output_dir / "subject_summary.csv", subject_rows)
    _write_csv(args.output_dir / "flip_examples.csv", example_rows)
    _write_json(args.output_dir / "output_checks.json", {"datasets": output_rows})
    _write_markdown(
        path=args.output_dir / "diagnostic_summary.md",
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        dataset_rows=dataset_rows,
        output_rows=output_rows,
        subject_rows=subject_rows,
    )

    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
