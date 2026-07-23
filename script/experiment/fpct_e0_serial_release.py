from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def kubectl_json(*args: str) -> dict[str, Any]:
    output = subprocess.check_output(["kubectl", *args])
    return json.loads(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--continuation-job", required=True)
    parser.add_argument("--namespace", default="c2c-research")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    marker = args.run_root / "seeds/2026072201/active/seed_complete.json"
    record_path = args.run_root / "provenance/serial_continuation_release.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "RELEASED":
            return 0
        raise RuntimeError(f"existing non-released controller record: {record_path}")

    while True:
        source = kubectl_json(
            "get", "job", args.source_job, "-n", args.namespace, "-o", "json"
        )
        status = source.get("status", {})
        if int(status.get("failed", 0) or 0) > 0:
            atomic_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "SOURCE_FAILED_NOT_RELEASED",
                    "source_job": args.source_job,
                    "continuation_job": args.continuation_job,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 2

        if marker.is_file():
            completion = json.loads(marker.read_text(encoding="utf-8"))
            if completion.get("status") != "COMPLETE":
                raise RuntimeError(f"unexpected seed completion marker: {completion}")
            subprocess.run(
                [
                    "kubectl", "patch", "job", args.continuation_job,
                    "-n", args.namespace, "--type=merge",
                    "-p", '{"spec":{"suspend":false}}',
                ],
                check=True,
            )
            continuation = kubectl_json(
                "get", "job", args.continuation_job,
                "-n", args.namespace, "-o", "json",
            )
            atomic_json(
                record_path,
                {
                    "schema_version": 1,
                    "status": "RELEASED",
                    "source_job": args.source_job,
                    "source_completion": completion,
                    "continuation_job": args.continuation_job,
                    "continuation_uid": continuation["metadata"]["uid"],
                    "released_at": datetime.now(timezone.utc).isoformat(),
                    "accuracy_or_correctness_read": False,
                },
            )
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
