from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
from typing import Any
from urllib import request


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class JobClient:
    def __init__(self, namespace: str, in_cluster: bool) -> None:
        self.namespace = namespace
        self.in_cluster = in_cluster
        if in_cluster:
            host = os.environ["KUBERNETES_SERVICE_HOST"]
            port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
            self.base = f"https://{host}:{port}/apis/batch/v1/namespaces/{namespace}/jobs"
            service_account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
            self.token = (service_account / "token").read_text(encoding="utf-8").strip()
            self.context = ssl.create_default_context(cafile=str(service_account / "ca.crt"))

    def _request(self, method: str, job: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        req = request.Request(
            f"{self.base}/{job}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/merge-patch+json",
            },
        )
        with request.urlopen(req, context=self.context, timeout=30) as response:
            return json.loads(response.read())

    def get(self, job: str) -> dict[str, Any]:
        if self.in_cluster:
            return self._request("GET", job)
        output = subprocess.check_output(
            ["kubectl", "get", "job", job, "-n", self.namespace, "-o", "json"]
        )
        return json.loads(output)

    def release(self, job: str) -> dict[str, Any]:
        if self.in_cluster:
            return self._request("PATCH", job, {"spec": {"suspend": False}})
        subprocess.run(
            [
                "kubectl", "patch", "job", job,
                "-n", self.namespace, "--type=merge",
                "-p", '{"spec":{"suspend":false}}',
            ],
            check=True,
        )
        return self.get(job)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--continuation-job", required=True)
    parser.add_argument("--namespace", default="c2c-research")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--in-cluster", action="store_true")
    args = parser.parse_args()
    client = JobClient(args.namespace, args.in_cluster)

    marker = args.run_root / "seeds/2026072201/active/seed_complete.json"
    record_path = args.run_root / "provenance/serial_continuation_release.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "RELEASED":
            return 0
        raise RuntimeError(f"existing non-released controller record: {record_path}")

    while True:
        source = client.get(args.source_job)
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
            continuation = client.release(args.continuation_job)
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
