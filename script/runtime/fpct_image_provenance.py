from __future__ import annotations

"""Create the immutable source record used when .git is absent in an image."""

import argparse
import hashlib
import json
from pathlib import Path


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == ".fpct_image_provenance.json" or relative.startswith(("local/", ".git/", "__pycache__/")) or "/__pycache__/" in relative:
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file_sha(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    payload = {"schema_version": 1, "head": args.head, "branch": args.branch, "upstream": args.upstream, "tree_sha256": tree_sha(root)}
    (root / ".fpct_image_provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
