from __future__ import annotations

import hashlib

from script.analysis.fpct_e1_split_lock import DOMAIN, rank_sha


def test_e1_rank_is_domain_separated_and_deterministic() -> None:
    task = "ai2-arc"
    group = "a" * 64
    expected = hashlib.sha256(DOMAIN + task.encode() + b"\0" + group.encode()).hexdigest()
    assert rank_sha(task, group) == expected
    assert rank_sha(task, group) == rank_sha(task, group)
    assert rank_sha("openbookqa", group) != expected


def test_e1_domain_is_frozen() -> None:
    assert DOMAIN == b"fpct-e1-pilot-v1\0"
