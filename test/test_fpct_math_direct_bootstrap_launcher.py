from __future__ import annotations

import sys

from script.experiment.fpct_math_direct_bootstrap_launcher import (
    preseal_wandb_import_finder,
)


def test_preseal_wandb_import_finder_is_single_and_idempotent() -> None:
    preseal_wandb_import_finder()
    preseal_wandb_import_finder()
    finders = [
        value
        for value in sys.meta_path
        if type(value).__module__ == "wandb.sdk.lib.import_hooks"
        and type(value).__qualname__ == "ImportHookFinder"
    ]
    assert len(finders) == 1
