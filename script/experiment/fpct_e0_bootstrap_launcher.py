from __future__ import annotations

from pathlib import Path
import runpy
import sys
from typing import Any


HOOK_ID = "fpct_e0_preseal_import_finder_v1"
HOOK_MODULE = "__fpct_e0_preseal_never_imported__"


def _unused_hook(module: Any) -> None:
    del module


def preseal_wandb_import_finder() -> None:
    from wandb.sdk.lib import import_hooks

    import_hooks.register_post_import_hook(
        _unused_hook, HOOK_ID, HOOK_MODULE
    )
    import_hooks.unregister_post_import_hook(HOOK_MODULE, HOOK_ID)
    finders = [
        value for value in sys.meta_path
        if type(value).__module__ == "wandb.sdk.lib.import_hooks"
        and type(value).__qualname__ == "ImportHookFinder"
    ]
    if len(finders) != 1:
        raise RuntimeError("expected exactly one presealed W&B import finder")


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("bootstrap path is required")
    bootstrap = Path(sys.argv[1]).resolve(strict=True)
    expected = Path("/opt/fpct/script/runtime/fpct_bootstrap.py")
    if bootstrap != expected:
        raise RuntimeError(f"unexpected bootstrap path: {bootstrap}")
    forwarded = [str(bootstrap), *sys.argv[2:]]
    preseal_wandb_import_finder()
    sys.argv = forwarded
    runpy.run_path(str(bootstrap), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
