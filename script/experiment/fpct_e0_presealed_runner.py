from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any


ORIGINAL_RUNNER = Path("/opt/fpct-e0/fpct_e0_runner.py")
BOOTSTRAP = "/opt/fpct/script/runtime/fpct_bootstrap.py"
LAUNCHER = "/opt/fpct-e0-recovery/fpct_e0_bootstrap_launcher.py"


def load_original() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fpct_e0_original_runner", ORIGINAL_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIGINAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument(
        "--config-map-root", type=Path, default=Path("/opt/fpct-e0")
    )
    parser.add_argument("--run-root", type=Path, default=Path("/fpct-e0"))
    args = parser.parse_args()

    original = load_original()
    original_command = original._training_command

    def presealed_command(
        seed: int, arm: str, config: Path, attestation: Path
    ) -> list[str]:
        command = original_command(seed, arm, config, attestation)
        positions = [
            index for index, value in enumerate(command)
            if value == BOOTSTRAP
        ]
        if positions != [8]:
            raise RuntimeError(
                f"unexpected bootstrap command position: {positions}"
            )
        command.insert(positions[0], LAUNCHER)
        return command

    original._training_command = presealed_command
    payload = original.run_seed(
        args.seed,
        args.attempt,
        args.config_map_root.resolve(),
        args.run_root.resolve(),
    )
    print(original.json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
