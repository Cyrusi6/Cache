"""Rosetta research package.

This file intentionally has no import side effects.  Its presence makes the
repository's ``rosetta`` tree a regular package so an unrelated editable or
namespace-package installation cannot merge into ``rosetta.__path__``.
"""

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parent)]
