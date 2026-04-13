"""Runtime safeguards for third-party battle engine failures.

This module intentionally handles *only* failures originating from poke-env/
Showdown internals. Exceptions from this repository are always re-raised.
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_NAME = _REPO_ROOT.name.lower()
_CONFIGURED = False


class ThirdPartyBattleError(RuntimeError):
    """Wraps a fatal poke-env/showdown failure that should not stop training."""


def configure_external_runtime_messages() -> None:
    """Route poke-env/showdown warnings to stderr with a dedicated prefix.

    Keeps application prints on stdout while making third-party diagnostics easy
    to visually separate.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter = logging.Formatter("[poke-env] %(levelname)s: %(message)s")
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    for logger_name in ("poke_env", "websockets"):
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.WARNING)

    original_showwarning = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        filename_l = str(filename).lower()
        if "poke_env" in filename_l or "showdown" in filename_l:
            print(
                f"[poke-env][warning] {category.__name__}: {message}",
                file=sys.stderr,
            )
            return
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning
    _CONFIGURED = True


def is_external_battle_exception(exc: BaseException) -> bool:
    """True iff traceback is from poke-env/showdown and not this repo's code."""

    extracted = traceback.extract_tb(exc.__traceback__)
    if not extracted:
        return False

    has_third_party_frame = False
    has_repo_frame = False

    for frame in extracted:
        fpath = Path(frame.filename).resolve()
        frame_lower = str(fpath).lower()
        if _is_repo_frame(fpath):
            has_repo_frame = True
        if "poke_env" in frame_lower or "showdown" in frame_lower:
            has_third_party_frame = True

    return has_third_party_frame and not has_repo_frame


def _is_repo_frame(frame_path: Path) -> bool:
    """Return True when a traceback frame points at this repository."""

    if frame_path == _REPO_ROOT or _REPO_ROOT in frame_path.parents:
        return True

    parts_lower = [part.lower() for part in frame_path.parts]
    try:
        repo_root_index = parts_lower.index(_REPO_ROOT_NAME)
    except ValueError:
        return False

    relative_parts = frame_path.parts[repo_root_index + 1 :]
    if not relative_parts:
        return False

    return (_REPO_ROOT / Path(*relative_parts)).exists()
