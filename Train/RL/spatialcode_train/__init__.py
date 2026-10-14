from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_RL_ROOT = _PKG_DIR.parent

# Expose Train/RL/trainer and Train/RL/config as spatialcode_train subpackages.
__path__ = [str(_PKG_DIR), str(_RL_ROOT)]
