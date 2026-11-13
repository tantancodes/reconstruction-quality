"""Archived tests: import bd3d_legacy from archive/src and reuse the core test fixtures."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'archive' / 'src'), str(ROOT / 'tests')]

from fixtures import dataset, fake_api, tetra  # noqa: E402,F401
