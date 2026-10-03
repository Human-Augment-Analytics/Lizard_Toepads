"""Make the ``annotator`` package importable in tests regardless of CWD."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
