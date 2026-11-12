"""Root-level entry point invoking src/evaluate.py to reproduce evaluation from scratch."""
from __future__ import annotations

import sys

from src.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
