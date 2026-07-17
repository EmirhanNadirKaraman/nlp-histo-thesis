"""
eval/llm_judge.py — Backward-compatible wrapper.

Delegates to the new eval.llm_judge package (Phase 2 harness).

Usage (unchanged):
  python eval/llm_judge.py
  python eval/llm_judge.py --n 10 --seed 7 --tests q1,q2
  python eval/llm_judge.py --mode batch --tests q1,q2,q3,q5 --n 15

See eval/llm_judge/__main__.py for all available flags.
"""
from __future__ import annotations

import os
import sys

# Running as `python eval/llm_judge.py` puts eval/ on sys.path, not the project
# root. Insert the parent directory so `import eval` resolves correctly.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main() -> None:
    """Delegate to the new eval.llm_judge package."""
    from eval.llm_judge.__main__ import main as _main
    _main()


if __name__ == "__main__":
    main()
