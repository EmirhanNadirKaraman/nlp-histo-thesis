#!/usr/bin/env python3
"""Compatibility wrapper — the implementation now lives in the installed package.

    python scripts/run_paper.py …   ==   nlp-histo knowledge …

Kept so existing commands (and HOW_TO_RUN.md, until the documentation pass) keep
working. It contains no logic and no argument parser of its own: it forwards to
``nlp_histo.workflows.knowledge``, which the CLI calls too, so there is exactly one
implementation.

REMOVAL PLAN: delete once the documentation pass has replaced every
`python scripts/run_paper.py` invocation with `nlp-histo knowledge`.
"""
from nlp_histo.workflows.knowledge import main

if __name__ == "__main__":
    raise SystemExit(main())
