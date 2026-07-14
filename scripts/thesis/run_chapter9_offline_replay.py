#!/usr/bin/env python3
"""Compatibility wrapper — the implementation now lives in the installed package.

    python scripts/thesis/run_chapter9_offline_replay.py --artifact-root .
      ==  nlp-histo replay chapter9 --artifact-root .

NOTE: --artifact-root is now REQUIRED. The replay used to infer the repository root
from its own file location and silently read out/ and eval/ from there; an installed
package has no repository beside it, so the artifact tree is explicit.

REMOVAL PLAN: delete once the documentation pass has replaced this invocation with
`nlp-histo replay chapter9`.
"""
from nlp_histo.workflows.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
