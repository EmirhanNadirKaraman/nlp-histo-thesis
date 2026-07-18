#!/usr/bin/env python3
"""Compatibility wrapper — the implementation now lives in the installed package.

    python scripts/thesis/run_chapter9_offline_replay.py --artifact-root .
      ==  nlp-histo replay results --artifact-root .

NOTE: --artifact-root is now REQUIRED. The replay used to infer the repository root
from its own file location and silently read out/ and eval/ from there; an installed
package has no repository beside it, so the artifact tree is explicit.

The `chapter9` in this filename is stale — it dates from an eleven-chapter markdown
draft whose results chapter was numbered 9. The thesis has six chapters and this
replay feeds chapter 4. The filename is deliberately NOT renamed: this module exists
to keep the old invocation path working, so renaming it would defeat its purpose.

REMOVAL PLAN: delete once the documentation pass has replaced this invocation with
`nlp-histo replay results`.
"""
from nlp_histo.workflows.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
