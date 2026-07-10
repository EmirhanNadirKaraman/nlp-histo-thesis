"""B-081 live verification: does GeminiBatchProvider map responses to the right
custom_id?

Submits a few requests to BOTH Gemini voters (L1 = gemini-2.5-flash-lite,
L2 = gemini-2.5-flash), each with a UNIQUE marker baked into both its custom_id
and its prompt, then retrieves and checks that the response returned for each
custom_id actually contains THAT request's marker (not another's). This proves
the metadata key-match end to end and is immune to whatever order the backend
returns responses in.

PASS  → fix works live (metadata round-trips; no positional fallback).
FAIL  → cross-wiring survived, or the "matched by POSITION" fallback fired
        (printed below) → metadata is not round-tripping; needs plan B.

Needs GOOGLE_API_KEY. Costs a few tiny batch calls. Blocks while the batch runs.

Run:  python scripts/_gemini_roundtrip_probe.py
"""
import json
import logging
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(str(_REPO / ".env"))

from pipeline.stages.knowledge_extraction.batch.gemini_batch import GeminiBatchProvider
from pipeline.stages.knowledge_extraction.batch.models import BatchRequest

MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]  # L1, L2
N_PER_MODEL = 4
POLL_SECS = 20
MAX_WAIT_SECS = 60 * 30


# Capture any "matched by POSITION" fallback warning from the provider.
class _WarnCatcher(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _requests(model: str) -> list[BatchRequest]:
    reqs = []
    for i in range(N_PER_MODEL):
        marker = f"MARKER_{model.split('-')[-1]}_{i:02d}"  # unique per (model, i)
        reqs.append(BatchRequest(
            custom_id=f"probe__{marker}",
            messages=[{"role": "user",
                       "content": f'Respond with ONLY this exact JSON and nothing else: '
                                  f'{{"marker": "{marker}"}}'}],
            model=model,
            provider="gemini",
            max_tokens=64,
            temperature=0.0,
        ))
    return reqs


def _check_model(provider: GeminiBatchProvider, model: str) -> bool:
    print(f"\n=== {model} ===")
    reqs = _requests(model)
    expected = {r.custom_id: re.search(r"MARKER_[^\"]+", r.messages[0]["content"]).group(0)
                for r in reqs}

    job = provider.submit(reqs, openai_tool={})
    print(f"submitted: {job.job_id} ({len(reqs)} requests)")

    waited = 0
    while True:
        job = provider.check(job)
        if job.status in ("completed", "failed"):
            break
        if waited >= MAX_WAIT_SECS:
            print(f"TIMEOUT after {waited}s (status={job.status}) — batch still running.")
            return False
        time.sleep(POLL_SECS)
        waited += POLL_SECS
        print(f"  …{waited}s status={job.status}")

    if job.status == "failed":
        print("batch FAILED")
        return False

    results = provider.retrieve(job)
    ok = True
    for r in results:
        want = expected.get(r.custom_id)
        if want is None:
            print(f"  ✗ unexpected custom_id returned: {r.custom_id!r}")
            ok = False
            continue
        content = r.content or ""
        got = None
        try:
            got = json.loads(content).get("marker")
        except Exception:
            m = re.search(r"MARKER_[A-Za-z0-9_]+", content)
            got = m.group(0) if m else None
        match = (got == want)
        print(f"  {'✓' if match else '✗'} {r.custom_id} -> marker={got!r} (want {want!r})")
        ok = ok and match
    print(f"{model}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    catcher = _WarnCatcher()
    logging.getLogger("pipeline.stages.knowledge_extraction.batch.gemini_batch").addHandler(catcher)

    provider = GeminiBatchProvider()
    all_ok = all(_check_model(provider, m) for m in MODELS)

    print("\n" + "=" * 50)
    pos_warnings = [m for m in catcher.msgs if "matched by POSITION" in m]
    if pos_warnings:
        print(f"⚠️  {len(pos_warnings)} POSITIONAL-FALLBACK warning(s) fired — metadata did NOT "
              f"round-trip; fix degraded to positional. Plan B needed.")
        for m in pos_warnings:
            print("   " + m)
    else:
        print("✓ no positional-fallback warning — metadata round-tripped (key match active).")
    print(f"OVERALL: {'PASS — B-081 code fix verified live' if all_ok and not pos_warnings else 'FAIL / INCONCLUSIVE'}")
    print("=" * 50)
    sys.exit(0 if (all_ok and not pos_warnings) else 1)


if __name__ == "__main__":
    main()
