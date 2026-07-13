#!/usr/bin/env python3
"""LLM/API pre-flight — tiny sync calls across every model the pipeline uses.

Run this BEFORE generating silver labels or re-priming the MAP cascade, so a
missing key, a revoked model, or a renamed model ID surfaces in seconds instead
of after an expensive batch submit. It hits the same provider SDKs + the same
model IDs production uses:

  OpenAI    (OPENAI_API_KEY)    — L1/L2 voter chat models + text-embedding-3-small
  Anthropic (ANTHROPIC_API_KEY) — L2/L3 voter chat models + silver Opus (claude-opus-4-7)
  Gemini    (GOOGLE_API_KEY)    — L1/L2 voter chat models + gemini-embedding-001

``sync`` does tiny chat + embedding calls per model (auth + model access).
``batch`` submits a tiny 1-request batch to EACH provider's Batch API (the prime
uses batch for all three) with the strict MAP tool schema, polls, and reports per
provider. The OpenAI batch path has failed before (jobs reach "failed" status);
on an OpenAI failure this dumps the OpenAI error file (the real per-request reason).

Usage (from repo root):
  python scripts/check_apis.py sync                       # all providers, chat + embeddings (toy prompt)
  python scripts/check_apis.py sync-real                  # real MAP prompt + structured output, per provider
  python scripts/check_apis.py batch [--wait 600]         # batch submit + poll, all providers
  python scripts/check_apis.py batch-check <provider> <id>  # re-check one batch (openai|claude|gemini)
  python scripts/check_apis.py list [--limit 10]         # list recent batches per provider (recover ids)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

_PROMPT = "Reply with the single word: ok."
_EMBED_INPUT = ["CD30 is positive in ALCL"]


# ── Model inventory (sourced from production config) ──────────────────────────

def _inventory() -> dict[str, list[str]]:
    """{provider: [chat model ids]} — deduped, in tier order."""
    from pipeline.stages.knowledge_extraction.batch.voter_configs import (
        OPENAI_L1, OPENAI_L1_B, OPENAI_L2,
        GEMINI_L1, GEMINI_L2,
        CLAUDE_L2, CLAUDE_L3,
    )
    from eval.silver.generation.generator import DEFAULT_MODEL as SILVER_OPUS

    def _dedupe(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "openai":    _dedupe([OPENAI_L1, OPENAI_L1_B, OPENAI_L2]),
        "anthropic": _dedupe([CLAUDE_L2, CLAUDE_L3, SILVER_OPUS]),
        "gemini":    _dedupe([GEMINI_L1, GEMINI_L2]),
    }


# ── Per-provider tiny chat probes ─────────────────────────────────────────────

def _probe_openai_chat(models: list[str]) -> bool:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("  openai     SKIP — OPENAI_API_KEY not set")
        return False
    from openai import OpenAI
    client = OpenAI(api_key=key)
    ok = True
    for m in models:
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=m, messages=[{"role": "user", "content": _PROMPT}], max_tokens=16,
            )
            got = bool(r.choices[0].message.content)
            ok = ok and got
            print(f"  openai     {m:30s} {'OK' if got else 'OK but empty'}  [{time.time()-t0:.1f}s]")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  openai     {m:30s} FAIL  {type(exc).__name__}: {exc}")
    return ok


def _probe_anthropic_chat(models: list[str]) -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  anthropic  SKIP — ANTHROPIC_API_KEY not set")
        return False
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    ok = True
    for m in models:
        t0 = time.time()
        try:
            r = client.messages.create(
                model=m, max_tokens=16,
                messages=[{"role": "user", "content": _PROMPT}],
            )
            got = bool(r.content)
            ok = ok and got
            print(f"  anthropic  {m:30s} {'OK' if got else 'OK but empty'}  [{time.time()-t0:.1f}s]")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  anthropic  {m:30s} FAIL  {type(exc).__name__}: {exc}")
    return ok


def _probe_gemini_chat(models: list[str]) -> bool:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("  gemini     SKIP — GOOGLE_API_KEY not set")
        return False
    import google.genai as genai
    client = genai.Client(api_key=key)
    ok = True
    for m in models:
        t0 = time.time()
        try:
            r = client.models.generate_content(model=m, contents=_PROMPT)
            got = bool(getattr(r, "text", None))
            ok = ok and got
            print(f"  gemini     {m:30s} {'OK' if got else 'OK but empty'}  [{time.time()-t0:.1f}s]")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  gemini     {m:30s} FAIL  {type(exc).__name__}: {exc}")
    return ok


def _probe_embeddings() -> bool:
    from eval.silver.matching.matcher import EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL
    ok = True
    okey = os.environ.get("OPENAI_API_KEY")
    if okey:
        try:
            from eval.silver.matching.embedders import OpenAIEmbedder
            v = OpenAIEmbedder(okey)(_EMBED_INPUT)
            print(f"  openai     {EMBEDDING_MODEL:30s} OK  dim={len(v[0])}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  openai     {EMBEDDING_MODEL:30s} FAIL  {type(exc).__name__}: {exc}")
    else:
        print(f"  openai     {EMBEDDING_MODEL:30s} SKIP — OPENAI_API_KEY not set")
    gkey = os.environ.get("GOOGLE_API_KEY")
    if gkey:
        try:
            from eval.silver.matching.embedders import GeminiEmbedder
            v = GeminiEmbedder(gkey)(_EMBED_INPUT)
            print(f"  gemini     {GEMINI_EMBEDDING_MODEL:30s} OK  dim={len(v[0])}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  gemini     {GEMINI_EMBEDDING_MODEL:30s} FAIL  {type(exc).__name__}: {exc}")
    else:
        print(f"  gemini     {GEMINI_EMBEDDING_MODEL:30s} SKIP — GOOGLE_API_KEY not set")
    return ok


def cmd_sync() -> int:
    inv = _inventory()
    print("SYNC chat (auth + model access), per provider:")
    c_ok = _probe_openai_chat(inv["openai"])
    a_ok = _probe_anthropic_chat(inv["anthropic"])
    g_ok = _probe_gemini_chat(inv["gemini"])
    print("EMBEDDINGS (sweep + eval matching need these):")
    e_ok = _probe_embeddings()
    all_ok = c_ok and a_ok and g_ok and e_ok
    print(f"\n{'ALL OK' if all_ok else 'SOME CHECKS FAILED — see above'}")
    print("Note: this is the SYNC path. The MAP prime uses the BATCH API for all "
          "three providers — run `batch` to validate that path + the strict tool schema.")
    return 0 if all_ok else 1


def cmd_sync_real() -> int:
    """Real-prompt sync test: the actual MAP chain (build_map_chain +
    with_structured_output(strict=True)) on one real source case, per provider.

    Exercises the full sync prompt-schema mechanism — real MAP system prompt,
    structured-output tool/schema, and the parse into AuditableSummary — over the
    DIRECT-API factories (same keys as batch). Note Gemini sync goes through its
    OpenAI-compatible endpoint, a *different* mechanism from the Gemini batch path
    (native response_schema), so this genuinely adds coverage.
    """
    from eval.silver.data.jsonl_utils import read_jsonl
    from eval.silver.analysis.pipeline_sweep import case_to_file_data
    from eval.silver.data.schemas import SourceCase
    from pipeline.stages.knowledge_extraction.stages.map_stage import _format_sentences
    from pipeline.stages.knowledge_extraction.llm_providers import (
        anthropic_direct_chat, gemini_direct_chat, openai_direct_chat,
    )
    from pipeline.stages.knowledge_extraction.prompts import build_map_chain

    src = Path("eval/data/source_cases_related15.jsonl")
    if not src.exists():
        print(f"source cases not found: {src}", file=sys.stderr)
        return 2
    cases = list(read_jsonl(src, SourceCase))
    if not cases:
        print("no source cases in file", file=sys.stderr)
        return 2
    case = cases[0]
    chunk = case_to_file_data(case)["sentences_with_provenance"][:10]
    inp = {"pmcid": case.pmcid, "chunk_id": "C1", "text": _format_sentences(chunk)}

    inv = _inventory()
    probes = [
        ("openai",    inv["openai"][0],    "OPENAI_API_KEY",    openai_direct_chat),
        ("gemini",    inv["gemini"][0],    "GOOGLE_API_KEY",    gemini_direct_chat),
        ("anthropic", inv["anthropic"][0], "ANTHROPIC_API_KEY", anthropic_direct_chat),
    ]
    print(f"SYNC real-prompt MAP (build_map_chain + structured output) on case {case.case_id}:")
    ok = True
    for pname, model, env, factory in probes:
        if not os.environ.get(env):
            print(f"  {pname:9s} SKIP — {env} not set")
            continue
        t0 = time.time()
        try:
            res = build_map_chain(factory(model)).invoke(inp)
            parsed = res.get("parsed") if isinstance(res, dict) else None
            err = res.get("parsing_error") if isinstance(res, dict) else None
            if parsed is not None:
                print(f"  {pname:9s} {model:28s} OK — parsed AuditableSummary, "
                      f"{len(parsed.findings)} findings  [{time.time()-t0:.1f}s]")
            else:
                ok = False
                print(f"  {pname:9s} {model:28s} NO PARSE — parsing_error={err}  [{time.time()-t0:.1f}s]")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  {pname:9s} {model:28s} FAIL  {type(exc).__name__}: {exc}")
    print(f"\n{'ALL OK' if ok else 'SOME CHECKS FAILED — see above'}")
    return 0 if ok else 1


# ── Batch path (the prime submits batches to ALL THREE providers) ─────────────

# (batch-provider name, env var). Anthropic's batch provider key is "claude".
_BATCH_PROBES = [
    ("openai", "OPENAI_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY"),
    ("gemini", "GOOGLE_API_KEY"),
]


def _batch_model(pname: str, inv: dict[str, list[str]]) -> str:
    return {"openai": inv["openai"][0], "claude": inv["anthropic"][0],
            "gemini": inv["gemini"][0]}[pname]


def cmd_batch(wait: int) -> int:
    from pipeline.stages.knowledge_extraction.batch.dispatch import OPENAI_MAP_TOOL, build_providers
    from pipeline.stages.knowledge_extraction.batch.models import BatchRequest
    inv = _inventory()
    messages = [{"role": "user", "content": "Extract findings from: CD30 is positive in ALCL."}]

    active: list[tuple[str, object, object]] = []  # (pname, provider, job)
    print("Submitting a tiny 1-request batch per provider:")
    for pname, env in _BATCH_PROBES:
        if not os.environ.get(env):
            print(f"  {pname:8s} SKIP — {env} not set")
            continue
        model = _batch_model(pname, inv)
        try:
            prov = build_providers({pname})[pname]
            req = BatchRequest(custom_id=f"check__{pname}", messages=messages,
                               model=model, provider=pname, max_tokens=128)
            job = prov.submit([req], OPENAI_MAP_TOOL)
            print(f"  {pname:8s} submitted  job={job.job_id}  model={model}")
            active.append((pname, prov, job))
        except Exception as exc:  # noqa: BLE001
            print(f"  {pname:8s} SUBMIT-FAIL  {type(exc).__name__}: {exc}")
    if not active:
        print("\nNo batches submitted (no keys set?).")
        return 1

    print(f"\nPolling {len(active)} batch(es), up to {wait}s (submit already validated auth):")
    deadline = time.time() + wait
    pending = list(active)
    rc = 0
    while pending and time.time() < deadline:
        time.sleep(15)
        still = []
        for pname, prov, job in pending:
            prov.check(job)
            if job.status in ("completed", "failed"):
                rc |= _report_batch(pname, prov, job)
            else:
                still.append((pname, prov, job))
        pending = still
    for pname, prov, job in pending:
        print(f"  {pname:8s} still in_progress — re-check: "
              f"python scripts/check_apis.py batch-check {pname} {job.job_id}")
    return rc


def _report_batch(pname: str, prov, job) -> int:
    if job.status == "completed":
        try:
            results = prov.retrieve(job)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pname:8s} completed but retrieve failed: {type(exc).__name__}: {exc}")
            return 1
        ok = all(r.content for r in results)
        for r in results:
            print(f"  {pname:8s} {r.custom_id:18s} {'OK' if r.content else 'ERR: ' + str(r.error)}")
        return 0 if ok else 1
    print(f"  {pname:8s} FAILED (job={job.job_id})")
    if pname == "openai":
        _dump_openai_error_file(job.job_id)
    else:
        try:
            for r in prov.retrieve(job):
                print(f"    {r.custom_id:18s} {'OK' if r.content else 'ERR: ' + str(r.error)}")
        except Exception as exc:  # noqa: BLE001
            print(f"    (no per-request detail available: {type(exc).__name__}: {exc})")
    return 1


def _dump_openai_error_file(batch_id: str) -> None:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b = client.batches.retrieve(batch_id)
    print(f"    status={b.status}  request_counts={getattr(b, 'request_counts', None)}")
    if getattr(b, "errors", None):
        print(f"    batch.errors: {b.errors}")
    efid = getattr(b, "error_file_id", None)
    if not efid:
        print("    no error_file_id (failure may be batch-level, not per-request)")
        return
    txt = client.files.content(efid).text
    print("    --- error file (first 20 lines) ---")
    for line in txt.splitlines()[:20]:
        print("    " + line)


def cmd_list(limit: int) -> int:
    """List recent batch jobs per provider — recover ids of orphaned submissions."""
    import datetime as _dt

    def _ts(v):
        if isinstance(v, (int, float)):
            return _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc).isoformat()
        return str(v)

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            print(f"OpenAI — last {limit} batches:")
            for b in c.batches.list(limit=limit).data:
                print(f"  {b.id}  status={b.status}  created={_ts(b.created_at)}  "
                      f"counts={getattr(b, 'request_counts', None)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  openai list FAIL: {type(exc).__name__}: {exc}")
    else:
        print("OpenAI — SKIP (OPENAI_API_KEY not set)")

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            print(f"Anthropic — last {limit} batches:")
            for b in c.messages.batches.list(limit=limit).data:
                print(f"  {b.id}  status={getattr(b, 'processing_status', '?')}  "
                      f"created={_ts(getattr(b, 'created_at', '?'))}  "
                      f"counts={getattr(b, 'request_counts', None)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  anthropic list FAIL: {type(exc).__name__}: {exc}")
    else:
        print("Anthropic — SKIP (ANTHROPIC_API_KEY not set)")

    # Gemini
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            import google.genai as genai
            c = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            print(f"Gemini — recent batches (up to {limit}):")
            n = 0
            for b in c.batches.list(config={"page_size": limit}):
                print(f"  {b.name}  state={getattr(b, 'state', '?')}  "
                      f"created={_ts(getattr(b, 'create_time', '?'))}")
                n += 1
                if n >= limit:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"  gemini list FAIL: {type(exc).__name__}: {exc}")
    else:
        print("Gemini — SKIP (GOOGLE_API_KEY not set)")
    return 0


def cmd_batch_check(provider: str, batch_id: str) -> int:
    from pipeline.stages.knowledge_extraction.batch.dispatch import build_providers
    from pipeline.stages.knowledge_extraction.batch.models import ProviderJob
    if provider not in {"openai", "claude", "gemini"}:
        print(f"unknown provider {provider!r} (expected openai|claude|gemini)", file=sys.stderr)
        return 2
    prov = build_providers({provider})[provider]
    # claude's retrieve reads output_location == batch_id; openai's check sets it
    # on completion; gemini needs the custom-id list embedded at submit, so a bare
    # re-check can only report status for gemini (re-run `batch` for its content).
    job = ProviderJob(provider=provider, job_id=batch_id, status="submitted",
                      model="?", request_count=0,
                      output_location=batch_id if provider == "claude" else None)
    prov.check(job)
    print(f"{provider} batch {batch_id} status={job.status}")
    if job.status == "in_progress":
        print("  still in progress — re-run later.")
        return 0
    if provider == "gemini" and job.status == "completed":
        print("  completed — but per-request content needs the original submit's id list; "
              "re-run `batch` to validate Gemini output content.")
        return 0
    return _report_batch(provider, prov, job)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM/API pre-flight (sync all providers + batch all providers)")
    ap.add_argument("mode", choices=["sync", "sync-real", "batch", "batch-check", "list"])
    ap.add_argument("args", nargs="*",
                    help="for batch-check: <provider> <batch_id> (provider = openai|claude|gemini)")
    ap.add_argument("--wait", type=int, default=600,
                    help="max seconds to poll submitted batches (default 600)")
    ap.add_argument("--limit", type=int, default=10,
                    help="how many recent batches to show in `list` mode (default 10)")
    args = ap.parse_args()

    if args.mode == "sync":
        return cmd_sync()
    if args.mode == "sync-real":
        return cmd_sync_real()
    if args.mode == "batch":
        return cmd_batch(args.wait)
    if args.mode == "list":
        return cmd_list(args.limit)
    if len(args.args) != 2:
        print("batch-check requires: <provider> <batch_id>  (provider = openai|claude|gemini)",
              file=sys.stderr)
        return 2
    return cmd_batch_check(args.args[0], args.args[1])


if __name__ == "__main__":
    sys.exit(main())
