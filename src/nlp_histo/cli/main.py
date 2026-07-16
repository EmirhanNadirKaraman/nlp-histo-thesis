"""The ``nlp-histo`` console entry point.

One command, subcommands per stage of the pipeline::

    nlp-histo db init | check
    nlp-histo acquire download | unpack | organize
    nlp-histo ingest
    nlp-histo ner extract | merge | export
    nlp-histo knowledge          [may make PAID API calls]
    nlp-histo replay chapter9    [offline, free]

Every handler imports its implementation *lazily*, inside the function that runs it.
That is what keeps ``--help`` cheap and safe: printing help must never open a database
connection, load scispaCy, initialise UMLS, construct a provider SDK client, or create
a cache or output directory. Importing this module does none of those things either.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

COST_WARNING = (
    "WARNING: this command makes PAID LLM API calls. Cost scales with the number of "
    "papers and the cascade profile. Use a cheap profile and a small --limit first."
)


# ── handlers: each imports its workflow lazily, then delegates ────────────────

def _db(args: argparse.Namespace) -> int:
    from nlp_histo.database import init_db

    return init_db.main(["--check-only"] if args.db_command == "check" else [])


def _acquire(args: argparse.Namespace) -> int:
    if args.acquire_command == "download":
        from nlp_histo.acquisition.downloader import download_papers

        report = download_papers(
            args.pmcid_file, args.output_dir, overwrite=args.overwrite
        )
        # ANY failure fails the run, even beside successes: a corpus quietly missing the
        # papers NCBI said it had is worse than a red exit code. The return value used to
        # be discarded entirely, so even a total failure exited 0 (B-117).
        #
        # Skips are not failures — a paper outside the OA subset, or one already on disk,
        # is an expected answer rather than a denied request.
        #
        # Successful archives are kept on a partial failure; only unusable ones are
        # removed (see download_file), so a re-run resumes instead of refetching.
        if not report.ok:
            print(
                f"error: {report.failed} of {report.requested} requested paper(s) failed "
                f"to download — {report.summary()}. See the per-PMCID reasons above. "
                "A 404 on a link NCBI advertised means the OA package it lists is not "
                "retrievable: an upstream problem, not a local one. Successfully "
                "downloaded archives have been kept; re-running resumes.",
                file=sys.stderr,
            )
            return 1
    elif args.acquire_command == "unpack":
        from nlp_histo.acquisition.tarballs import unpack_tarballs

        unpack_tarballs(args.input_dir, args.output_dir, overwrite=args.overwrite)
    else:  # organize
        from nlp_histo.acquisition.organizer import organize_pdfs

        organize_pdfs(args.input_dir, args.pdf_dir, args.xml_dir)
    return 0


def _echo_db_target() -> None:
    """Announce the resolved database before a command touches it.

    `db init` / `db check` always did; `ingest` and `ner` did not, so a run silently
    redirected by an inherited DB_* variable gave no sign of where it was writing —
    which is how a test ingest nearly landed in the production corpus (B-113).
    Best-effort: never let the announcement itself break the command.
    """
    try:
        from nlp_histo.database import get_db_connection
        from nlp_histo.database.env_routing import print_target

        print_target(get_db_connection())
    except Exception:  # noqa: BLE001 — diagnostics must not gate real work
        pass


def _ingest(args: argparse.Namespace) -> int:
    from nlp_histo.pipeline.stages.pdf_text_extraction import runner

    _echo_db_target()
    runner.main(args.forwarded)
    return 0


def _ner(args: argparse.Namespace) -> int:
    if args.ner_command == "extract":
        from nlp_histo.ner import batch_ner as mod
    elif args.ner_command == "merge":
        from nlp_histo.ner import merge_entities_by_umls as mod
    else:  # export
        from nlp_histo.ner import export_disease_entities as mod
    _echo_db_target()
    return mod.main(args.forwarded)


def _knowledge(args: argparse.Namespace) -> int:
    from nlp_histo.workflows import knowledge

    return knowledge.main(args.forwarded)


def _replay(args: argparse.Namespace) -> int:
    from nlp_histo.workflows import replay

    return replay.main(args.forwarded)


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlp-histo",
        description="Auditable knowledge extraction from histopathology literature.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # db ----------------------------------------------------------------------
    db = sub.add_parser("db", help="Create or verify the database schema.")
    db_sub = db.add_subparsers(dest="db_command", metavar="<subcommand>")
    db_sub.add_parser("init", help="Create and verify the schema (safe to re-run).")
    db_sub.add_parser("check", help="Verify the schema only; create nothing.")
    db.set_defaults(handler=_db)

    # acquire -----------------------------------------------------------------
    acquire = sub.add_parser("acquire", help="Download and prepare the PMC corpus.")
    acq_sub = acquire.add_subparsers(dest="acquire_command", metavar="<subcommand>")

    dl = acq_sub.add_parser("download", help="Download OA article packages for a PMCID list.")
    dl.add_argument("--pmcid-file", type=_path, required=True,
                    help="Text file of PMCIDs, one per line.")
    dl.add_argument("--output-dir", type=_path, required=True,
                    help="Directory to write <PMCID>.tar.gz into.")
    dl.add_argument("--overwrite", action="store_true",
                    help="Re-download tarballs that already exist (default: skip).")

    un = acq_sub.add_parser("unpack", help="Extract downloaded tarballs.")
    un.add_argument("--input-dir", type=_path, required=True, help="Directory of *.tar.gz.")
    un.add_argument("--output-dir", type=_path, required=True, help="Where to extract.")
    un.add_argument("--overwrite", action="store_true",
                    help="Re-extract papers that already exist (default: skip).")

    org = acq_sub.add_parser("organize", help="Flatten extracted papers into PDF/XML trees.")
    org.add_argument("--input-dir", type=_path, required=True, help="Extracted corpus directory.")
    org.add_argument("--pdf-dir", type=_path, required=True, help="Destination for PDFs.")
    org.add_argument("--xml-dir", type=_path, required=True, help="Destination for XMLs.")

    acquire.set_defaults(handler=_acquire)

    # ingest ------------------------------------------------------------------
    ingest = sub.add_parser(
        "ingest",
        help="Extract text, figures and tables from PDFs into the database.",
        description="PDF → hierarchical text + media → PostgreSQL. "
                    "Options are passed through to the extraction runner; "
                    "see `nlp-histo ingest -- --help`.",
    )
    ingest.set_defaults(handler=_ingest)

    # ner ---------------------------------------------------------------------
    ner = sub.add_parser("ner", help="Named-entity recognition over ingested documents.")
    ner_sub = ner.add_subparsers(dest="ner_command", metavar="<subcommand>")
    for name, helptext in (
        ("extract", "Run scispaCy/UMLS NER over the corpus (--entity-cache supported)."),
        ("merge", "Merge entities by UMLS CUI."),
        ("export", "Export disease entities."),
    ):
        ner_sub.add_parser(name, help=helptext)
    ner.set_defaults(handler=_ner)

    # knowledge ---------------------------------------------------------------
    knowledge = sub.add_parser(
        "knowledge",
        help="LLM knowledge extraction over ingested papers. [PAID API CALLS]",
        description=COST_WARNING,
    )
    knowledge.set_defaults(handler=_knowledge)

    # replay ------------------------------------------------------------------
    replay = sub.add_parser("replay", help="Reproduce thesis results from frozen artifacts.")
    rep_sub = replay.add_subparsers(dest="replay_command", metavar="<subcommand>")
    rep_sub.add_parser(
        "chapter9",
        help="Chapter-9 offline replay (offline, free; --artifact-root is required).",
    )
    replay.set_defaults(handler=_replay)

    return parser


def _path(value: str):
    from pathlib import Path

    return Path(value)


# Commands whose remaining arguments belong to the underlying workflow, mapped to the
# number of tokens the CLI itself consumes ("ner extract" → 2, "knowledge" → 1).
_FORWARDING = {("ingest",): 1, ("knowledge",): 1,
               ("ner", "extract"): 2, ("ner", "merge"): 2, ("ner", "export"): 2,
               ("replay", "chapter9"): 2}


def _split_forwarded(argv: list[str]) -> tuple[list[str], list[str]]:
    """Return (args_for_this_cli, args_forwarded_to_the_workflow).

    argparse's REMAINDER cannot be used here: it does not capture a *leading* option,
    so `nlp-histo ner extract --entity-cache X` would be rejected as an unknown option
    of the parent parser. Slicing the raw argv after the command path is unambiguous.

    `--help` immediately after the command path is ours (it prints this CLI's help,
    including the cost warning); anything else goes to the workflow.

    A leading `--` forces everything after it to the workflow, `--help` included — that
    is how the workflow's own options are discovered (`nlp-histo ingest -- --help`).
    The `--` must be dropped rather than forwarded: argparse would read it as the
    positional separator and then reject `--help` as an unrecognized positional.
    """
    for path, consumed in _FORWARDING.items():
        if tuple(argv[:consumed]) == path:
            rest = argv[consumed:]
            if rest[:1] == ["--"]:
                return list(path), rest[1:]
            if rest[:1] and rest[0] in ("-h", "--help"):
                return argv, []
            return list(path), rest
    return argv, []


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    own, forwarded = _split_forwarded(raw)

    parser = build_parser()
    args = parser.parse_args(own)
    args.forwarded = forwarded

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    # A command group invoked without its subcommand: show that group's help.
    for group, dest in (
        ("db", "db_command"),
        ("acquire", "acquire_command"),
        ("ner", "ner_command"),
        ("replay", "replay_command"),
    ):
        if args.command == group and getattr(args, dest, None) is None:
            parser.parse_args([group, "--help"])
            return 1

    try:
        return handler(args)
    except Exception as exc:  # noqa: BLE001 — re-raised below unless it is ours
        # An env-routing conflict is raised while `nlp_histo.database` is imported, deep
        # inside a handler. It is a user-configuration problem with an actionable
        # message, so render it as an error rather than a traceback (B-113). Anything
        # else is a real fault and keeps its traceback.
        if type(exc).__name__ == "EnvRoutingConflict":
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
