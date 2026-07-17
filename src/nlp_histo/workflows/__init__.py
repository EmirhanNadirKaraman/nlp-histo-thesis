"""End-to-end workflows that the public CLI drives.

These are the orchestration layer: they compose the pipeline, database and
evaluation packages into the operations a user actually runs. Importing a workflow
performs no work — every entry point is a plain function or a ``main(argv)``.
"""
