"""The public ``nlp-histo`` command-line interface.

Importing this package pulls in nothing heavier than ``argparse``: the workflow
implementations are imported lazily inside each subcommand handler, so ``--help``
never touches a database, a model, an API client, or the filesystem.
"""
