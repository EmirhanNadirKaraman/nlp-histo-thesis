"""Corpus acquisition: download PMC article packages, unpack them, organize the media.

Importing this package performs no network access and touches no filesystem. Each
operation is a plain function taking explicit input and output paths; nothing is
resolved against the caller's working directory.
"""
