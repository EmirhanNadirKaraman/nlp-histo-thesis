"""Reusable evaluation library: dataset contracts, JSONL IO, deterministic
splits, embedding adapters, finding alignment, and the metrics computed from
matched findings.

Deliberately minimal: importing this package reads no repository artifact, loads
no model, and runs no experiment. The thesis experiment drivers (E01–E14, sweeps,
calibration, report generation) are repository-only and import *from* here — never
the reverse.
"""
