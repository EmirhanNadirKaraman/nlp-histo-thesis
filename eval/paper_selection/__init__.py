"""Offline paper-selection module.

Builds per-paper fingerprints (entities + regex + keyword features) from the
ingestion DB or JSONL exports, then chooses a 15-paper calibration set:
5 highly related, 5 diverse, 5 hard. No LLM/API calls.

Public entry point::

    python -m eval.paper_selection.run_select --output-version calibration_set_v1
"""

from .models import (
    HardnessBreakdown,
    PaperFingerprint,
    SelectionResult,
)

__all__ = ["HardnessBreakdown", "PaperFingerprint", "SelectionResult"]
