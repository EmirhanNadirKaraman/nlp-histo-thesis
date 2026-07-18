"""
eval.llm_judge — LLM-as-judge evaluation harness for the histopathology
knowledge_extraction / rule-extraction pipeline.

Uses Claude Opus as a silver-label judge / generator.
Labels produced here are proxy labels, not clinical ground truth.

Phase 1 tests:
  Q1  MAP finding precision + field correctness
  Q2  RELATE label accuracy (blind judging by default)
  Q3  Recall gap check (with + zero extraction paragraphs)
  Q5  Paragraph-level extraction F1

Usage:
  python -m eval.llm_judge --mode sync --tests q1,q2,q3,q5 --n 5
  python -m eval.llm_judge --mode batch --tests q1 --n 15
"""
from __future__ import annotations

# Version constants — bump when prompt or schema changes to invalidate cache.
PROMPT_VERSION = "v2"
SCHEMA_VERSION = "v1"
MODEL = "claude-opus-4-7"
LABEL_SOURCE = "claude_opus_silver"
