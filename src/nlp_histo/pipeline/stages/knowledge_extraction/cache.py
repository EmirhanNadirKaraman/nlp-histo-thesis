"""
Disk-backed cache for MAP outputs.

The MAP cache key is keyed on (input_hash, schema_version, prompt_version,
stage_name, cascade_signature) so a schema/prompt bump or a cascade reshuffle
naturally invalidates stale entries. Each MAP entry stores the run metadata
alongside the result (``{"metadata": {...}, "result": {...}}``).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .models import (
    AuditableSummary,
    MapRunMetadata,
)

logger = logging.getLogger(__name__)


class PipelineCache:
    """
    MAP-stage cache persisted as a single JSON file.

    MAP key components
    ------------------
    schema_version + prompt_version + stage_name + cascade_signature
      + voter_config_hash + nli_model_id + input_hash

    where ``input_hash = sha256(sorted(text_element_ids))`` is the chunk
    identity. The cascade signature is a stable hash of the ordered
    (provider, model) sequence used at L1+L2+L3 (see
    ``models.compute_cascade_signature``).

    ``voter_config_hash`` captures voter sampler settings — temperature and
    top_p — so editing ``voter_configs.py`` produces a different key even
    when (provider, model) tuples are unchanged. ``nli_model_id`` captures
    the active grounding NLI checkpoint so swapping
    ``$NLP_HISTO_NLI_MODEL`` invalidates persisted entries (their stored
    grounding scores would otherwise be served stale against a different
    NLI model).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._map: dict[str, dict] = {}
        self._hits = {"map": 0}
        self._misses = {"map": 0}
        self._load()

    # Persistence

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._map = data.get("map", {})
            logger.info("Cache loaded: %d map entries", len(self._map))
        except Exception as exc:
            logger.warning("Could not load cache from %s: %s", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"map": self._map}),
            encoding="utf-8",
        )

    # Key builders

    @staticmethod
    def _input_hash(sentences: list[dict]) -> str:
        ids = sorted(s.get("text_element_id", 0) for s in sentences)
        payload = ",".join(map(str, ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _map_key(sentences: list[dict], metadata: MapRunMetadata) -> str:
        """Deterministic versioned key for one MAP chunk.

        Format:
          ``{schema_version}|{prompt_version}|{stage}|{cascade}
            |{voter_config_hash}|{nli_model_id}|{input_hash}``

        ``voter_config_hash`` and ``nli_model_id`` default to ``""`` on
        ``MapRunMetadata`` for backward compatibility; old in-memory metadata
        constructions still hash without crashing but will not collide with
        newer-format keys.
        """
        return "|".join((
            metadata.schema_version,
            metadata.prompt_version,
            metadata.stage_name,
            metadata.cascade_signature,
            metadata.voter_config_hash,
            metadata.nli_model_id,
            PipelineCache._input_hash(sentences),
        ))

    # MAP

    def get_map(
        self,
        sentences: list[dict],
        metadata: MapRunMetadata,
    ) -> Optional[AuditableSummary]:
        """Versioned MAP cache lookup.

        Returns a hit only when the stored entry's ``metadata`` matches the
        caller's ``metadata`` exactly. Old un-wrapped entries are always
        treated as misses (they will be overwritten on the next ``set_map``).
        """
        key = self._map_key(sentences, metadata)
        entry = self._map.get(key)
        if entry is None or "result" not in entry or "metadata" not in entry:
            self._misses["map"] += 1
            return None
        # Belt-and-braces: the key includes versions, but if a future bug
        # writes mismatched metadata under the same key, fail closed.
        # ``nli_model_id`` and ``voter_config_hash`` default to "" for
        # entries written before those fields existed; the equality check
        # against the caller-supplied metadata still treats those as misses
        # when the caller now has non-empty values (the desired behaviour).
        stored_meta = entry["metadata"]
        for field in ("schema_version", "prompt_version", "stage_name",
                      "cascade_signature", "voter_config_hash",
                      "nli_model_id"):
            if stored_meta.get(field, "") != getattr(metadata, field):
                self._misses["map"] += 1
                return None
        self._hits["map"] += 1
        return AuditableSummary(**entry["result"])

    def set_map(
        self,
        sentences: list[dict],
        result: AuditableSummary,
        metadata: MapRunMetadata,
        voter_outputs: list[dict] | None = None,
    ) -> None:
        """Persist a chunk's MAP result.

        ``voter_outputs`` is an optional list of per-voter records
        (matching ``MapStage._buffer_voter_outputs``'s row shape) so cache
        hits on later runs can still populate ``sum_map_voter_outputs``.
        Older entries written before this field existed simply lack the
        key — readers tolerate that and degrade to "no voter outputs
        available for this cache hit".
        """
        entry: dict = {
            "metadata": metadata.model_dump(),
            "result":   result.model_dump(),
        }
        if voter_outputs is not None:
            entry["voter_outputs"] = voter_outputs
        self._map[self._map_key(sentences, metadata)] = entry

    def get_voter_outputs(
        self,
        sentences: list[dict],
        metadata: MapRunMetadata,
    ) -> list[dict] | None:
        """Return the per-voter records captured the last time this chunk's
        MAP result was cached, or None when the entry pre-dates the
        ``voter_outputs`` field (or simply was never persisted).
        """
        key = self._map_key(sentences, metadata)
        entry = self._map.get(key)
        if entry is None:
            return None
        outputs = entry.get("voter_outputs")
        if not isinstance(outputs, list):
            return None
        return outputs

    # Stats

    def stats_str(self) -> str:
        lines = ["Cache stats:"]
        total_h = total_m = 0
        for tier in ("map",):
            h, m = self._hits[tier], self._misses[tier]
            total_h += h
            total_m += m
            rate = f"{h/(h+m):.0%}" if (h + m) else "n/a"
            lines.append(f"  {tier:6}: {h} hits / {m} misses ({rate})")
        overall = f"{total_h/(total_h+total_m):.0%}" if (total_h + total_m) else "n/a"
        lines.append(f"  overall: {overall}")
        return "\n".join(lines)
