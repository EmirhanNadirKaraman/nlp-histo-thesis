"""
BlacklistManager

Thread-safe JSON-backed map of PMC IDs that previously caused processing
failures.  Each entry is stored together with the *reason* it failed and a
timestamp, so the on-disk JSON records a separate reason per item::

    {
      "blacklisted": {
        "PMC123_main": {"reason": "too large (12000 rows > 8000)",
                         "blacklisted_at": "2026-06-03T10:11:12.0"},
        "PMC456_main": {"reason": "Tensor on device cpu ... meta!",
                         "blacklisted_at": "2026-06-03T10:11:30.0"}
      },
      "last_updated": "2026-06-03T10:11:30.0"
    }

Persisted to disk after every mutation.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class BlacklistManager:
    """
    Thread-safe blacklist for failed documents, keyed by PMC ID with a
    per-item failure reason.

    Parameters
    ----------
    path:
        Path to the JSON file.  Created automatically if it does not exist.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path("out/failed_pdfs_blacklist.json")
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = self._load()

    # Public API

    def contains(self, pmcid: str) -> bool:
        with self._lock:
            return pmcid in self._entries

    def add(self, pmcid: str, reason: str = "") -> None:
        """Blacklist ``pmcid``, recording its own ``reason`` and a timestamp."""
        with self._lock:
            self._entries[pmcid] = {
                "reason": reason or "(no reason recorded)",
                "blacklisted_at": datetime.now().isoformat(),
            }
            self._persist()

    def reason(self, pmcid: str) -> Optional[str]:
        """Return the recorded failure reason for ``pmcid`` (or ``None``)."""
        with self._lock:
            entry = self._entries.get(pmcid)
            return entry.get("reason") if entry else None

    def remove(self, pmcid: str) -> None:
        with self._lock:
            self._entries.pop(pmcid, None)
            self._persist()

    def all(self) -> frozenset:
        with self._lock:
            return frozenset(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # Persistence

    def _load(self) -> dict:
        """Load the blacklist, tolerating both the new per-item map and the
        legacy flat-list format (``{"blacklisted": ["PMC...", ...]}``)."""
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

        raw = data.get("blacklisted", {}) if isinstance(data, dict) else {}
        out: dict[str, dict] = {}
        if isinstance(raw, dict):
            # New format: {pmcid: {"reason": ..., "blacklisted_at": ...}}.
            # Be lenient: a bare-string value is treated as the reason.
            for pmcid, val in raw.items():
                if isinstance(val, dict):
                    out[pmcid] = {
                        "reason": val.get("reason", "(no reason recorded)"),
                        "blacklisted_at": val.get("blacklisted_at"),
                    }
                else:
                    out[pmcid] = {"reason": str(val), "blacklisted_at": None}
        elif isinstance(raw, list):
            # Legacy format: a flat list of pmcids with no per-item reason.
            for pmcid in raw:
                out[pmcid] = {
                    "reason": "(legacy entry — reason not recorded)",
                    "blacklisted_at": None,
                }
        return out

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "blacklisted":  {k: self._entries[k] for k in sorted(self._entries)},
            "last_updated": datetime.now().isoformat(),
        }
        self._path.write_text(json.dumps(data, indent=2))
