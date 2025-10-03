# unav/navigator/i18n_labels.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, threading
from typing import Dict, Any

_DEFAULT_REL = "_i18n/labels.json"
_SECTIONS = ("places", "buildings", "floors", "destinations", "aliases")

class I18NLabels:
    """Thread-safe, mtime-aware reader for <DATA_FINAL_ROOT>/_i18n/labels.json"""
    def __init__(self, data_root: str, rel_path: str = _DEFAULT_REL, default_lang: str = "en"):
        self.path = os.path.join(data_root, rel_path)
        self.default_lang = default_lang
        self._lock = threading.RLock()
        self._labels: Dict[str, Any] = {}
        self._mtime: float | None = None

    def _ensure_loaded(self) -> None:
        try:
            st = os.stat(self.path)
            mtime = st.st_mtime
        except FileNotFoundError:
            with self._lock:
                self._labels = {s: {} for s in _SECTIONS}
                self._mtime = None
            return
        with self._lock:
            if self._mtime is not None and self._mtime == mtime:
                return
            try:
                raw = json.loads(open(self.path, "r", encoding="utf-8").read())
            except Exception:
                raw = {}
            data: Dict[str, Any] = {}
            for s in _SECTIONS:
                v = raw.get(s, {})
                data[s] = v if isinstance(v, dict) else {}
            self._labels = data
            self._mtime = mtime

    def label(self, section: str, key: str, lang: str, fallback: str) -> str:
        """Return label in lang -> fallback to 'en' -> fallback string."""
        self._ensure_loaded()
        entry = (self._labels.get(section, {}) or {}).get(key, {})
        if isinstance(entry, dict):
            s = entry.get(lang) or entry.get(self.default_lang)
            if isinstance(s, str) and s.strip():
                return s.strip()
        return fallback

    def alias(self, lang: str) -> Dict[str, str]:
        self._ensure_loaded()
        amap = (self._labels.get("aliases", {}) or {}).get(lang, {})
        return amap if isinstance(amap, dict) else {}
