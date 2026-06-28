"""SeismicLab — Ingestion engine. Runs all sources on their poll intervals."""

import sqlite3
import time
import logging
import threading
from datetime import datetime, timezone

from .sources import SOURCE_REGISTRY
from .store import QuakeStore

log = logging.getLogger("seismiclab.engine")


class IngestEngine:
    def __init__(self, store: QuakeStore = None, sources: list[str] = None):
        self.store = store or QuakeStore()
        self.sources = sources or list(SOURCE_REGISTRY.keys())
        self._last_fetch = {}
        self._running = False
        self._thread = None

    def fetch_source(self, name: str, retries: int = 2) -> dict:
        entry = SOURCE_REGISTRY.get(name)
        if not entry:
            log.warning(f"Unknown source: {name}")
            return {"source": name, "error": "unknown source"}

        for attempt in range(retries + 1):
            t0 = time.monotonic()
            try:
                samples = entry["fn"]()
                duration_ms = (time.monotonic() - t0) * 1000
                new_count = self.store.ingest(samples, source_name=name)
                self.store.log_ingest(name, len(samples), new_count, duration_ms)
                self._last_fetch[name] = time.time()
                return {
                    "source": name, "samples": len(samples),
                    "new": new_count, "duration_ms": round(duration_ms, 1),
                }
            except sqlite3.OperationalError as e:
                duration_ms = (time.monotonic() - t0) * 1000
                if "locked" in str(e) and attempt < retries:
                    log.warning(f"Source {name} DB locked, retry {attempt + 1}/{retries}")
                    time.sleep(5 * (attempt + 1))
                    continue
                log.error(f"Source {name} failed: {e}")
                self._last_fetch[name] = time.time()
                try:
                    self.store.log_ingest(name, 0, 0, duration_ms, error=str(e))
                except Exception:
                    pass
                return {"source": name, "error": str(e), "duration_ms": round(duration_ms, 1)}
            except Exception as e:
                duration_ms = (time.monotonic() - t0) * 1000
                log.error(f"Source {name} failed: {e}")
                self._last_fetch[name] = time.time()
                try:
                    self.store.log_ingest(name, 0, 0, duration_ms, error=str(e))
                except Exception:
                    pass
                return {"source": name, "error": str(e), "duration_ms": round(duration_ms, 1)}

    def fetch_all(self) -> list[dict]:
        results = []
        for name in self.sources:
            result = self.fetch_source(name)
            results.append(result)
        return results

    def _poll_loop(self):
        log.info(f"Ingestion engine started — {len(self.sources)} sources")
        self.fetch_all()

        while self._running:
            now = time.time()
            for name in self.sources:
                entry = SOURCE_REGISTRY.get(name)
                if not entry:
                    continue

                interval_s = entry["interval_min"] * 60
                last = self._last_fetch.get(name, 0)

                if now - last >= interval_s:
                    try:
                        self.fetch_source(name)
                    except Exception as e:
                        log.error(f"Poll error for {name}: {e}")
            time.sleep(30)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
