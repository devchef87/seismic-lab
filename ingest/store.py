"""QuakeWatch — SQLite storage for time-aligned multi-signal data."""

import sqlite3
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import math

from .sources import Sample

log = logging.getLogger("quakewatch.store")

DEFAULT_DB = Path(__file__).parent.parent / "data" / "quakewatch.db"

# --- cross-agency earthquake de-duplication (USGS + EMSC report the same quake) ---
# The same physical event arrives from two agencies under different id schemes, with
# slightly different time/location/magnitude. Collapse to one row at ingest time so the
# model doesn't double-count. Thresholds match scripts/dedup_earthquakes.py.
_EQ_DT_S = 30.0
_EQ_DIST_KM = 50.0
_EQ_DMAG = 1.0


def _eq_group(eid: str) -> str:
    # one group per reporting contributor; coincident events from DIFFERENT contributors
    # are the same physical quake (USGS `us`, tsunami centers `at`/`pt`, regional nets
    # `ak`/`hv`/`pr`/..., and EMSC each solve it independently). See dedup_earthquakes.py.
    if eid.startswith("emsc:"):
        return "emsc"   # current EMSC ingest
    if eid.startswith("emsc_"):
        return "emscL"  # legacy EMSC ingest
    return eid[:2]      # USGS-family network code: us, ak, at, ci, nc, hv, pr, tx, ...


def _haversine_km(la1, lo1, la2, lo2):
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _find_eq_twin(conn, eid, ts, lat, lon, mag):
    """Return an existing earthquake id that is the SAME physical event from a
    DIFFERENT source group (so a USGS event won't suppress another USGS event),
    else None. Uses the (lat,lon,timestamp) index for a cheap candidate prefilter."""
    if lat is None or lon is None or not ts:
        return None
    try:
        t = datetime.fromisoformat(ts)
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    lo = (t - timedelta(seconds=_EQ_DT_S)).isoformat()
    hi = (t + timedelta(seconds=_EQ_DT_S)).isoformat()
    g = _eq_group(eid)
    m = mag if mag is not None else 0.0
    for rid, rla, rlo, rmag in conn.execute(
        "SELECT id, lat, lon, magnitude FROM earthquakes "
        "WHERE timestamp BETWEEN ? AND ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
        (lo, hi, lat - 0.5, lat + 0.5, lon - 0.5, lon + 0.5),
    ):
        if rid == eid or _eq_group(rid) == g or rla is None or rlo is None:
            continue
        if abs((rmag if rmag is not None else 0.0) - m) > _EQ_DMAG:
            continue
        if _haversine_km(lat, lon, rla, rlo) <= _EQ_DIST_KM:
            return rid
    return None


class QuakeStore:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_analyzer()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    lat REAL,
                    lon REAL,
                    meta TEXT,
                    ingested_at TEXT NOT NULL,
                    UNIQUE(source, metric, timestamp, lat, lon)
                );

                CREATE INDEX IF NOT EXISTS idx_samples_source_ts
                    ON samples(source, timestamp);
                CREATE INDEX IF NOT EXISTS idx_samples_metric_ts
                    ON samples(metric, timestamp);
                CREATE INDEX IF NOT EXISTS idx_samples_ts
                    ON samples(timestamp);

                CREATE TABLE IF NOT EXISTS earthquakes (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    magnitude REAL NOT NULL,
                    depth_km REAL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    place TEXT,
                    type TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_eq_ts ON earthquakes(timestamp);
                CREATE INDEX IF NOT EXISTS idx_eq_mag ON earthquakes(magnitude);

                CREATE TABLE IF NOT EXISTS ingest_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    sample_count INTEGER,
                    new_count INTEGER,
                    duration_ms REAL,
                    error TEXT
                );
            """)

    def _init_analyzer(self):
        try:
            from event_analyzer import EventAnalyzer
            self._analyzer = EventAnalyzer(self.db_path)
            log.info("EventAnalyzer initialized — M4.5+ events will be auto-analyzed")
        except Exception as e:
            log.warning(f"EventAnalyzer not available: {e}")
            self._analyzer = None
        try:
            from llm_analyzer import LLMAnalyzer
            self._llm_analyzer = LLMAnalyzer(self.db_path)
            log.info("LLMAnalyzer initialized — high-risk events get Claude Opus analysis")
        except Exception as e:
            log.warning(f"LLMAnalyzer not available: {e}")
            self._llm_analyzer = None

    def _trigger_analysis(self, eq_id, mag, lat, lon, depth_km, place, timestamp):
        if not self._analyzer or mag < 4.5:
            return
        def _run():
            try:
                result = self._analyzer.analyze(eq_id, mag, lat, lon, depth_km, place, timestamp)
                if result and self._llm_analyzer and self._llm_analyzer.should_auto_analyze(
                        result.get("magnitude", 0), result.get("risk_score", 0)):
                    self._llm_analyzer.analyze(result)
            except Exception as e:
                log.error(f"EventAnalyzer failed for {eq_id}: {e}")
        threading.Thread(target=_run, name=f"analyze-{eq_id}", daemon=True).start()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=120000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ingest(self, samples: list[Sample], source_name: str = "") -> int:
        now = datetime.now(timezone.utc).isoformat()
        new_count = 0
        batch_size = 500

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            with self._conn() as conn:
                for s in batch:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO samples
                            (source, metric, timestamp, value, unit, lat, lon, meta, ingested_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            s.source, s.metric, s.timestamp, s.value, s.unit,
                            s.lat, s.lon, json.dumps(s.meta) if s.meta else None, now,
                        ))
                        if conn.total_changes:
                            new_count += 1
                    except sqlite3.IntegrityError:
                        pass

                    if s.source in ("usgs_earthquake", "emsc_earthquake", "seedlink_detection") and s.metric == "magnitude":
                        eq_id = s.meta.get("id", "")
                        # skip cross-agency twins (same quake already in from the other
                        # catalog) so the two-source feed doesn't double-count events
                        twin = None
                        if eq_id and s.source in ("usgs_earthquake", "emsc_earthquake"):
                            twin = _find_eq_twin(conn, eq_id, s.timestamp, s.lat, s.lon, s.value)
                        if eq_id and twin is None:
                            cursor = conn.execute("""
                                INSERT OR IGNORE INTO earthquakes
                                (id, timestamp, magnitude, depth_km, lat, lon, place, type)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                eq_id, s.timestamp, s.value, s.meta.get("depth_km"),
                                s.lat, s.lon, s.meta.get("place", ""),
                                s.meta.get("type", "earthquake"),
                            ))
                            if cursor.rowcount > 0:
                                self._trigger_analysis(
                                    eq_id, s.value, s.lat, s.lon,
                                    s.meta.get("depth_km", 0),
                                    s.meta.get("place", ""),
                                    s.timestamp)

        log.info(f"Ingested {source_name}: {new_count} new / {len(samples)} total")
        return new_count

    def bulk_insert_earthquakes(self, events: list[dict]) -> int:
        """Direct bulk insert into the earthquakes table for historical backfill.

        Bypasses the live EventAnalyzer / LLM-analysis trigger in ingest() — those
        are for real-time events, not for backfilling thousands of historical ones.
        Dedup is by `id` PRIMARY KEY (INSERT OR IGNORE); spatiotemporal dedup against
        other catalogs must be done by the caller before this point.
        """
        new_count = 0
        batch_size = 1000
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            with self._conn() as conn:
                for e in batch:
                    cur = conn.execute("""
                        INSERT OR IGNORE INTO earthquakes
                        (id, timestamp, magnitude, depth_km, lat, lon, place, type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        e["id"], e["timestamp"], e["magnitude"], e.get("depth_km"),
                        e["lat"], e["lon"], e.get("place", ""), e.get("type", "earthquake"),
                    ))
                    new_count += cur.rowcount
        return new_count

    def log_ingest(self, source: str, sample_count: int, new_count: int,
                   duration_ms: float, error: str = None):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO ingest_log (source, fetched_at, sample_count, new_count, duration_ms, error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source, datetime.now(timezone.utc).isoformat(),
                  sample_count, new_count, duration_ms, error))

    def query_samples(self, source: str = None, metric: str = None,
                      start: str = None, end: str = None, limit: int = 10000,
                      bucket_minutes: int = None) -> list[dict]:
        clauses = []
        params = []

        if source:
            clauses.append("source = ?")
            params.append(source)
        if metric:
            clauses.append("metric = ?")
            params.append(metric)
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if bucket_minutes and bucket_minutes > 0:
                bucket_secs = bucket_minutes * 60
                sql = (
                    f"SELECT strftime('%Y-%m-%dT%H:%M:%SZ', (CAST(strftime('%s', timestamp) AS INTEGER) / {bucket_secs}) * {bucket_secs}, 'unixepoch') as timestamp,"
                    f" source, metric, AVG(value) as value"
                    f" FROM samples {where}"
                    f" GROUP BY 1, source, metric"
                    f" ORDER BY 1 ASC"
                )
                rows = conn.execute(sql, params).fetchall()
            else:
                params.append(limit)
                rows = conn.execute(
                    f"SELECT * FROM samples {where} ORDER BY timestamp DESC LIMIT ?", params
                ).fetchall()
            return [dict(r) for r in rows]

    def query_earthquakes(self, start: str = None, end: str = None,
                          min_mag: float = None, limit: int = 5000) -> list[dict]:
        clauses = []
        params = []

        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)
        if min_mag is not None:
            clauses.append("magnitude >= ?")
            params.append(min_mag)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM earthquakes {where} ORDER BY timestamp DESC LIMIT ?", params
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            sources = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM samples GROUP BY source ORDER BY cnt DESC"
            ).fetchall()
            metrics = conn.execute(
                "SELECT metric, COUNT(*) as cnt FROM samples GROUP BY metric ORDER BY cnt DESC"
            ).fetchall()
            eq_count = conn.execute("SELECT COUNT(*) FROM earthquakes").fetchone()[0]

            ts_range = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM samples"
            ).fetchone()

            recent_ingests = conn.execute(
                "SELECT source, fetched_at, sample_count, new_count, duration_ms, error "
                "FROM ingest_log ORDER BY fetched_at DESC LIMIT 20"
            ).fetchall()

            return {
                "total_samples": total,
                "earthquake_count": eq_count,
                "sources": {r[0]: r[1] for r in sources},
                "metrics": {r[0]: r[1] for r in metrics},
                "time_range": {"start": ts_range[0], "end": ts_range[1]} if ts_range[0] else None,
                "recent_ingests": [
                    {"source": r[0], "fetched_at": r[1], "samples": r[2],
                     "new": r[3], "duration_ms": r[4], "error": r[5]}
                    for r in recent_ingests
                ],
            }
