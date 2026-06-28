"""SeismicLab Seismic Monitor — real-time waveform monitoring via FDSN with STA/LTA detection.

Polls waveforms from IRIS/EarthScope FDSN for key stations near monitored
seismic zones, runs STA/LTA trigger detection, and estimates magnitude from
amplitude. Detected events are injected into the SeismicLab store within
~30-60 seconds of ground motion.

Station metrics are persisted to SQLite for model training. Raw ~5 Hz samples
are kept in memory (1 hour) for live seismograph display; 30-second aggregates
(min, max, mean amplitude + STA/LTA ratio) are written to the station_metrics
table for long-term analysis and forecasting features.
"""

import logging
import os
import pickle
import sqlite3
import threading
import time
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import deque

log = logging.getLogger("seismiclab.seedlink")
UTC = timezone.utc

STATIONS = [
    # North America
    {"net": "IU", "sta": "COLA", "loc": "00", "cha": "BHZ", "lat": 64.87, "lon": -147.86, "name": "College, Alaska"},
    {"net": "IU", "sta": "COR",  "loc": "00", "cha": "BHZ", "lat": 44.59, "lon": -123.30, "name": "Corvallis, Oregon"},
    {"net": "IU", "sta": "TUC",  "loc": "00", "cha": "BHZ", "lat": 32.31, "lon": -110.78, "name": "Tucson, Arizona"},
    {"net": "IU", "sta": "ANMO", "loc": "00", "cha": "BHZ", "lat": 34.95, "lon": -106.46, "name": "Albuquerque, NM"},
    {"net": "IU", "sta": "TEIG", "loc": "00", "cha": "BHZ", "lat": 20.23, "lon": -88.28, "name": "Teoloyucan, Mexico"},
    # Caribbean / Central America
    {"net": "IU", "sta": "SJG",  "loc": "00", "cha": "BHZ", "lat": 18.11, "lon": -66.15, "name": "San Juan, Puerto Rico"},
    {"net": "II", "sta": "JTS",  "loc": "00", "cha": "BHZ", "lat": 10.29, "lon": -84.95, "name": "Las Juntas, Costa Rica"},
    # South America
    {"net": "II", "sta": "NNA",  "loc": "00", "cha": "BHZ", "lat": -11.99, "lon": -76.84, "name": "Nana, Peru"},
    {"net": "IU", "sta": "LCO",  "loc": "00", "cha": "BHZ", "lat": -29.01, "lon": -70.70, "name": "Las Campanas, Chile"},
    # Europe
    {"net": "II", "sta": "ESK",  "loc": "00", "cha": "BHZ", "lat": 55.32, "lon": -3.21, "name": "Eskdalemuir, Scotland"},
    {"net": "IU", "sta": "ANTO", "loc": "00", "cha": "BHZ", "lat": 39.87, "lon": 30.50, "name": "Ankara, Turkey"},
    # Africa / Middle East
    {"net": "IU", "sta": "GNI",  "loc": "00", "cha": "BHZ", "lat": 40.15, "lon": 44.74, "name": "Garni, Armenia"},
    {"net": "II", "sta": "MBAR", "loc": "00", "cha": "BHZ", "lat": -0.60, "lon": 30.74, "name": "Mbarara, Uganda"},
    # Central / South Asia
    {"net": "II", "sta": "AAK",  "loc": "00", "cha": "BHZ", "lat": 42.64, "lon": 74.49, "name": "Ala Archa, Kyrgyzstan"},
    {"net": "II", "sta": "DGAR", "loc": "00", "cha": "BHZ", "lat": -7.41, "lon": 72.45, "name": "Diego Garcia"},
    # East Asia
    {"net": "IU", "sta": "ULN",  "loc": "00", "cha": "BHZ", "lat": 47.87, "lon": 107.05, "name": "Ulaanbaatar, Mongolia"},
    {"net": "IU", "sta": "INCN", "loc": "00", "cha": "BHZ", "lat": 37.48, "lon": 126.62, "name": "Incheon, Korea"},
    {"net": "IU", "sta": "MAJO", "loc": "00", "cha": "BHZ", "lat": 36.54, "lon": 138.20, "name": "Matsushiro, Japan"},
    {"net": "II", "sta": "ERM",  "loc": "00", "cha": "BHZ", "lat": 42.02, "lon": 143.16, "name": "Erimo, Japan"},
    # Southeast Asia / Pacific
    {"net": "IU", "sta": "DAV",  "loc": "00", "cha": "BHZ", "lat": 7.07, "lon": 125.58, "name": "Davao, Philippines"},
    {"net": "II", "sta": "KAPI", "loc": "00", "cha": "BHZ", "lat": -5.01, "lon": 119.75, "name": "Kappang, Indonesia"},
    {"net": "IU", "sta": "GUMO", "loc": "00", "cha": "BHZ", "lat": 13.59, "lon": 144.87, "name": "Guam"},
    # Oceania
    {"net": "IU", "sta": "CTAO", "loc": "00", "cha": "BHZ", "lat": -20.09, "lon": 146.25, "name": "Charters Towers, Australia"},
    {"net": "II", "sta": "TAU",  "loc": "00", "cha": "BHZ", "lat": -42.91, "lon": 147.32, "name": "Hobart, Tasmania"},
]

POLL_INTERVAL = 30
STA_SEC = 2.0
LTA_SEC = 50.0
TRIGGER_RATIO = 3.5
COOLDOWN_SEC = 120

BACKFILL_HOURS = 72


def _init_station_db(db_path):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS station_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            station TEXT NOT NULL,
            amp_min REAL NOT NULL,
            amp_max REAL NOT NULL,
            amp_mean REAL NOT NULL,
            sta_lta_ratio REAL NOT NULL,
            triggered INTEGER NOT NULL DEFAULT 0,
            sample_rate REAL,
            UNIQUE(station, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_sm_station_ts
            ON station_metrics(station, timestamp);
        CREATE INDEX IF NOT EXISTS idx_sm_ts
            ON station_metrics(timestamp);
    """)
    conn.commit()
    conn.close()


class SeedLinkMonitor:

    def __init__(self, store):
        self.store = store
        self._db_path = store.db_path
        self._running = False
        self._thread = None
        self._backfill_thread = None
        self._detections = deque(maxlen=200)
        self._station_status = {}
        self._last_trigger = {}
        self._waveforms = {}
        self._client = None
        self._subscribers = {}
        self._sub_lock = threading.Lock()
        self._cache_path = os.path.join(os.path.dirname(str(self._db_path)), "waveform_cache.pkl")
        self._persist_counter = 0
        _init_station_db(self._db_path)
        self._load_waveform_cache()

    def subscribe(self, station_key):
        """Subscribe to live samples for a station. Returns an asyncio.Queue."""
        import asyncio
        q = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            if station_key not in self._subscribers:
                self._subscribers[station_key] = set()
            self._subscribers[station_key].add(q)
        return q

    def unsubscribe(self, station_key, q):
        with self._sub_lock:
            if station_key in self._subscribers:
                self._subscribers[station_key].discard(q)

    def _notify_subscribers(self, station_key, new_samples, sample_rate):
        """Push new samples to all subscribers for a station."""
        with self._sub_lock:
            subs = self._subscribers.get(station_key, set()).copy()
        if not subs:
            return
        msg = {"samples": new_samples, "sr": sample_rate, "ts": datetime.now(UTC).isoformat()}
        dead = []
        for q in subs:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        if dead:
            with self._sub_lock:
                for q in dead:
                    self._subscribers.get(station_key, set()).discard(q)

    def _load_waveform_cache(self):
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, 'rb') as f:
                cached = pickle.load(f)
            now = datetime.now(UTC)
            loaded = 0
            for key, wf in cached.items():
                ts = datetime.fromisoformat(wf["timestamp"])
                age = (now - ts).total_seconds()
                if age > 3600:
                    continue
                sr = wf["sample_rate"]
                keep_secs = max(0, 3600 - age)
                keep_n = int(keep_secs * sr)
                if keep_n <= 0:
                    continue
                wf["samples"] = wf["samples"][-keep_n:]
                self._waveforms[key] = wf
                loaded += 1
            if loaded:
                log.info(f"Loaded waveform cache: {loaded} stations, "
                         f"oldest data {age:.0f}s ago")
        except Exception as e:
            log.debug(f"Could not load waveform cache: {e}")

    def _persist_waveform_cache(self):
        try:
            cache = {}
            for key, wf in self._waveforms.items():
                cache[key] = {
                    "sample_rate": wf["sample_rate"],
                    "samples": wf["samples"][-int(3600 * wf["sample_rate"]):],
                    "timestamp": wf["timestamp"],
                }
            tmp = self._cache_path + ".tmp"
            with open(tmp, 'wb') as f:
                pickle.dump(cache, f, protocol=4)
            os.replace(tmp, self._cache_path)
        except Exception as e:
            log.debug(f"Could not persist waveform cache: {e}")

    def _compute_global_resonance(self):
        """Compute Global Seismic Resonance from stacked station FFTs.

        Uses the in-memory waveform buffers (~5 Hz, up to 1h per station).
        Takes the last 10 minutes of data, bandpass-filters to the Earth
        free-oscillation band (2-20 mHz), computes power spectra, and stacks
        across all stations to extract a planetary resonance signal.
        """
        from scipy.signal import butter, sosfilt

        WINDOW_SEC = 600
        F_LOW = 0.002   # 2 mHz
        F_HIGH = 0.020  # 20 mHz

        spectra = []
        freqs_out = None

        for key, wf in self._waveforms.items():
            sr = wf["sample_rate"]
            if sr < 1:
                continue
            n_samples = int(WINDOW_SEC * sr)
            samples = wf["samples"]
            if len(samples) < n_samples // 2:
                continue
            data = np.array(samples[-n_samples:], dtype=np.float64)

            data -= np.mean(data)
            data *= np.hanning(len(data))

            nyq = sr / 2
            if F_HIGH >= nyq:
                continue
            try:
                sos = butter(4, [F_LOW / nyq, F_HIGH / nyq], btype='band', output='sos')
                filtered = sosfilt(sos, data)
            except Exception:
                continue

            n = len(filtered)
            fft_vals = np.fft.rfft(filtered)
            power = (np.abs(fft_vals) ** 2) / n
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)

            mask = (freqs >= F_LOW) & (freqs <= F_HIGH)
            if not np.any(mask):
                continue

            spectra.append(power[mask])
            if freqs_out is None:
                freqs_out = freqs[mask]

        if len(spectra) < 3 or freqs_out is None:
            return

        min_len = min(len(s) for s in spectra)
        spectra = [s[:min_len] for s in spectra]
        freqs_out = freqs_out[:min_len]
        stacked = np.mean(spectra, axis=0)

        peak_idx = np.argmax(stacked)
        peak_freq_mhz = freqs_out[peak_idx] * 1000
        peak_power = float(stacked[peak_idx])
        total_power = float(np.sum(stacked))
        spectral_centroid_mhz = float(np.sum(freqs_out * stacked) / np.sum(stacked) * 1000) if total_power > 0 else 0

        log_power = float(np.log10(total_power + 1e-30))

        ts_iso = datetime.now(UTC).replace(microsecond=0).isoformat()

        try:
            from ingest.sources import Sample
            samples = [
                Sample(source="seismic_resonance", metric="peak_freq",
                       timestamp=ts_iso, value=round(peak_freq_mhz, 3), unit="mHz"),
                Sample(source="seismic_resonance", metric="spectral_power",
                       timestamp=ts_iso, value=round(log_power, 4), unit="dB"),
                Sample(source="seismic_resonance", metric="spectral_centroid",
                       timestamp=ts_iso, value=round(spectral_centroid_mhz, 3), unit="mHz"),
                Sample(source="seismic_resonance", metric="station_count",
                       timestamp=ts_iso, value=len(spectra), unit=""),
            ]
            self.store.ingest(samples, source_name="seismic_resonance")
        except Exception as e:
            log.debug(f"Resonance ingest failed: {e}")

    def get_station_resonance(self, station_key):
        """Return the waveform bandpass-filtered to the free-oscillation band.

        Same format as live waveform data so the client can render it
        with the same seismograph drawing code.
        """
        from scipy.signal import butter, sosfilt

        wf = self._waveforms.get(station_key)
        if not wf or len(wf["samples"]) < 200:
            return None

        sr = wf["sample_rate"]
        if sr < 1:
            return None

        F_LOW = 0.002
        F_HIGH = 0.050
        nyq = sr / 2
        if F_HIGH >= nyq:
            F_HIGH = nyq * 0.9

        data = np.array(wf["samples"], dtype=np.float64)
        data -= np.mean(data)

        try:
            sos = butter(4, [F_LOW / nyq, F_HIGH / nyq], btype='band', output='sos')
            filtered = sosfilt(sos, data)
        except Exception:
            return None

        return {
            "station": station_key,
            "samples": filtered.tolist(),
            "sample_rate": sr,
            "timestamp": wf["timestamp"],
        }

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info(f"Seismic monitor started — {len(STATIONS)} stations via FDSN")
        self._backfill_thread = threading.Thread(target=self._backfill, daemon=True)
        self._backfill_thread.start()

    def stop(self):
        self._running = False

    def get_recent_detections(self, limit=20):
        return list(self._detections)[-limit:]

    def get_station_status(self):
        result = []
        for stn in STATIONS:
            key = f"{stn['net']}.{stn['sta']}"
            st = self._station_status.get(key, {})
            result.append({
                "station": key,
                "name": stn["name"],
                "lat": stn["lat"],
                "lon": stn["lon"],
                "sta_lta_ratio": st.get("ratio", 0),
                "peak_amplitude": st.get("peak", 0),
                "triggered": st.get("triggered", False),
                "connected": st.get("connected", False),
                "last_update": st.get("last_update", ""),
            })
        return result

    def get_waveform(self, station_key, scale="6h"):
        """Return waveform data for a station.

        Returns DB envelope for 6h/12h/24h scales.
        Live mode uses WebSocket instead of this method.
        """
        return self._get_envelope_from_db(station_key, scale)

    def _get_envelope_from_db(self, station_key, scale):
        hours = {"30m": 0.5, "1h": 1, "6h": 6, "12h": 12, "24h": 24}.get(scale, 0.5)
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            rows = conn.execute(
                "SELECT amp_min, amp_max FROM station_metrics "
                "WHERE station = ? AND timestamp >= ? ORDER BY timestamp ASC",
                (station_key, cutoff)
            ).fetchall()
            conn.close()
        except Exception:
            rows = []

        envelope = [{"mn": r[0], "mx": r[1]} for r in rows]
        return {
            "station": station_key,
            "scale": scale,
            "mode": "envelope",
            "envelope": envelope,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _get_client(self):
        if self._client is None:
            from obspy.clients.fdsn import Client
            import warnings
            warnings.filterwarnings("ignore", message=".*IRIS.*EarthScope.*")
            self._client = Client("IRIS")
        return self._client

    def _run_loop(self):
        try:
            from obspy import UTCDateTime
            self._get_client()
        except ImportError:
            log.error("obspy not installed — seismic monitor disabled")
            self._running = False
            return
        except Exception as e:
            log.error(f"FDSN client init failed: {e}")
            self._running = False
            return

        log.info("FDSN client connected to IRIS/EarthScope")

        while self._running:
            try:
                self._poll_all_stations()
            except Exception as e:
                log.error(f"Poll cycle error: {e}")
            time.sleep(POLL_INTERVAL)

    def _store_metric(self, station_key, ts_iso, amp_min, amp_max, amp_mean,
                      sta_lta_ratio, triggered, sample_rate):
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.execute(
                "INSERT OR IGNORE INTO station_metrics "
                "(timestamp, station, amp_min, amp_max, amp_mean, sta_lta_ratio, triggered, sample_rate) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts_iso, station_key,
                 round(amp_min, 1), round(amp_max, 1), round(amp_mean, 2),
                 round(sta_lta_ratio, 3), int(triggered), round(sample_rate, 2))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"Failed to store metric for {station_key}: {e}")

    def _poll_all_stations(self):
        from obspy import UTCDateTime

        client = self._get_client()
        now = UTCDateTime()

        for stn in STATIONS:
            key = f"{stn['net']}.{stn['sta']}"
            try:
                st = client.get_waveforms(
                    stn["net"], stn["sta"], stn["loc"], stn["cha"],
                    now - 65, now - 5,
                    attach_response=False,
                )
                if not st or len(st) == 0:
                    self._station_status[key] = {"connected": False}
                    continue

                tr = st[0]
                raw = tr.data.astype(float)
                sr = tr.stats.sampling_rate

                # Store downsampled waveform (~5 Hz) in memory for live seismograph
                step = max(1, int(sr / 5))
                ds = raw[::step]
                buf = self._waveforms.get(key)
                if buf is None:
                    buf = {"sample_rate": round(sr / step, 2), "samples": [],
                           "timestamp": ""}
                    self._waveforms[key] = buf
                new_samples = ds.tolist()
                buf["samples"].extend(new_samples)
                max_samples = int(3600 * buf["sample_rate"])
                if len(buf["samples"]) > max_samples:
                    buf["samples"] = buf["samples"][-max_samples:]
                buf["timestamp"] = datetime.now(UTC).isoformat()
                self._notify_subscribers(key, new_samples, buf["sample_rate"])

                data = np.abs(raw)

                sta_n = max(1, int(STA_SEC * sr))
                lta_n = max(1, int(LTA_SEC * sr))

                if len(data) < lta_n:
                    lta_n = max(sta_n * 5, len(data) // 2)
                    if len(data) < lta_n:
                        self._station_status[key] = {"connected": True, "ratio": 0, "peak": 0, "triggered": False,
                                                      "last_update": datetime.now(UTC).isoformat()}
                        continue

                sta_arr = np.convolve(data, np.ones(sta_n) / sta_n, mode='valid')
                lta_arr = np.convolve(data, np.ones(lta_n) / lta_n, mode='valid')

                min_len = min(len(sta_arr), len(lta_arr))
                sta_arr = sta_arr[-min_len:]
                lta_arr = lta_arr[-min_len:]

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(lta_arr > 0, sta_arr / lta_arr, 0)

                max_ratio = float(np.max(ratio)) if len(ratio) > 0 else 0
                peak = float(np.max(data))
                triggered = max_ratio > TRIGGER_RATIO

                self._station_status[key] = {
                    "connected": True,
                    "ratio": round(max_ratio, 2),
                    "peak": round(peak, 1),
                    "triggered": triggered,
                    "last_update": datetime.now(UTC).isoformat(),
                }

                # Persist to database
                ts_iso = datetime.now(UTC).replace(microsecond=0).isoformat()
                self._store_metric(
                    key, ts_iso,
                    amp_min=float(raw.min()),
                    amp_max=float(raw.max()),
                    amp_mean=float(np.mean(data)),
                    sta_lta_ratio=max_ratio,
                    triggered=triggered,
                    sample_rate=sr,
                )

                if triggered:
                    last_t = self._last_trigger.get(key, 0)
                    now_ts = time.time()
                    if now_ts - last_t > COOLDOWN_SEC:
                        self._last_trigger[key] = now_ts
                        est_mag = self._estimate_magnitude(peak)
                        self._on_detection(stn, max_ratio, peak, est_mag)

            except Exception as e:
                self._station_status[key] = {"connected": False}
                if "no data" not in str(e).lower() and "204" not in str(e):
                    log.debug(f"Station {key} poll failed: {e}")

        self._persist_counter += 1
        if self._persist_counter % 2 == 0:
            self._persist_waveform_cache()
        if self._persist_counter % 4 == 0:
            try:
                self._compute_global_resonance()
            except Exception as e:
                log.debug(f"Resonance computation error: {e}")

    def _estimate_magnitude(self, amplitude):
        if amplitude <= 0:
            return 0.0
        log_amp = math.log10(max(1, amplitude))
        est = log_amp * 1.1 - 1.5
        return max(0.5, min(8.5, round(est, 1)))

    def _on_detection(self, stn_info, ratio, peak, est_mag):
        ts = datetime.now(UTC)
        key = f"{stn_info['net']}.{stn_info['sta']}"

        detection = {
            "timestamp": ts.isoformat(),
            "station": key,
            "station_name": stn_info["name"],
            "lat": stn_info["lat"],
            "lon": stn_info["lon"],
            "estimated_magnitude": est_mag,
            "sta_lta_ratio": round(ratio, 1),
            "peak_amplitude": round(peak, 1),
            "source": "fdsn_detection",
        }
        self._detections.append(detection)

        log.info(f"DETECTION {key} ~M{est_mag:.1f} STA/LTA={ratio:.1f} "
                 f"peak={peak:.0f} ({stn_info['name']})")

        if est_mag >= 3.5:
            from ingest.sources import Sample
            sample = Sample(
                source="seedlink_detection", metric="magnitude",
                timestamp=ts.isoformat(), value=est_mag, unit="Mw_est",
                lat=stn_info["lat"], lon=stn_info["lon"],
                meta={
                    "station": key,
                    "place": f"Near {stn_info['name']} (waveform detection)",
                    "peak_amplitude": peak,
                    "sta_lta_ratio": ratio,
                    "type": "waveform_detection",
                    "id": f"sl_{stn_info['sta']}_{int(ts.timestamp())}",
                },
            )
            try:
                self.store.ingest([sample], source_name="seedlink_detection")
            except Exception as e:
                log.error(f"Failed to store detection: {e}")

    # ── Historical backfill ──────────────────────────────────

    def _backfill(self):
        """Backfill station_metrics with historical FDSN data on startup."""
        time.sleep(5)
        try:
            from obspy import UTCDateTime
            client = self._get_client()
        except Exception as e:
            log.error(f"Backfill: client init failed: {e}")
            return

        conn = sqlite3.connect(str(self._db_path), timeout=30)

        for stn in STATIONS:
            if not self._running:
                break
            key = f"{stn['net']}.{stn['sta']}"

            # Check how far back we already have data
            row = conn.execute(
                "SELECT MIN(timestamp) FROM station_metrics WHERE station = ?",
                (key,)
            ).fetchone()
            existing_earliest = row[0] if row and row[0] else None

            now = UTCDateTime()
            target_start = now - (BACKFILL_HOURS * 3600)

            if existing_earliest:
                existing_dt = datetime.fromisoformat(existing_earliest)
                if existing_dt.tzinfo is None:
                    existing_dt = existing_dt.replace(tzinfo=UTC)
                target_end_dt = existing_dt
                target_end = UTCDateTime(target_end_dt)
                if target_end <= target_start:
                    log.info(f"Backfill {key}: already have {BACKFILL_HOURS}h of data")
                    continue
                hours_needed = (target_end - target_start) / 3600
            else:
                target_end = now - 120
                hours_needed = BACKFILL_HOURS

            log.info(f"Backfill {key}: fetching {hours_needed:.0f}h of history...")

            rows_inserted = 0
            chunk_hours = 1
            t = target_start

            while t < target_end and self._running:
                t_end = min(t + chunk_hours * 3600, target_end)
                try:
                    st = client.get_waveforms(
                        stn["net"], stn["sta"], stn["loc"], stn["cha"],
                        t, t_end, attach_response=False,
                    )
                    if st and len(st) > 0:
                        rows_inserted += self._process_backfill_chunk(
                            conn, key, st[0])
                except Exception:
                    pass
                t = t_end
                time.sleep(0.3)

            log.info(f"Backfill {key}: {rows_inserted} rows inserted")

        conn.close()
        log.info("Backfill complete for all stations")

    def _process_backfill_chunk(self, conn, station_key, trace):
        """Split a trace into 30-second windows and insert metrics."""
        raw = trace.data.astype(float)
        sr = trace.stats.sampling_rate
        start_time = trace.stats.starttime
        window_samples = int(30 * sr)

        if len(raw) < window_samples:
            return 0

        rows = []
        for i in range(0, len(raw) - window_samples + 1, window_samples):
            chunk = raw[i:i + window_samples]
            abs_chunk = np.abs(chunk)

            # STA/LTA for this window
            sta_n = max(1, int(STA_SEC * sr))
            lta_n = max(1, int(LTA_SEC * sr))
            if len(abs_chunk) >= lta_n:
                sta_arr = np.convolve(abs_chunk, np.ones(sta_n) / sta_n, mode='valid')
                lta_arr = np.convolve(abs_chunk, np.ones(lta_n) / lta_n, mode='valid')
                min_len = min(len(sta_arr), len(lta_arr))
                sta_arr = sta_arr[-min_len:]
                lta_arr = lta_arr[-min_len:]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_arr = np.where(lta_arr > 0, sta_arr / lta_arr, 0)
                max_ratio = float(np.max(ratio_arr))
            else:
                max_ratio = 0.0

            window_time = start_time + (i / sr)
            ts_dt = datetime(
                window_time.year, window_time.month, window_time.day,
                window_time.hour, window_time.minute, window_time.second,
                tzinfo=UTC
            )
            ts_iso = ts_dt.isoformat()

            rows.append((
                ts_iso, station_key,
                round(float(chunk.min()), 1),
                round(float(chunk.max()), 1),
                round(float(np.mean(abs_chunk)), 2),
                round(max_ratio, 3),
                int(max_ratio > TRIGGER_RATIO),
                round(sr, 2),
            ))

        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO station_metrics "
                "(timestamp, station, amp_min, amp_max, amp_mean, sta_lta_ratio, triggered, sample_rate) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows
            )
            conn.commit()

        return len(rows)
