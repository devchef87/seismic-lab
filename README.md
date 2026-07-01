# SeismicLab

Real-time seismic intelligence platform. Multi-source data fusion for earthquake monitoring, analysis, and experimental prediction.

**Live dashboard** with 53 seismic stations (24 live-streamed via SeedLink), DART buoy network, solar wind, geomagnetic indices, tidal stress, thermal anomalies, and volcanic activity, all in one place.

## Disclaimer

I broke my foot recently and have been laid up, so this became my boredom buster. I'm not a seismologist, scientist, or domain expert of any kind, I'm a software engineer who got curious about earthquakes and fell down a rabbit hole. The data correlations and models here are experimental and should not be used for any real-world safety decisions. If you actually know seismology, I'd love your input, that's the whole point of open sourcing this.

## What makes this different

Most seismic tools pull from one or two data sources. SeismicLab ingests from **27 sources** across 5 domains and correlates signals that are rarely studied together:

| Domain | Sources | Data Points |
|---|---|---|
| Seismic | FDSN/IRIS (53 stations, 24 live via SeedLink), USGS, EMSC | 41M+ station metrics |
| Oceanic | DART buoy network (47 stations), NOAA tides | 11.4M+ pressure readings |
| Solar/Space | NOAA SWPC, GOES magnetometer, NASA DONKI, OMNI | 13M+ samples |
| Geomagnetic | Intermagnet, Kyoto Dst, GFZ Kp, cosmic rays | 1M+ samples |
| Geophysical | Tidal potential, GPS strain, FIRMS thermal, OLR | 2M+ samples |

**Total: ~108M data points** with an 11-year historical backfill (2015–present).

## Quick Start

```bash
git clone https://github.com/devchef87/seismic-lab.git
cd seismic-lab

# Install dependencies
pip install -r requirements.txt

# Download the dataset from HuggingFace (→ ~32GB local DB)
python scripts/download_data.py

# Start the server
python server.py

# (optional) start the realtime swarm-escalation watch services
./run_realtime.sh start
```

Open `http://localhost:8000` in your browser.

## Dataset

The full dataset is hosted on HuggingFace: [Groovedev/seismic-lab-data](https://huggingface.co/datasets/Groovedev/seismic-lab-data)

Available as compressed Parquet files, partitioned by year. Updated daily.

```bash
# Full download + DB rebuild
python scripts/download_data.py

# Incremental update
python scripts/download_data.py --since 2026-06-01
```

### Tables

| Table | Rows | Description |
|---|---|---|
| `samples` | 54M | Time-series from all signal sources (solar wind, IMF, Kp, Dst, tidal, X-ray, etc.) |
| `station_metrics` | 41M | Per-station seismic amplitude, STA/LTA ratio, trigger state (53 stations) |
| `dart_readings` | 11.4M | DART buoy seafloor pressure measurements |
| `earthquakes` | 640K | USGS + EMSC catalog, de-duplicated across agencies (2015–present, ~20K at M5+) |
| `thermal_anomalies` | 622K | FIRMS thermal hotspots near volcanoes |
| `gps_strain` (gps.db) | 168K | GPS crustal-deformation strain residuals |
| `eruptions` | 2,224 | Historical eruption records |
| `volcanoes` | 1,215 | Smithsonian volcano registry |
| `dart_stations` | 47 | DART buoy metadata (position, depth, region) |

## Dashboard

The HUD provides:

- **Real-time seismic map**, earthquake feed with magnitude/time filters, station waveforms
- **Workbench**, interactive charts for any signal (solar wind, Kp, tidal potential, GOES magnetometer, etc.) with auto-annotated anomaly badges
- **Zone analysis**, active seismic zone detection with multi-signal correlation
- **DART network**, seafloor pressure monitoring with mode transition detection
- **Volcanic activity**, thermal anomaly tracking near active volcanoes
- **Swarm-escalation watch**, the live forecasting panel (see below), calibrated probability that an active swarm escalates to a mainshock

## Event-Level Escalation Model (the primary forecasting model)

The original zone-level model tried to answer "will an M5+ happen here this hour?" across a grid. When declustered and tested on independent mainshocks, it scored near random (~0.54 AUC) — almost all the apparent skill was the trivial fact that aftershocks follow big quakes (Omori's law).

The breakthrough came from flipping the paradigm: instead of polling zones on a schedule, **score each earthquake on arrival**. When a new quake comes in, the system immediately evaluates the sequence context at that location — how many prior events, what magnitude trajectory, what pattern shape — and returns an escalation probability. Like a webhook, not a cron job.

> Given this earthquake and the sequence at its location, what's the probability
> a **M+1.0 larger event** follows within **7 days** at the same location
> (Gardner-Knopoff magnitude-scaled interaction radius)?

**Performance: 0.87 macro AUC** across 13 zones (up from 0.49 with the old 221-feature zone model). Every zone above 0.77. Uses only **25 catalog/sequence features** — no environmental data needed.

### How it works

The model discovered that **sequence shape is the entire signal**. The top 4 features account for 74% of importance:

| Feature | Importance | What it captures |
|---|---|---|
| `mag_range_7d` | 32.5% | Spread between smallest and largest event in 7-day window |
| `rumble_ratio` | 21.5% | Max magnitude / median — high means one event towers over the rest |
| `trigger_mag` | 11.0% | Magnitude of the current earthquake |
| `n_events_24h` | 9.2% | How active this location is right now |

### Sequence patterns

Analysis of 641K M2.5+ events (2015–present) revealed distinct trajectory shapes that dramatically lift escalation probability:

| Pattern | Description | Escalation rate |
|---|---|---|
| **Rumble** | 5+ events, one significantly larger than rest | 85.2% |
| **Double-tap** | Two similar-magnitude events, then a jump | 67.0% |
| **Staircase** | 3+ consecutive magnitude increases | 38.7% |
| **Accelerating** | Positive magnitude trend | 15–45% |
| **Active sequence** | 3+ events, no dominant shape | 10–35% |
| **Isolated** | No nearby prior activity | 2–15% |

Event count follows a logarithmic curve: 0→5 events gains +16pp (33%→49%), but 5→50 only adds +36pp more. The knee is at ~5 events. Pattern-conditioned lift adds +16–29pp over flat sequences at the same event count.

### What didn't work

Environmental features (DART seafloor pressure, solar wind, GOES magnetometer, tidal potential, Dst, Kp, IMF) were tested as a second-stage modulator on top of the sequence model. Result: **+0.0004 macro AUC** — zero lift. None of the 10 environmental features cracked the top 15. The sequence pattern captures everything the model needs.

### Alert thresholds

| Probability | UI treatment |
|---|---|
| < 0.30 | Don't show in escalation monitor (normal map dots) |
| 0.30 – 0.55 | **Watch** — yellow, show sequence context |
| 0.55 – 0.80 | **Elevated** — orange, prominent marker |
| > 0.80 | **Alert** — red, notification-worthy |

Events within 150km are clustered into one entry per active sequence. A typical day has ~400 scored events but only ~10–15 active sequences above 30%.

## Swarm-Escalation Watch (zone-level model)

The zone-level swarm model still runs in parallel, providing an area-level view: "these zones have active swarms." It's a LightGBM model over seismicity rate/acceleration, b-value, tidal stress, DART seafloor loading, GPS crustal deformation, and a volcanic prior. Pooled AUC ~0.66, with real within-zone skill in **Alaska, South America, New Zealand, Japan/Kurils** (~0.59–0.73 AUC) and **California** (~0.63 after deep backfill).

The event-level model (per-earthquake granularity) and zone-level model (swarm cluster view) complement each other — both run from the realtime engine.

Alert levels: **WATCH**, **ADVISORY**, **NORMAL** — surfaced only in the validated zones above.

### Running it

```bash
# starts 3 background services: live EMSC small-event poller,
# volcanic-alert ingest, and the realtime scoring engine
./run_realtime.sh start      # status | stop | restart

# they keep data/tier2_watch.json current; the dashboard reads that file
```

Detection is **cluster-centered**: instead of a fixed lat/lon grid (which missed swarms outside its zones and split faults at cell boundaries), the engine clusters recent small events *anywhere on Earth* and scores a region centered on each swarm's own centroid. So it auto-covers active regions the fixed zones didn't (Tonga, Vanuatu, the western Aleutians, etc.) with no edge-splitting. It caches slow signals (solar/geomag/tidal) hourly and re-detects on new activity, with a rising/falling trend per swarm. (`--mode grid` still offers the original 13 fixed zones.) Schema and UI contract are in `TIER2_WATCH_API.md`; deployment notes in `TIER2_DEPLOYMENT_REPORT.md`.

## Research

The `lab/` directory contains reproducible experiments. These are exploratory, I'm learning as I go:

- **`solar_seismic_coupling.py`**, Statistical analysis of solar wind / geomagnetic correlations with M6.5+ seismicity. Finding: 2.3x increased likelihood during high solar wind periods (p=0.02).
- **`tidal_triggering.py`**, Tidal stress analysis on fault systems. Seafloor pressure trend reversals significant at p=0.02 before M6.5+ events.
- **`dart_case_study.py`**, DART buoy pressure precursor analysis (Venezuela M7.5 case study).
- **`train_stgnn.py`**, Spatio-Temporal Graph Neural Network for multi-zone prediction.
- **`train_zone_test.py`**, Zone-focused training with full feature set (seismic + solar + tidal + DART).
- **`deep_backfill.py`**, 11-year (2015–present) seismic waveform backfill from FDSN/IRIS, across 53 stations (Alaska, Chile, Taiwan, Kamchatka, NZ, Caribbean + the GSN backbone).
- **`train_ensemble.py`**, the multi-signal feature pipeline + LightGBM trainer (occurrence and tier-2 escalation objectives) across 13 zones. Includes sequence trajectory features (rumble, staircase, double-tap).
- **`declustering_experiment.py`**, Gardner-Knopoff declustering test that showed the original occurrence model was largely scoring aftershocks, the finding that motivated the swarm-escalation reframe.
- **`tier2_escalation.py`**, the swarm-escalation discrimination experiment (catalog-only vs multi-signal, confirming the non-seismic signals add real skill).
- **`tier2_watch.py`**, trains + calibrates the deployed model and writes the alert-banded watch bundle.

Scripts (`scripts/`):
- **`train_event_model.py`**, the event-level escalation model trainer (0.87 AUC). 25 features, 641K events, 3-stage ablation (catalog-only, catalog+env, M4+-only). This is the primary model.
- **`sequence_analysis.py`**, full sequence chain analysis: parameter sweep for interaction radius, trajectory shapes (rumble, staircase, double-tap), event-count vs escalation curves.
- **`gated_ablation.py`**, 4-experiment ablation showing environmental features add zero lift when gated on active sequences.

## Architecture

```
server.py              FastAPI server + WebSocket live feed
seedlink.py            Real-time SeedLink waveform monitor (24 stations)
predict.py             XGBoost predictor + hotspot scanner
prediction_engine.py   ST-GNN prediction engine
threat.py              Zone threat detection
event_analyzer.py      Post-event analysis
alerts.py              Email notifications (optional)

run_realtime.sh        Launcher for the 3 swarm-escalation services
lab/event_scorer.py       Event-level escalation scorer (25 features, 0.87 AUC)
lab/realtime_engine.py    Event-driven scoring -> data/tier2_watch.json + event_scores.json
lab/ingest_emsc_live.py   Live EMSC small-event (foreshock) poller
lab/ingest_volcanic_alerts.py  USGS HANS volcanic alert ingest

ingest/
  engine.py            Polling engine (27 sources on configurable intervals)
  store.py             SQLite time-series store
  sources.py           Data source connectors
  features.py          Feature engineering
  backfill.py          Historical data backfill

dashboard/static/
  index.html           Single-page HUD
  seismiclab.js        Dashboard logic (~3000 lines)
  style.css            Dark theme UI
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `NASA_API_KEY` | No | NASA API for CME/flare data. Falls back to DEMO_KEY |
| `FIRMS_API_KEY` | No | NASA FIRMS for thermal anomaly data. Free registration |
| `MAPTILER_API_KEY` | No | Satellite basemap tiles. Free tier at [maptiler.com](https://cloud.maptiler.com/account/keys/) |
| `MAPBOX_ACCESS_TOKEN` | No | High-res satellite basemap tiles. Free tier at [mapbox.com](https://account.mapbox.com/access-tokens/) |

The dashboard works without any API keys, most data sources (USGS, NOAA, SeedLink, Intermagnet, DART) are open and require no authentication.

## License

MIT

## Contributing

This is a hobby project that grew into something I think others might find useful. I'm especially interested in contributions from people who actually know what they're doing in these areas:

- Seismology / geophysics domain knowledge
- New data source integrations
- Feature engineering for prediction models
- Model architectures beyond XGBoost and ST-GNN
- Dashboard UX improvements
- Statistical analysis of cross-domain correlations

Built by [Groove](https://groovedev.ai)
