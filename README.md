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

## Swarm-Escalation Watch (the forecasting model)

The forecasting panel doesn't try to predict arbitrary earthquakes, that turned out to be mostly an illusion. When I declustered the catalog and tested the original "will an M5+ happen in this zone" model on *independent* mainshocks, it scored near random (~0.54 AUC). Almost all the apparent skill was the trivial fact that aftershocks follow big quakes (Omori's law). So the model was rebuilt around the one question that's both honest and useful:

> A seismic **swarm is active** here right now. What's the calibrated probability it
> **escalates to an independent M5+ mainshock within 72h**, versus fizzling out (which ~94% of swarms do)?

It's **declustered** (independent mainshocks only, no aftershock inflation), **calibrated** (when it says 30%, ~30% actually escalate, verified on held-out data), and **multi-signal**, a LightGBM model over seismicity rate/acceleration, b-value, tidal stress, DART seafloor loading, GPS crustal deformation, and a volcanic prior. On a future-holdout test the pooled discrimination is ~0.66 AUC, but the honest *within-zone* number — can it rank which swarm in a given region escalates? — is only ~0.52, near chance globally. Real within-zone skill exists in a handful of well-sampled zones (**Alaska, South America, New Zealand, Japan/Kurils**, ~0.59–0.73 AUC, plus **California** ~0.63 after a deep catalog backfill), so the live watch only raises alerts there and shows swarms elsewhere as informational. It's genuinely hard and far from solved — but it's an honest, calibrated number on the right problem, scoped to where it actually works.

Alert levels: **WATCH**, **ADVISORY**, **NORMAL** — surfaced only in the validated zones above. (A WARNING tier was dropped: in holdout its highest-confidence calls escalated 0% of the time — all false alarms — so it's capped to WATCH.) The panel is a live watchlist, often short or empty (empty = nothing building, not "all clear").

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
- **`train_ensemble.py`**, the multi-signal feature pipeline + LightGBM trainer (occurrence and tier-2 escalation objectives) across 13 zones.
- **`declustering_experiment.py`**, Gardner-Knopoff declustering test that showed the original occurrence model was largely scoring aftershocks, the finding that motivated the swarm-escalation reframe.
- **`tier2_escalation.py`**, the swarm-escalation discrimination experiment (catalog-only vs multi-signal, confirming the non-seismic signals add real skill).
- **`tier2_watch.py`**, trains + calibrates the deployed model and writes the alert-banded watch bundle.

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
lab/realtime_engine.py    Event-driven tier-2 scoring -> data/tier2_watch.json
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
