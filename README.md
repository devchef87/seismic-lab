# SeismicLab

Real-time seismic intelligence platform. Multi-source data fusion for earthquake monitoring, analysis, and experimental prediction.

**Live dashboard** with 24 seismic stations, DART buoy network, solar wind, geomagnetic indices, tidal stress, thermal anomalies, and volcanic activity — all in one place.

## Disclaimer

I'm not a seismologist, scientist, or domain expert of any kind — I'm a software engineer who got curious about earthquakes and fell down a rabbit hole. I broke my foot recently and have been laid up, so this became my boredom buster. The data correlations and models here are experimental and should not be used for any real-world safety decisions. If you actually know seismology, I'd love your input — that's the whole point of open sourcing this.

## What makes this different

Most seismic tools pull from one or two data sources. SeismicLab ingests from **27 sources** across 5 domains and correlates signals that are rarely studied together:

| Domain | Sources | Data Points |
|---|---|---|
| Seismic | FDSN/IRIS SeedLink (24 stations), USGS, EMSC | 6.9M+ station metrics |
| Oceanic | DART buoy network (40+ stations), NOAA tides | 11.4M+ pressure readings |
| Solar/Space | NOAA SWPC, GOES magnetometer, NASA DONKI, OMNI | 13M+ samples |
| Geomagnetic | Intermagnet, Kyoto Dst, GFZ Kp, cosmic rays | 1M+ samples |
| Geophysical | Tidal potential, GPS strain, FIRMS thermal, OLR | 2M+ samples |

**Total: 54M+ data points** with 5-year historical backfill.

## Quick Start

```bash
git clone https://github.com/devchef87/seismic-lab.git
cd seismic-lab

# Install dependencies
pip install -r requirements.txt

# Download the dataset from HuggingFace (~2GB compressed → 19GB local DB)
python scripts/download_data.py

# Start the server
python server.py
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
| `samples` | 36M | Time-series from all signal sources (solar wind, IMF, Kp, Dst, tidal, X-ray, etc.) |
| `dart_readings` | 11.4M | DART buoy seafloor pressure measurements |
| `station_metrics` | 6.9M | Per-station seismic amplitude, STA/LTA ratio, trigger state |
| `earthquakes` | 327K | USGS + EMSC earthquake catalog |
| `thermal_anomalies` | 601K | FIRMS thermal hotspots near volcanoes |
| `dart_stations` | 40+ | DART buoy metadata (position, depth, region) |
| `volcanoes` | 1,400+ | Smithsonian volcano registry |
| `eruptions` | 600+ | Historical eruption records |

## Dashboard

The HUD provides:

- **Real-time seismic map** — earthquake feed with magnitude/time filters, station waveforms
- **Workbench** — interactive charts for any signal (solar wind, Kp, tidal potential, GOES magnetometer, etc.) with auto-annotated anomaly badges
- **Zone analysis** — active seismic zone detection with multi-signal correlation
- **DART network** — seafloor pressure monitoring with mode transition detection
- **Volcanic activity** — thermal anomaly tracking near active volcanoes
- **AI forecasts** — experimental zone-level predictions (model in development)

## Research

The `lab/` directory contains reproducible experiments. These are exploratory — I'm learning as I go:

- **`solar_seismic_coupling.py`** — Statistical analysis of solar wind / geomagnetic correlations with M6.5+ seismicity. Finding: 2.3x increased likelihood during high solar wind periods (p=0.02).
- **`tidal_triggering.py`** — Tidal stress analysis on fault systems. Seafloor pressure trend reversals significant at p=0.02 before M6.5+ events.
- **`dart_case_study.py`** — DART buoy pressure precursor analysis (Venezuela M7.5 case study).
- **`train_stgnn.py`** — Spatio-Temporal Graph Neural Network for multi-zone prediction.
- **`train_zone_test.py`** — Zone-focused training with full feature set (seismic + solar + tidal + DART).
- **`deep_backfill.py`** — 5-year seismic waveform backfill from FDSN/IRIS.

## Architecture

```
server.py              FastAPI server + WebSocket live feed
seedlink.py            Real-time SeedLink waveform monitor (24 stations)
predict.py             XGBoost predictor + hotspot scanner
prediction_engine.py   ST-GNN prediction engine
threat.py              Zone threat detection
event_analyzer.py      Post-event analysis
alerts.py              Email notifications (optional)

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

The dashboard works without any API keys — most data sources (USGS, NOAA, SeedLink, Intermagnet, DART) are open and require no authentication.

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
