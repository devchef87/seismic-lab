# Event-Level Escalation Model — Frontend Integration Brief

> **Update:** the `/api/event-scores` payload now also carries a regional
> **big-event (M6+) watch** — see `BIG_EVENT_WATCH_BRIEF.md`. Additive only;
> nothing in this document changed.

## What Changed

We built a fundamentally new model that scores **individual earthquakes on arrival** instead of polling zones hourly. When a new quake comes in, the system immediately evaluates the sequence context at that location and returns an escalation probability.

**Performance**: Macro AUC **0.87** (up from 0.49 with the zone-level model). Every zone above 0.77.

## New API Endpoint

### `GET /api/event-scores`

Returns the last 48h of scored earthquakes, **clustered by location** so each active sequence appears once (not once per event). Most dangerous first.

```json
{
  "generated": "2026-07-01T03:30:00+00:00",
  "model": "event_escalation_v1",
  "model_auc": 0.87,
  "total_scored": 417,
  "n_sequences": 127,
  "high_risk_count": 8,
  "sequences": [
    {
      "event_id": "emsc:20260701_0000042",
      "time": "2026-07-01T02:15:33+00:00",
      "magnitude": 3.8,
      "max_magnitude": 4.4,
      "depth_km": 12.4,
      "lat": -1.23,
      "lon": 120.57,
      "escalation_prob": 0.513,
      "sequence_pattern": "rumble",
      "magnitude_probs": {
        "5.0": 0.385,
        "5.5": 0.221,
        "6.0": 0.074,
        "6.5": 0.031,
        "7.0": 0.011
      },
      "n_events_in_cluster": 42,
      "latest_event_id": "emsc:20260701_0000055",
      "sequence_context": {
        "events_24h": 8,
        "events_7d": 42,
        "events_30d": 67,
        "max_mag_7d": 4.4,
        "rumble_ratio": 1.63,
        "consec_up": 5,
        "mag_trend_72h": 0.12,
        "hours_since_last": 1.3
      }
    }
  ],
  "events": [ ... ]
}
```

### Key Fields

| Field | Type | Description |
|---|---|---|
| `escalation_prob` | float 0-1 | Probability a larger event (trigger + M1.0) follows within 7 days at this location. **This is the primary signal.** |
| `sequence_pattern` | string | Classified pattern type (see below) |
| `magnitude_probs` | object | Per-threshold magnitude probabilities (see below) |
| `sequence_context` | object | Raw sequence features the model used — for display |

### Magnitude Probabilities (`magnitude_probs`)

Breaks down the escalation probability by specific magnitude thresholds. Each key is a magnitude (e.g. `"6.0"`), each value is the probability of reaching **at least** that magnitude within 7 days.

Computed as: `escalation_prob × P(follower ≥ Mx | escalation happened, for sequences at this magnitude level)`, using an empirical exceedance table built from 641K historical events.

Only includes thresholds **above the current sequence max** (no point showing P(≥M5.0) when M5.2 already happened). Omits negligible probabilities (< 0.1%).

**Suggested display** (in the event detail panel or on hover):
```
Escalation: 51%
  ≥M5.0   38%
  ≥M5.5   22%
  ≥M6.0    7.4%
  ≥M6.5    3.1%
  ≥M7.0    1.1%
```

### Sequence Pattern Types

| Pattern | What it means | Typical prob range |
|---|---|---|
| `isolated` | No nearby prior activity | 0.02 – 0.15 |
| `early_sequence` | 1-2 prior events, sequence just starting | 0.05 – 0.20 |
| `active_sequence` | 3+ events, no dominant shape yet | 0.10 – 0.35 |
| `staircase` | 3+ consecutive magnitude increases | 0.15 – 0.50 |
| `double_tap` | Two similar-mag events, then a jump | 0.20 – 0.55 |
| `accelerating` | Magnitudes trending upward | 0.15 – 0.45 |
| `rumble` | 5+ events, one significantly larger than rest | 0.30 – 0.70 |

### Important: Use `sequences`, not `events`

The `sequences` array is **deduplicated by location** — all events within 150km of each other are collapsed into one entry, keeping the highest-probability event as representative. Use this for the escalation monitor, NOT `events` (which has every individual quake).

A typical day has ~400 scored events but only ~10-15 active sequences with prob >= 30%.

### Filtering for the Escalation Monitor

**Don't show `isolated` pattern events in the alert list.** Filter to:
```js
sequences.filter(s => s.sequence_pattern !== 'isolated' && s.escalation_prob >= 0.30)
```

This typically yields 5-15 entries — the genuine active sequences worth watching.

### Suggested Alert Thresholds

> **2026-07-21: probabilities recalibrated — thresholds changed.**
> `escalation_prob` is now a calibrated empirical frequency (0.30 means ~30%
> of identical historical situations escalated — verified against live
> outcomes). Raw scores used to run 2–4× hot, so values dropped across the
> board and the old 0.30/0.55/0.80 bands would show almost nothing. Use:

| Prob (calibrated) | Suggested UI treatment |
|---|---|
| < 0.10 | Don't show in escalation monitor (still show on map as normal dots) |
| 0.10 – 0.30 | **Watch** — yellow highlight, show sequence context |
| 0.30 – 0.50 | **Elevated** — orange, prominent marker, show event chain on click |
| > 0.50 | **Alert** — red, bold display, notification-worthy (realized ~60%+ historically) |

## Relationship to Existing tier2_watch

The **existing** `/api/tier2/watch` (zone-level swarm model) **stays as-is** — it provides the swarm cluster view with escalation zones.

The **new** `/api/event-scores` complements it with per-event granularity. Think of it as:
- **tier2_watch** = "these zones have active swarms" (area view)
- **event-scores** = "this specific quake has X% chance of being a foreshock" (point view)

Both run in parallel from the realtime engine.

## Suggested UI Integration

### Map Layer
- Each scored event gets a marker on the map
- Marker size/color keyed to `escalation_prob`:
  - < 0.15: standard earthquake dot (existing)
  - 0.15+: pulsing ring, color ramps from yellow → orange → red
  - `rumble` and `double_tap` patterns get a distinctive icon or badge

### Event Detail Panel (on click)
- **Escalation probability** as a prominent gauge/bar (0-100%)
- **Sequence pattern** as a badge/label with tooltip explaining what it means
- **Sequence context** as a compact summary:
  - "42 events in 7 days, 8 in last 24h"
  - "Magnitude trend: ↑ accelerating"
  - "Rumble ratio: 1.63 (many small → spike)"
  - "Last event: 1.3 hours ago"
- **Mini timeline**: small sparkline of the last N events in the sequence (mag over time)

### Escalation Monitor (the sidebar list)
- Use `sequences` array, filtered: `sequence_pattern !== 'isolated' && escalation_prob >= 0.30`
- Sorted by probability (already sorted in response)
- Each entry shows:
  - Probability as percentage
  - Max magnitude in sequence (`max_magnitude`)
  - Location (lat/lon or place name)
  - Pattern badge (`rumble`, `staircase`, `double_tap`, etc.)
  - Event count: `n_events_in_cluster` (e.g. "42 events")
- Auto-updates on new scores (poll `/api/event-scores` every 60s)

## Polling / Refresh

The engine scores new events on each tick (~60s). Poll `/api/event-scores` at the same cadence as `/api/tier2/watch`. The response is lightweight (max 200 events, ~50KB).

## What the Model Does NOT Do

- Does **not** predict when/where an earthquake will happen from nothing
- Does **not** replace the swarm-level view — complements it
- Does **not** use environmental data (solar, tidal, etc.) — purely catalog sequence features
- `escalation_prob` is for a **M+1.0 larger event** within **7 days** at the **same location** (Gardner-Knopoff magnitude-scaled radius, ~10-100km depending on trigger size)
- `magnitude_probs` breaks this down by specific thresholds (M5+, M6+, M7+, etc.) using historical exceedance rates — these are empirical, not model predictions
