# Big-Event Watch — Frontend Integration Brief

## What's New

A second model now runs alongside the escalation scorer. It answers a different question:

- **Escalation monitor** (existing): "will *this sequence* produce an event M+1.0 bigger?"
- **Big-event watch** (new): "will an **M6+** occur within **100km** of here in the next **30 days**?"

Both are served from the same endpoint — `GET /api/event-scores`. **No breaking changes**: all existing fields are untouched; everything new is additive. The current UI keeps working with zero changes.

## Honest model performance (read this)

Trained on 640K events, temporal holdout. For regions with **no recent M5+** (the hard case — no obvious hint), the top-ranked alerts precede an actual M6+ about **1 time in 7** (vs 1.4% base rate — an 11× lift). AUC 0.71.

This is **screening, not prediction**. The UI copy must never say "an M6 is predicted." Correct framing: *"conditions here statistically resemble those that preceded M6+ events."* Most watch entries will NOT be followed by a big event.

## New Payload Fields

### 1. Top-level `big_event_watch` array

The headline feature. Usually **0–4 entries globally**; often empty. Empty = normal, not "no data."

```json
"big_event_watch": [
  {
    "cluster_id": "emsc:20260701_0000042",
    "lat": 40.4, "lon": 142.3,
    "time": "2026-07-01T18:22:10+00:00",
    "m6_prob": 0.36,
    "m55_prob": 0.52,
    "level": "watch",
    "first_event": false,
    "regional_context": {
      "events_100km_30d": 47,
      "max_mag_100km_30d": 5.0,
      "quiescence": -0.31,
      "b_value_30d": 0.84,
      "hist_m6_frac": 0.021
    },
    "n_events_in_cluster": 12,
    "max_magnitude": 5.0
  }
]
```

| Field | Meaning |
|---|---|
| `m6_prob` | P(M6+ within 100km / 30 days). **The primary number.** |
| `m55_prob` | Same for M5.5+ (always ≥ `m6_prob`) |
| `level` | `"watch"` (p ≥ 0.30) or `"elevated"` (p ≥ 0.55) |
| `first_event` | `true` = no M5+ here in 30 days — the model sees something in the *small* events. Rarer and more noteworthy than `false` (which often means a big aftershock zone). |
| `cluster_id` | Matches an `event_id` in the `sequences` array — join for place name, pattern, etc. |
| `regional_context.quiescence` | Positive = activity built up then went quiet (a known precursor pattern — worth surfacing in a tooltip) |
| `regional_context.b_value_30d` | Low (< ~0.8) = the local magnitude mix is skewed large — stress indicator |
| `regional_context.hist_m6_frac` | How productive this region historically is at M6+ |

### 2. Per-event/sequence additions (in `events` and via `sequences`)

Each scored event now also carries:

```json
"big_event_probs": {"5.5": 0.18, "6.0": 0.09},
"big_event_first": true,
"big_event_context": { ...same shape as regional_context above... }
```

## Suggested UI

**New "Major Event Watch" panel** — above or beside the Escalation Monitor, since it's rarer and higher-stakes:

- Render only `big_event_watch` entries (already filtered + sorted server-side).
- Entry: `M6 34% · 30d` + place name (join via `cluster_id` → sequences) + level chip (amber for watch, red for elevated).
- Badge `first_event: true` entries distinctly — e.g. a "quiet zone" tag. These are the cases where nothing big has happened yet and the model is reading the small-event pattern. That's the model's whole reason for existing.
- Empty state: `"No regions on major-event watch"` — this is the normal state, show it positively.
- On click: fly to location, open the existing sequence detail panel.

**Map**: give watch regions a distinct marker from escalation rings — suggest a slow-pulsing 100km-radius circle (that's the literal spatial meaning of the probability), amber/red by level.

**Tooltip/detail copy suggestion**:
> Regional conditions resemble historical precursors of M6+ events (34% within 30 days vs ~1% baseline). Based on: 47 events in 30d, b-value 0.84, recent quiescence.

## What NOT to do

- Don't merge this into the escalation list — different question, different timescale (30d vs 7d), different spatial meaning (100km region vs sequence).
- Don't show `m6_prob` on every map dot; only `big_event_watch` entries.
- Don't use alarmist language. "Watch" ≠ warning. No countdowns, no "predicted magnitude."
- Don't treat absence of a region as "safe" — 25% of M6+ events strike with no catalog precursor at all. Consider a footnote in the panel: *"Covers regions with detectable precursory activity; ~1 in 4 major events occur without warning signs."*

## Polling

No change — same `/api/event-scores` poll you already do (~60s). The new fields update on the same engine tick.
