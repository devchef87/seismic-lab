"""SeismicLab Alert Scanner — monitors threat detector, emails on escalation, logs all predictions."""

import os
import json
import base64
import logging
import asyncio
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

log = logging.getLogger("seismiclab.alerts")
UTC = timezone.utc

ENV_PATH = Path(os.environ.get("ENV_PATH", str(Path(__file__).parent / ".env")))
PREDICTION_LOG = Path(__file__).parent / "data" / "predictions.jsonl"
SENDER = os.environ.get("ALERT_SENDER", "")
RECIPIENT = os.environ.get("ALERT_RECIPIENT", "")
TOKEN_URI = "https://oauth2.googleapis.com/token"

ESCALATE_THRESHOLD = 0.60
ALERT_THRESHOLD = 0.75
COOLDOWN_MINUTES = 120
SCAN_INTERVAL_SECONDS = 120


def _load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                os.environ.setdefault(k.strip(), v)


def _get_gmail_service():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GOOGLE_EMAIL_REFRESH_TOKEN"],
        token_uri=TOKEN_URI,
        client_id=os.environ["GOOGLE_EMAIL_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_EMAIL_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _send_email(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg.attach(MIMEText(html_body, "html"))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service = _get_gmail_service()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    log.info(f"Alert email sent: {subject}")


def _build_alert_email(zone, level):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    score_pct = f"{zone['threat_score'] * 100:.0f}%"
    trend_word = "rising" if zone.get("trend", 0) > 0.005 else "falling" if zone.get("trend", 0) < -0.005 else "steady"

    signals_html = ""
    for s in zone.get("signals", []):
        sev = s["severity"]
        bar_color = "#dc4632" if sev > 0.6 else "#d2a032" if sev > 0.3 else "#5a8abf"
        signals_html += f"""
        <tr>
            <td style="padding:6px 12px;font-size:13px;color:#e0e0e0">{s['signal']}</td>
            <td style="padding:6px 12px;font-size:13px;color:{bar_color};text-align:right">{sev * 100:.0f}%</td>
        </tr>
        <tr><td colspan="2" style="padding:0 12px 8px;font-size:11px;color:#888">{s['detail']}</td></tr>"""

    level_color = {"ALERT": "#dc4632", "ELEVATED": "#d2a032", "WATCH": "#5a8abf"}.get(level, "#5a8abf")

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;max-width:640px;margin:0 auto;background:#0a0c12;color:#ccc;border-radius:12px;overflow:hidden">
        <div style="background:linear-gradient(135deg,#0d1117,#161b22);padding:32px 28px 24px;border-bottom:1px solid rgba(255,255,255,0.06)">
            <div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#666;margin-bottom:12px">SeismicLab Alert</div>
            <div style="font-size:28px;font-weight:600;color:{level_color};margin-bottom:4px">{level}</div>
            <div style="font-size:18px;color:#e0e0e0">{zone['name']}</div>
            <div style="font-size:12px;color:#666;margin-top:8px">{ts}</div>
        </div>

        <div style="padding:24px 28px">
            <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.5px">Threat Score</td>
                    <td style="padding:8px 0;font-size:24px;font-weight:600;color:{level_color};text-align:right">{score_pct}</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Trend</td>
                    <td style="padding:8px 0;font-size:14px;color:#e0e0e0;text-align:right">{trend_word} ({zone.get('trend', 0):+.3f})</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Foreshock Rate</td>
                    <td style="padding:8px 0;font-size:14px;color:#e0e0e0;text-align:right">{zone['rate_zscore_6h']:.1f}σ ({zone['events_6h']} events/6h)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">b-value</td>
                    <td style="padding:8px 0;font-size:14px;color:#e0e0e0;text-align:right">{zone.get('b_value_7d', '--')} (Δ {zone['b_delta']:+.3f})</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Rate Acceleration</td>
                    <td style="padding:8px 0;font-size:14px;color:#e0e0e0;text-align:right">{zone['accel_score'] * 100:.0f}%</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Events (24h / 7d / 30d)</td>
                    <td style="padding:8px 0;font-size:14px;color:#e0e0e0;text-align:right">{zone['events_24h']} / {zone['events_7d']} / {zone['events_30d']}</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Geomag Onset</td>
                    <td style="padding:8px 0;font-size:14px;color:{'#d2a032' if zone.get('geomag_onset', {}).get('score', 0) > 0.2 else '#e0e0e0'};text-align:right">{zone.get('geomag_onset', {}).get('score', 0) * 100:.0f}%{' — ' + ', '.join(s['signal'] for s in zone.get('geomag_onset', {}).get('signals', [])) if zone.get('geomag_onset', {}).get('signals') else ''}</td>
                </tr>
                {f'''<tr>
                    <td style="padding:8px 0;font-size:12px;color:#666">Geomag Boost</td>
                    <td style="padding:8px 0;font-size:14px;color:#d2a032;text-align:right">+{zone["geomag_boost"] * 100:.1f}%</td>
                </tr>''' if zone.get('geomag_boost') else ''}
            </table>

            {f'''<div style="margin-top:16px">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#666;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.06)">Active Signals</div>
                <table style="width:100%;border-collapse:collapse">{signals_html}</table>
            </div>''' if signals_html else ''}

            <div style="margin-top:24px;padding:16px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.05)">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#666;margin-bottom:8px">Prediction Record</div>
                <div style="font-size:12px;color:#999;line-height:1.6">
                    This alert was generated automatically by SeismicLab Tier 1 threat detection at <strong style="color:#ccc">{ts}</strong>.
                    Zone <strong style="color:#ccc">{zone['name']}</strong> has crossed the {level.lower()} threshold with a composite threat score of <strong style="color:{level_color}">{score_pct}</strong>.
                    All detector readings at the time of this alert have been logged to the prediction ledger.
                </div>
            </div>
        </div>

        <div style="padding:16px 28px;border-top:1px solid rgba(255,255,255,0.04);text-align:center">
            <a href="http://10.0.0.205:8422" style="color:#5a8abf;font-size:12px;text-decoration:none">Open SeismicLab Dashboard →</a>
        </div>
    </div>
    """
    return html


def _log_prediction(zone, level, alerted):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "zone_id": zone["zone_id"],
        "zone_name": zone["name"],
        "center": zone["center"],
        "level": level,
        "threat_score": zone["threat_score"],
        "trend": zone.get("trend", 0),
        "foreshock_score": zone["foreshock_score"],
        "rate_zscore_6h": zone["rate_zscore_6h"],
        "rate_zscore_1h": zone.get("rate_zscore_1h", 0),
        "b_value_7d": zone.get("b_value_7d"),
        "b_value_30d": zone.get("b_value_30d"),
        "b_delta": zone["b_delta"],
        "accel_score": zone["accel_score"],
        "mag_anomaly_score": zone.get("mag_anomaly", {}).get("score", 0),
        "geomag_onset_score": zone.get("geomag_onset", {}).get("score", 0),
        "geomag_onset_signals": [s["signal"] for s in zone.get("geomag_onset", {}).get("signals", [])],
        "geomag_boost": zone.get("geomag_boost", 0),
        "tidal_stress": zone.get("tidal_stress", 0),
        "tidal_boost": zone.get("tidal_boost", 0),
        "events_1h": zone["events_1h"],
        "events_6h": zone["events_6h"],
        "events_24h": zone["events_24h"],
        "events_7d": zone["events_7d"],
        "events_30d": zone["events_30d"],
        "signals": zone.get("signals", []),
        "alerted": alerted,
    }
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


class AlertScanner:

    def __init__(self, threat_detector):
        self.detector = threat_detector
        self._cooldowns = {}
        self._prev_levels = {}
        self._initialized = False
        _load_env()

    def _should_alert(self, zone_id, level, score):
        if level in ("NORMAL", "WATCH"):
            return False
        if score < ESCALATE_THRESHOLD:
            return False

        prev_level = self._prev_levels.get(zone_id, "NORMAL")
        level_rank = {"NORMAL": 0, "WATCH": 1, "ELEVATED": 2, "ALERT": 3}
        cur_rank = level_rank.get(level, 0)
        prev_rank = level_rank.get(prev_level, 0)

        if cur_rank <= prev_rank:
            return False

        now = datetime.now(UTC)
        last_sent = self._cooldowns.get(zone_id)
        if last_sent and (now - last_sent).total_seconds() < COOLDOWN_MINUTES * 60:
            return False

        return True

    async def scan_once(self):
        try:
            zones = await asyncio.to_thread(self.detector.scan)
        except Exception as e:
            log.error(f"Alert scan failed: {e}")
            return

        if not self._initialized:
            for zone in zones:
                self._prev_levels[zone["zone_id"]] = zone["level"]
            self._initialized = True
            log.info(f"Alert baseline set — {len(zones)} zones: " +
                     ", ".join(f"{z['name']}={z['level']}" for z in zones if z["level"] != "NORMAL"))
            return

        for zone in zones:
            level = zone["level"]
            score = zone["threat_score"]
            zone_id = zone["zone_id"]

            should_alert = self._should_alert(zone_id, level, score)

            if level != "NORMAL":
                _log_prediction(zone, level, alerted=should_alert)

            if should_alert:
                try:
                    subject = f"⚠ SeismicLab {level}: {zone['name']} — {score * 100:.0f}% threat"
                    html = _build_alert_email(zone, level)
                    await asyncio.to_thread(_send_email, subject, html)
                    self._cooldowns[zone_id] = datetime.now(UTC)
                    self._prev_levels[zone_id] = level
                except Exception as e:
                    log.error(f"Failed to send alert for {zone['name']}: {e}")

            self._prev_levels[zone_id] = level

    async def run_forever(self):
        log.info(f"Alert scanner started — checking every {SCAN_INTERVAL_SECONDS}s, "
                 f"escalate ≥{ESCALATE_THRESHOLD * 100:.0f}%, alert ≥{ALERT_THRESHOLD * 100:.0f}%, "
                 f"cooldown {COOLDOWN_MINUTES}min")
        while True:
            await self.scan_once()
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)
