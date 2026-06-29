#!/usr/bin/env bash
# SeismicLab — realtime services launcher (Linux / macOS)
#
#   ./run_realtime.sh start     # start all 3 services in the background
#   ./run_realtime.sh status    # show running state
#   ./run_realtime.sh stop      # stop them
#   ./run_realtime.sh restart
#
# Set PYTHON=/path/to/python to override the interpreter.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$HERE/lab"
LOGS="$HERE/logs"; mkdir -p "$LOGS"
PIDFILE="$HERE/.realtime_pids"
PY="${PYTHON:-python3}"

# name : script + args (scoring cadence tunable here)
SERVICES=(
  "emsc_live|$LAB/ingest_emsc_live.py --loop 300"          # offshore foreshocks current (5 min)
  "volcanic_alerts|$LAB/ingest_volcanic_alerts.py --loop 1800"  # volcanic alert overlay (30 min)
  "realtime_engine|$LAB/realtime_engine.py --tick 60"      # tier-2 scoring engine (event-driven)
)

start() {
  if [ -f "$PIDFILE" ]; then echo "already started (see: $0 status). stop first."; exit 1; fi
  command -v "$PY" >/dev/null || { echo "python not found ($PY). set PYTHON=..."; exit 1; }
  [ -f "$HERE/models/tier2_watch_lgb.txt" ] || { echo "model bundle missing — train first (see QUICKSTART.md)"; exit 1; }
  : > "$PIDFILE"
  for s in "${SERVICES[@]}"; do
    name="${s%%|*}"; cmd="${s#*|}"
    nohup $PY $cmd >> "$LOGS/$name.log" 2>&1 &
    echo "$name $!" >> "$PIDFILE"
    echo "started $name (pid $!) -> logs/$name.log"
  done
  echo "SeismicLab realtime is up. Watch feed: data/tier2_watch.json"
}

stop() {
  [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
  while read -r name pid; do
    if kill "$pid" 2>/dev/null; then echo "stopped $name ($pid)"; else echo "$name already gone"; fi
  done < "$PIDFILE"
  rm -f "$PIDFILE"
}

status() {
  [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
  while read -r name pid; do
    if kill -0 "$pid" 2>/dev/null; then echo "  $name  RUNNING  (pid $pid)"; else echo "  $name  DEAD"; fi
  done < "$PIDFILE"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) stop; sleep 2; start ;;
  *) echo "usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
