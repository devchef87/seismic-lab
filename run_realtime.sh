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
  if [ -f "$PIDFILE" ]; then
    while read -r _n p; do kill -0 "$p" 2>/dev/null && { echo "already started (see: $0 status). stop first."; exit 1; }; done < "$PIDFILE"
    rm -f "$PIDFILE"   # stale pidfile (e.g. after reboot/crash) — clear and restart
  fi
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

ensure() {  # (re)start only services that aren't running — idempotent, for a watchdog cron
  command -v "$PY" >/dev/null || exit 1
  [ -f "$HERE/models/tier2_watch_lgb.txt" ] || exit 1
  declare -A alive
  [ -f "$PIDFILE" ] && while read -r n p; do kill -0 "$p" 2>/dev/null && alive[$n]="$p"; done < "$PIDFILE"
  : > "$PIDFILE.tmp"
  for s in "${SERVICES[@]}"; do
    name="${s%%|*}"; cmd="${s#*|}"
    if [ -n "${alive[$name]:-}" ]; then
      echo "$name ${alive[$name]}" >> "$PIDFILE.tmp"
    else
      nohup $PY $cmd >> "$LOGS/$name.log" 2>&1 &
      echo "$name $!" >> "$PIDFILE.tmp"; echo "(re)started $name (pid $!)"
    fi
  done
  mv "$PIDFILE.tmp" "$PIDFILE"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  ensure) ensure ;;
  restart) stop; sleep 2; start ;;
  *) echo "usage: $0 {start|stop|status|ensure|restart}"; exit 1 ;;
esac
