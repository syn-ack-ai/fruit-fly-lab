#!/usr/bin/env bash
# Start / stop / restart the Fruit Fly Laboratory web server on this machine.
#   tools/lab_server.sh start|stop|restart|status
# Settings (environment, with defaults):
#   FLY_HOST=0.0.0.0  FLY_PORT=8000  FLY_THREADS=3  FLY_PACE=1.0
#   FLY_CAMERA_ROTATE=180  (camera mounted upside down; 0 otherwise)
#   FLY_DYNAMICS=calibrated (or published)
#   FLY_DT=0.2             integration step in ms. The published model uses 0.1;
#                          the Pi 5 is memory-bandwidth bound and runs an
#                          always-active brain in real time only at 0.2 (exact
#                          integration, so only spike-time resolution changes;
#                          calibration targets hold within a few %). 0.1 on the Mac.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/simulation/outputs"
PIDF="$OUT/server.pid"
LOG="$OUT/server.log"
export FLY_HOST="${FLY_HOST:-0.0.0.0}" FLY_PORT="${FLY_PORT:-8000}"
export FLY_CAMERA_ROTATE="${FLY_CAMERA_ROTATE:-180}"
# the more realistic fly: adaptation + antennal-lobe slow inhibition
# (data/metadata/dynamics_calibrated.json); "published" = Shiu et al. 2024 exactly
export FLY_DYNAMICS="${FLY_DYNAMICS:-calibrated}"
export FLY_DT="${FLY_DT:-0.2}"
export FLY_THREADS="${FLY_THREADS:-2}"   # 3+ engine threads starve the Python loop on the Pi
export FLY_BLOCK_MS="${FLY_BLOCK_MS:-2}" # closed-loop interval; halves Python work (see session.py)
mkdir -p "$OUT"

# anchored to a python executable at the start of the command line, so a shell
# or editor whose arguments merely mention these modules is never matched
pids() { pgrep -f '^[^ ]*python[^ ]* -u -m visualization[.]server' ; }
cams() { pgrep -f '^[^ ]*python[^ ]* -m brain[.]sensory[.]camera' ; }

stop() {
  local p
  p="$(pids)"
  if [ -n "$p" ]; then
    kill -TERM $p 2>/dev/null
    for _ in $(seq 1 50); do [ -z "$(pids)" ] && break; sleep 0.1; done
    p="$(pids)"; [ -n "$p" ] && { echo "forcing: $p"; kill -KILL $p 2>/dev/null; sleep 0.3; }
  fi
  p="$(cams)"; [ -n "$p" ] && { echo "stopping orphaned camera worker: $p"; kill -KILL $p 2>/dev/null; }
  rm -f "$PIDF"
  echo "stopped"
}

start() {
  if [ -n "$(pids)" ]; then echo "already running: $(pids)"; return 0; fi
  cd "$ROOT"
  setsid nohup "$ROOT/.venv/bin/python" -u -m visualization.server > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$FLY_PORT/api/state" > /dev/null && break; sleep 0.5
  done
  pids > "$PIDF"
  if curl -sf "http://127.0.0.1:$FLY_PORT/api/state" > /dev/null; then
    echo "running (pid $(cat "$PIDF")) on http://$(hostname -I | awk '{print $1}'):$FLY_PORT"
  else
    echo "failed to start; see $LOG"; tail -20 "$LOG"; return 1
  fi
}

status() {
  local p; p="$(pids)"
  [ -n "$p" ] && echo "server: $p" || echo "server: not running"
  local c; c="$(cams)"; [ -n "$c" ] && echo "camera worker: $c"
  return 0
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 2 ;;
esac
