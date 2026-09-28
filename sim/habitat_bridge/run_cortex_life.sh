#!/usr/bin/env bash
# A pet's lifetime with the prototype neocortex (cortex/v0.py) on the fly brain,
# in the Habitat small house with rewards. For every seed, three lifetimes in
# parallel, all with mushroom-body learning on:
#   none     the fly brain alone
#   v0       fly brain + neocortex (map, drives, critic -> dopamine), memories kept across days
#   amnesic  the same neocortex, but its place memories are wiped every night
#   manners  v0 with cat-like manners around its person (CONDS="none v0 manners")
#   talk     v0 with manners + the LLM personality (cortex/personality.py) and a scripted
#            talking person; needs the model at LLM_URL (default: the NVIDIA PAIR router on the box, port 1234)
#   pet      battery pet: the bowl is a charging dock, hunger = charge, cat-like naps
#   petreflex  control: the battery pet with the old always-on feeding reflex at the dock
#   petlidar the battery pet + a simulated 2D lidar -> looming, antennal touch and obstacle speed limit
#   petobst  ablation: lidar used only for the obstacle speed limit
#   petsense ablation: lidar used only for the fly's looming and touch senses
#   petttc   lidar senses + a footprint time-to-collision filter (Nav2 Collision Monitor "approach")
#   petsteer petlidar + steering toward open space before obstacles (robot/avoid.py)
#   petroute petsteer + the neocortex's obstacle map and routes around it (cortex/obstacle_map.py)
#   pettalk  the battery pet + personality + scripted talking person
# SAFE_SPEED=1 adds the robot's near-person speed limit (robot/safety.py) to every condition.
# SPEECH=1 lets the scripted person talk in every condition (only "talk" listens).
# FACE_URL=http://127.0.0.1:8010/state shows the face (robot/face_server.py); one pet only.
# PORT0 (default 6040) sets the first Habitat port.
# REAL=1: only the real rover's senses (camera, lidar, odometry, battery, dock
# contacts): no odour, no owner petting/treats, no bitter plant (--real-senses).
# DATASET=merged|malecns|fafb picks the brain (default: config.py's default).
# HAB_HOST=user@host runs the Habitat servers on that machine (e.g. the Linux
# box, which renders) and the brains HERE (e.g. the Mac's CPU engine), joined by
# SSH tunnels on the same ports; HAB_DIR is the repo there (default
# fly-lab/fruit-fly-lab), THREADS the CPU threads per brain (default 2).
# BODY=rover gives the pet the Waveshare UGV Rover's footprint instead of Spot's
# (sim/habitat_bridge/bodies.py); bumps are then measured on the rover's body.
# SEEDS="1 2" runs 2 x 3 lifetimes (one Habitat server each). Linux box, CUDA engine.
set -u
cd "$(dirname "$0")/../.."
HAB=~/miniforge3/envs/habitat/bin/python
OUT=${OUT:-simulation/outputs/habitat/cortex_life}
DAYS=${DAYS:-"0 1 2 3 4 5 6 7 8 9"}
SECS=${SECS:-120}
HFOV=${HFOV:-150}
SEEDS=${SEEDS:-1}
CONDS=(${CONDS:-none v0 amnesic})
SAFE=""; [ "${SAFE_SPEED:-0}" = 1 ] && SAFE=--safe-speed
[ "${SPEECH:-0}" = 1 ] && SAFE="$SAFE --speech"
[ -n "${FACE_URL:-}" ] && SAFE="$SAFE --face $FACE_URL"
[ "${REAL:-0}" = 1 ] && SAFE="$SAFE --real-senses"
[ -n "${DATASET:-}" ] && export FLY_DATASET=$DATASET
LLM_URL=${LLM_URL:-http://127.0.0.1:1234/v1/chat/completions}
mkdir -p $OUT
JOBS=()
port=${PORT0:-6040}
for s in $SEEDS; do for c in ${CONDS[@]}; do JOBS+=("$port $c $s"); port=$((port + 1)); done; done
SERVER_ARGS="--hfov $HFOV --house small --max-seconds $SECS --body ${BODY:-spot}"
if [ -n "${HAB_HOST:-}" ]; then
  HAB_DIR=${HAB_DIR:-fly-lab/fruit-fly-lab}
  case "$HAB_DIR" in /*) HAB_ABS=$HAB_DIR ;; *) HAB_ABS="\$HOME/$HAB_DIR" ;; esac
  # the socket key: both ends must share it (sim/habitat_bridge/authkey.py);
  # pass FLY_HABITAT_KEY through if set here, else the remote ~/.fly_habitat_key
  # must equal this machine's
  KEYENV=""; [ -n "${FLY_HABITAT_KEY:-}" ] && KEYENV="--setenv=FLY_HABITAT_KEY=$FLY_HABITAT_KEY"
  UNITS=(); for j in "${JOBS[@]}"; do set -- $j; UNITS+=("hab-split-$1"); done
  # stop only this script's servers (their units) and the tunnel, also on exit
  stop() {
    [ -n "${TUNNEL:-}" ] && kill $TUNNEL 2>/dev/null
    ssh -o BatchMode=yes $HAB_HOST "systemctl --user stop ${UNITS[*]} 2>/dev/null; systemctl --user reset-failed ${UNITS[*]} 2>/dev/null" || true
  }
  trap stop EXIT
  stop; sleep 1
  # the remote clock at start: only "server ready" lines from THIS run count
  # (journalctl keeps the lines of every earlier run of the same unit)
  T0=$(ssh -o BatchMode=yes $HAB_HOST "date +%s")
  for j in "${JOBS[@]}"; do set -- $j
    ssh -o BatchMode=yes $HAB_HOST "cd $HAB_ABS && systemd-run --user --quiet --unit hab-split-$1 --working-directory=$HAB_ABS $KEYENV \$HOME/miniforge3/envs/habitat/bin/python -m sim.habitat_bridge.habitat_server --port $1 $SERVER_ARGS"
  done
  for j in "${JOBS[@]}"; do set -- $j
    ready=0
    for i in $(seq 1 240); do
      ssh -o BatchMode=yes $HAB_HOST "journalctl --user -u hab-split-$1 --since @$T0 --no-pager -q | grep -q 'server ready'" && { ready=1; break; }
      sleep 2
    done
    [ $ready = 1 ] || { echo "Habitat server on port $1 did not start"; exit 1; }
  done
  FWD=(); for j in "${JOBS[@]}"; do set -- $j; FWD+=(-L "$1:127.0.0.1:$1"); done
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes "${FWD[@]}" $HAB_HOST & TUNNEL=$!
  sleep 3
else
  # stop only this script's servers (exact ports), also on exit or Ctrl-C
  stop() { for j in "${JOBS[@]}"; do set -- $j; pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port $1( |$)"; done; }
  trap stop EXIT
  stop; sleep 1
  for j in "${JOBS[@]}"; do set -- $j
    nohup $HAB -m sim.habitat_bridge.habitat_server --port $1 $SERVER_ARGS > /tmp/hserver_$1.log 2>&1 &
  done
  for j in "${JOBS[@]}"; do set -- $j; for i in $(seq 1 240); do grep -q "server ready" /tmp/hserver_$1.log && break; sleep 1; done; done
fi
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=${THREADS:-2} PYTHONUNBUFFERED=1
if [ -z "${FLY_NATIVE_LIB:-}" ] && [ -f native/liblif_cuda.so ]; then export FLY_NATIVE_LIB=$(pwd)/native/liblif_cuda.so; fi
echo "brain engine: ${FLY_NATIVE_LIB:-CPU}; seeds: $SEEDS"
life() {  # port condition seed
  local cx=$2 d=$OUT/$2
  [ $2 = amnesic ] && cx=v0_amnesic
  [ $2 = manners ] && cx=v0_manners
  local extra=""
  [ $2 = talk ] && cx=v0_manners && extra="--personality $LLM_URL --speech"
  # the battery pet: dock = charger, hunger = charge, naps (BATTERY = starting charge)
  [ $2 = pet ] && cx=pet && extra="--battery ${BATTERY:-0.7}"
  [ $2 = petreflex ] && cx=pet && extra="--battery ${BATTERY:-0.7} --dock-reflex"
  [ $2 = petlidar ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar"
  [ $2 = petobst ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar --lidar-use limit"
  [ $2 = petsense ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar --lidar-use senses"
  [ $2 = petsteer ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar --avoid"
  [ $2 = petroute ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar --avoid --route"
  [ $2 = petttc ] && cx=pet && extra="--battery ${BATTERY:-0.7} --lidar --lidar-use ttc"
  [ $2 = pettalk ] && cx=pet && extra="--battery ${BATTERY:-0.7} --personality $LLM_URL --speech"
  [ "$SEEDS" != "1" ] && d=$OUT/seed$3/$2
  mkdir -p $d; rm -rf $d/state $d/mb_weights.npy
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode brain --home --learning on \
      --weights $d/mb_weights.npy --cortex $cx --cortex-state $d/state --seed $3 \
      --episodes $DAYS --seconds $SECS --out $d --body ${BODY:-spot} $SAFE $extra \
      2>&1 | grep --line-buffered -v Warn > $d/lifetime.log
}
PIDS=()
for j in "${JOBS[@]}"; do set -- $j; life $1 $2 $3 & PIDS+=($!); done
wait ${PIDS[@]}
stop
echo done
