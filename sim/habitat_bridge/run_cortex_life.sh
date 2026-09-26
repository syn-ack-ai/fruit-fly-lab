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
#   pettalk  the battery pet + personality + scripted talking person
# SAFE_SPEED=1 adds the robot's near-person speed limit (robot/safety.py) to every condition.
# SPEECH=1 lets the scripted person talk in every condition (only "talk" listens).
# FACE_URL=http://127.0.0.1:8010/state shows the face (robot/face_server.py); one pet only.
# PORT0 (default 6040) sets the first Habitat port.
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
LLM_URL=${LLM_URL:-http://127.0.0.1:1234/v1/chat/completions}
mkdir -p $OUT
JOBS=()
port=${PORT0:-6040}
for s in $SEEDS; do for c in ${CONDS[@]}; do JOBS+=("$port $c $s"); port=$((port + 1)); done; done
# stop only this script's servers (exact ports), also on exit or Ctrl-C
stop() { for j in "${JOBS[@]}"; do set -- $j; pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port $1( |$)"; done; }
trap stop EXIT
stop; sleep 1
for j in "${JOBS[@]}"; do set -- $j
  nohup $HAB -m sim.habitat_bridge.habitat_server --port $1 --hfov $HFOV --house small --max-seconds $SECS > /tmp/hserver_$1.log 2>&1 &
done
for j in "${JOBS[@]}"; do set -- $j; for i in $(seq 1 240); do grep -q "server ready" /tmp/hserver_$1.log && break; sleep 1; done; done
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=2 PYTHONUNBUFFERED=1
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
  [ $2 = pettalk ] && cx=pet && extra="--battery ${BATTERY:-0.7} --personality $LLM_URL --speech"
  [ "$SEEDS" != "1" ] && d=$OUT/seed$3/$2
  mkdir -p $d; rm -rf $d/state $d/mb_weights.npy
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode brain --home --learning on \
      --weights $d/mb_weights.npy --cortex $cx --cortex-state $d/state --seed $3 \
      --episodes $DAYS --seconds $SECS --out $d $SAFE $extra \
      2>&1 | grep --line-buffered -v Warn > $d/lifetime.log
}
PIDS=()
for j in "${JOBS[@]}"; do set -- $j; life $1 $2 $3 & PIDS+=($!); done
wait ${PIDS[@]}
stop
echo done
