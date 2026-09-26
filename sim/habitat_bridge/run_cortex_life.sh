#!/usr/bin/env bash
# A pet's lifetime with the prototype neocortex (cortex/v0.py) on the fly brain,
# in the Habitat small house with rewards. Three lifetimes in parallel, all with
# mushroom-body learning on:
#   none     the fly brain alone
#   v0       fly brain + neocortex (map, drives, critic -> dopamine), memories kept across days
#   amnesic  the same neocortex, but its place memories are wiped every night
set -u
cd "$(dirname "$0")/../.."
HAB=~/miniforge3/envs/habitat/bin/python
OUT=${OUT:-simulation/outputs/habitat/cortex_life}
DAYS=${DAYS:-"0 1 2 3 4 5 6 7 8 9"}
SECS=${SECS:-120}
HFOV=${HFOV:-150}
PORTS=(6040 6041 6042)
CONDS=(none v0 amnesic)
mkdir -p $OUT
stop() { pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port 604[0-2]" ; }
stop; sleep 1
for p in ${PORTS[@]}; do
  nohup $HAB -m sim.habitat_bridge.habitat_server --port $p --hfov $HFOV --house small --max-seconds $SECS > /tmp/hserver_$p.log 2>&1 &
done
for p in ${PORTS[@]}; do for i in $(seq 1 180); do grep -q "server ready" /tmp/hserver_$p.log && break; sleep 1; done; done
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=2 PYTHONUNBUFFERED=1
if [ -z "${FLY_NATIVE_LIB:-}" ] && [ -f native/liblif_cuda.so ]; then export FLY_NATIVE_LIB=$(pwd)/native/liblif_cuda.so; fi
echo "brain engine: ${FLY_NATIVE_LIB:-CPU}"
life() {  # port condition
  local cx=$2
  [ $2 = amnesic ] && cx=v0_amnesic
  mkdir -p $OUT/$2; rm -rf $OUT/$2/state $OUT/$2/mb_weights.npy
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode brain --home --learning on \
      --weights $OUT/$2/mb_weights.npy --cortex $cx --cortex-state $OUT/$2/state \
      --episodes $DAYS --seconds $SECS --out $OUT/$2 \
      2>&1 | grep --line-buffered -v Warn > $OUT/$2/lifetime.log
}
PIDS=()
for i in 0 1 2; do life ${PORTS[$i]} ${CONDS[$i]} & PIDS+=($!); done
wait ${PIDS[@]}
stop
for c in ${CONDS[@]}; do echo "== $c"; cat $OUT/$c/lifetime.log; done
