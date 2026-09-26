#!/usr/bin/env bash
# Can the neocortex steer the fly brain? An ORACLE cortex (knows the shortest
# path to the food bowl) drives the top-down channels one at a time:
#   none    no cortex (the fly brain alone, as in the home lifetimes)
#   goal    FC2 goal -> PFL3 -> DNa02 (central complex)
#   attend  phantom LC10a target -> AOTU -> DNa02 (pursuit pathway)
#   both    goal + attend
# EPISODES x SECS each, four Habitat servers in parallel. Linux box.
set -u
cd "$(dirname "$0")/../.."
HAB=~/miniforge3/envs/habitat/bin/python
OUT=simulation/outputs/habitat/cortex_oracle
EPISODES=${EPISODES:-"0 1 2 3 4 5"}
SECS=${SECS:-60}
HFOV=${HFOV:-150}
PORTS=(6030 6031 6032 6033)
CONDS=(none goal attend both)
mkdir -p $OUT
stop() { pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port 603[0-3]" ; }
stop; sleep 1
for p in ${PORTS[@]}; do
  nohup $HAB -m sim.habitat_bridge.habitat_server --port $p --hfov $HFOV --house small --max-seconds $SECS > /tmp/hserver_$p.log 2>&1 &
done
for p in ${PORTS[@]}; do for i in $(seq 1 180); do grep -q "server ready" /tmp/hserver_$p.log && break; sleep 1; done; done
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=2
if [ -z "${FLY_NATIVE_LIB:-}" ] && [ -f native/liblif_cuda.so ]; then export FLY_NATIVE_LIB=$(pwd)/native/liblif_cuda.so; fi
echo "brain engine: ${FLY_NATIVE_LIB:-CPU}"
run() {  # port condition
  local cx=oracle ch=$2
  [ $2 = none ] && cx=none ch=goal
  [ $2 = both ] && ch=goal,attend
  mkdir -p $OUT/$2
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode brain --home \
      --cortex $cx --channels $ch --episodes $EPISODES --seconds $SECS --out $OUT/$2 \
      2>&1 | grep -v Warn > $OUT/$2/run.log
}
PIDS=()
for i in 0 1 2 3; do run ${PORTS[$i]} ${CONDS[$i]} & PIDS+=($!); done
wait ${PIDS[@]}
stop
for c in ${CONDS[@]}; do echo "== $c"; cat $OUT/$c/run.log; done
