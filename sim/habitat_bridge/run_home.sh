#!/usr/bin/env bash
# A pet's "lifetime" in the Habitat small house with rewards (food bowl, bitter
# plant, owner pets and treats): N episodes ("days") of SECS each, mushroom-body
# learning ON (weights kept across days) vs OFF (same circuit, frozen), in
# parallel on two Habitat servers. Linux box, CUDA engine when built.
set -u
cd "$(dirname "$0")/../.."
HAB=~/miniforge3/envs/habitat/bin/python
NAV=${NAV:-}           # NAV=1: learned valence gates a central-complex goal
OUT=simulation/outputs/habitat/home${NAV:+_nav}
DAYS=${DAYS:-"0 1 2 3 4 5 6 7 8 9"}
SECS=${SECS:-60}
HFOV=${HFOV:-150}
mkdir -p $OUT/on $OUT/off simulation/outputs/habitat/videos/rewards
# stop only this script's servers (exact ports), also on exit or Ctrl-C
stop() { for p in 6020 6021; do pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port $p( |$)"; done; }
trap stop EXIT
stop; sleep 1
for p in 6020 6021; do
  nohup $HAB -m sim.habitat_bridge.habitat_server --port $p --hfov $HFOV --house small --max-seconds $SECS > /tmp/hserver_$p.log 2>&1 &
done
for p in 6020 6021; do for i in $(seq 1 120); do grep -q "server ready" /tmp/hserver_$p.log && break; sleep 1; done; done
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=4
if [ -z "${FLY_NATIVE_LIB:-}" ] && [ -f native/liblif_cuda.so ]; then export FLY_NATIVE_LIB=$(pwd)/native/liblif_cuda.so; fi
echo "brain engine: ${FLY_NATIVE_LIB:-CPU}"
rm -f $OUT/on/mb_weights.npy $OUT/off/mb_weights.npy
life() {  # port learning
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode brain --home --learning $2 \
      --weights $OUT/$2/mb_weights.npy ${NAV:+--nav} --episodes $DAYS --seconds $SECS --out $OUT/$2 \
      --topdown $(pwd)/$OUT/topdown.npz ${VIDEO:+--video-dir $(pwd)/simulation/outputs/habitat/videos/rewards/$2} \
      2>&1 | grep -v Warn > $OUT/$2/lifetime.log
}
life 6020 on & P1=$!
life 6021 off & P2=$!
wait $P1 $P2           # not the servers
stop
echo "ON:"; cat $OUT/on/lifetime.log; echo "OFF:"; cat $OUT/off/lifetime.log
