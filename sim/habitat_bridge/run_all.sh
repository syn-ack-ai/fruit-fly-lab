#!/usr/bin/env bash
# Follow-a-person comparison in the Habitat 3.0 small HSSD house (Linux box).
# Starts two Habitat servers (53-deg = real robot head, 150-deg = fly-like view)
# and runs the fly brain, the body rhythm without brain, and a still robot.
set -u
cd "$(dirname "$0")/../.."
HAB=~/miniforge3/envs/habitat/bin/python
OUT=simulation/outputs/habitat
EPS="0 1 2 3"
SECS=${SECS:-60}
mkdir -p $OUT/videos
# stop only this script's servers (exact ports), also on exit or Ctrl-C
stop() { for p in 6010 6011; do pkill -f "^[^ ]*python -m sim.habitat_bridge.habitat_server --port $p( |$)"; done; }
trap stop EXIT
stop; sleep 1
for spec in "6010 53" "6011 150"; do
  set -- $spec
  nohup $HAB -m sim.habitat_bridge.habitat_server --port $1 --hfov $2 --house small --max-seconds $SECS --video > /tmp/hserver_$1.log 2>&1 &
done
for p in 6010 6011; do for i in $(seq 1 90); do grep -q "server ready" /tmp/hserver_$p.log && break; sleep 1; done; done
export PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=8
# the CUDA brain engine (make -C native cuda) when built: ~5x faster, bit-identical
if [ -z "${FLY_NATIVE_LIB:-}" ] && [ -f native/liblif_cuda.so ]; then
  export FLY_NATIVE_LIB=$(pwd)/native/liblif_cuda.so
fi
echo "brain engine: ${FLY_NATIVE_LIB:-native/liblif.so (CPU)}"
run() {  # port mode tag
  .venv/bin/python -m sim.habitat_bridge.brain_client --port $1 --mode $2 --episodes $EPS \
      --seconds $SECS --out $OUT/$3 --video-dir $(pwd)/$OUT/videos/$3 2>&1 | grep -v Warn
}
run 6010 still fov53
run 6010 body_only fov53
run 6010 brain fov53
run 6011 brain fov150
run 6011 body_only fov150
stop
