# Fly brain in a Habitat 3.0 home (follow-a-person)

The simulated fruit-fly brain (FlyWire connectome, calibrated dynamics) drives a
robot in Meta's Habitat 3.0 social-navigation task: a person walks around a real
3D-scanned house (HSSD) and the robot should find and follow them.

```
habitat_server.py  (conda env "habitat", Python 3.9)      brain_client.py  (repo .venv)
  Habitat 3.0 social-nav env, small HSSD house    <--->    fly brain (Session, native engine)
  person geometry through the pet's head camera            robot.head encoders: LC10a (pursuit),
  robot base velocity                                        LC4/LPLC2 (approach), resting ORNs
                                                           ForagingBody motor readout -> velocity
```

The two processes talk over `multiprocessing.connection` on localhost (plain
Python values only).

## What is real and what is approximated
- Person perception is GROUND TRUTH (not a detector): bearing, elevation and
  angular size of the person's head and shoulders (0.5 m) through a virtual head
  camera 0.35 m above the floor, pitched up 20 deg, with the real Orbit head's 53 deg
  field of view (`--hfov`), visible only if a ray from the camera reaches them.
- The same encoders as the real robot head feed the connectome (robot/head.py:
  ObjectEncoder -> LC10a, HeadLoomingStimulus -> LC4/LPLC2, RestingOlfaction).
- Motor readout: fly/body/foraging_body.ForagingBody (DNg100 forward walking,
  DNa01/DNa02 steering, MDN backward, spontaneous walking rhythm). Fly -> robot
  scaling: speed x 0.025 (12 mm/s -> 0.3 m/s), limited to -0.3..0.5 m/s; turn rate
  x 0.5, limited to +-120 deg/s. Takeoffs (escape) become a dash.
- The humanoid walks to random places at ~0.7 m/s (the task's own oracle
  navigation, slowed from its training speed). Robot-person contact is counted as
  a bump and robot-furniture contacts are counted too; the episode goes on (the
  task would end it on a person bump or after too many furniture contacts).
- Spot stands in for the robot body (kinematic base-velocity control).

## Install (Linux box, user-level, no sudo)
```bash
curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3
~/miniforge3/bin/conda create -y -n habitat python=3.9 cmake=3.14.0
~/miniforge3/bin/conda install -y -n habitat habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
~/miniforge3/bin/conda install -y -n habitat -c conda-forge git-lfs
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git ~/habitat-lab
cd ~/habitat-lab && ~/miniforge3/envs/habitat/bin/pip install -e habitat-lab -e habitat-baselines
~/miniforge3/envs/habitat/bin/pip install pillow==10.4.0
# data (~770 MB, ungated): scenes = the Habitat 3.0 benchmark subset of HSSD (3 houses)
mkdir -p ~/habitat-data && ln -sfn ~/habitat-data ~/habitat-lab/data
export PATH=~/miniforge3/envs/habitat/bin:$PATH && git lfs install --skip-repo
for u in habitat_humanoids hab_spot_arm hab3-episodes hab3_bench_assets ycb; do
  python -m habitat_sim.utils.datasets_download --uids $u --data-path data/; done
```
The full HSSD scene set (`hssd-hab`) is gated on HuggingFace (accept the licence,
then `--username/--password <token>`); not needed for the benchmark houses.

## Run
```bash
# one server + one client
~/miniforge3/envs/habitat/bin/python -m sim.habitat_bridge.habitat_server --port 6010 --house small &
PYTHONPATH=. FLY_DYNAMICS=calibrated FLY_DT=0.1 FLY_THREADS=8 \
  .venv/bin/python -m sim.habitat_bridge.brain_client --mode brain --episodes 0 1 --seconds 60 \
  --video-dir $PWD/simulation/outputs/habitat/videos
# the full comparison (brain at 53 and 150 deg field of view, body rhythm without brain, still)
sim/habitat_bridge/run_all.sh
```
Results: `simulation/outputs/habitat/<tag>/<mode>_ep<k>.json` (per-step log and
summary, including Habitat's own `social_nav_stats`) and videos under
`simulation/outputs/habitat/videos/`. The videos come from Spot's `head_rgb`
sensor, which does not follow the base in kinematic mode, so they are in effect a
fixed observer camera watching the robot and the person (the brain's geometry is
computed separately, see above).

## First results (small house, 4 x 60 s episodes per condition, 2026-09-25)
| condition | person visible | time < 2 m | found (visible & < 2 m) | person bumps |
|---|---|---|---|---|
| still robot | 0.1% | 21% | 1/4 | 2 |
| body rhythm, no brain, 53 deg | 0.4% | 19% | 1/4 | 1 |
| fly brain, 53 deg (real head) | 2.5% | 10% | 2/4 | 0 |
| body rhythm, no brain, 150 deg | 5.5% | 19% | 2/4 | 1 |
| fly brain, 150 deg (fly-like view) | 12.6% | 27% | 3/4 | 24 |
With a wide view the brain keeps the person in sight ~2x more and stays close
more often than the brain-less body, but it also runs into them (pursuit has no
stopping distance). n = 4: not significant; a trend to test with more episodes.

## Speed (i9-9900K + RTX 3080 Ti)
- Habitat env with all Spot sensors: ~100 steps/s (1/120 s each), i.e. ~0.8x real time.
- Brain with resting sensory activity: 2.1 ms per simulated ms at 8 threads (memory
  bound; the Mac runs the same brain ~10x faster).

## Neocortex prototype (cortex/, 2026-09-25)
The cortex sits on top of the fly brain and only biases it top-down
(cortex/topdown.py): an FC2 goal direction (central complex), a "phantom"
LC10a target in the goal direction (pursuit pathway), and its critic's
reward-prediction error into the PAM/PPL1 dopamine neurons. The fly brain
still does all the moving.

Oracle check (`run_cortex_oracle.sh`: a cortex that knows the shortest path to
the food bowl; 6 x 60 s, small house, 150 deg view, no mushroom body):
| channel | reached bowl | near bowl (<1 m) | eating | person bumps |
|---|---|---|---|---|
| none (fly brain alone) | 2/6 | 4.6 s | 2.6 s | 21 |
| FC2 goal | 3/6 | 8.8 s | 4.8 s | 19 |
| LC10a attend | 5/6 | 32.5 s | 18.6 s | 29 |
| goal + attend | 6/6 | 32.1 s | 18.5 s | 16 |
The pursuit pathway is the cortex's effective steering route; the FC2 goal
alone is weak (as in brain/navigation/goal.py) but adds to it.

Lifetime (`run_cortex_life.sh`): fly brain alone vs fly + cortex v0 (learned
cognitive map, drives, critic -> dopamine) vs the same cortex with its place
memories wiped every night. `summarise_home.py <dir>` tabulates any run.
