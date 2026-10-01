# Milo on the Jetson Orin Nano

The fly brain runs on the Jetson's GPU (`native/liblif_cuda.so`, ~2x real
time for the complete brain) and drives the Waveshare UGV Rover through the
same brain client as the Habitat studies (`sim/habitat_bridge/brain_client.py
--rover hw`, `robot/rover_world.py`).

```
brain_client --rover hw
  head camera (Logitech Orbit, /dev/video0)        robot/head.py (own process): the person,
    person detector: YOLOX-tiny, TensorRT, GPU       robot/detector.py, native/trt_detect.cpp
  D500 lidar, /dev/ttyUSB0, 230400 baud            robot/d500.py
  UGV base (ESP32), /dev/ttyTHS1, 115200 baud      robot/ugv.py: drive, odometry, battery
```

Each 100 ms control step: the latest sensors -> the connectome (1 ms blocks)
-> the body readout -> motor lag -> lidar safety layer and steering -> the
wheels. The command reaches the wheels at the start of the next step (100 ms
latency). Stops, from the bottom up:
- the base stops by itself if no command arrives for 500 ms (the ESP32's
  heartbeat), so a crashed or stopped brain stops the robot;
- `RoverWorld` sends zero speed while the base's feedback or the lidar's
  scans are older than 0.5 s (a dead serial port, a stalled lidar motor);
  it also sends zero speed until the first scan has arrived;
- `--rover hw` refuses to start without `--lidar --lidar-port`.

The mushroom body's weights and the neocortex's memory are saved every 10
minutes and on exit (atomically), so a stop or a crash loses at most 10
minutes of learning. Caveat: odometry starts at (0, 0) on every start, so a
restored neocortex map (places, obstacles) is anchored to the old start until
the robot can localise itself (lidar SLAM, section 4 of the roadmap); start
from the same spot (the dock) meanwhile.

## One-time setup (needs sudo: run these yourself)

```bash
sudo usermod -aG dialout $USER            # serial ports; log out and back in
sudo cp deploy/jetson/jetson-clocks.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now jetson-clocks
sudo loginctl enable-linger $USER         # only to start Milo at boot without logging in
```

The 40-pin header's UART is `/dev/ttyTHS1` on this Orin Nano (JetPack 7
lists ttyTHS1 and ttyTHS2; Waveshare's own image may call it ttyTHS0: check
with `ls /dev/ttyTHS*` and the base's feedback). The D500's USB adapter
appears as `/dev/ttyUSB0` or `/dev/ttyACM0`; `/dev/serial/by-id/...` keeps
its name across reboots. After `usermod -aG dialout` log out and in (with
lingering, reboot) so the service's user session has the group.

In the repo (no sudo): `uv pip install --python .venv/bin/python pyserial`,
`make -C native && make -C native cuda && make -C native trt`, `mkdir -p ~/milo/runs`.

The person detector's model (YOLOX, Megvii, Apache-2.0; not in the repo) and
its TensorRT engine, built for this GPU (a few minutes each):

```bash
mkdir -p ~/milo/models && cd ~/milo/models
curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
cd ~/fly-lab/fruit-fly-lab && .venv/bin/python -m robot.detector --build ~/milo/models/yolox_tiny.onnx
```

`FLY_PERSON_ENGINE` picks another engine (yolox_nano, yolox_s). The Orbit's
pan/tilt motors are Logitech extension controls that this kernel does not
map by itself: without them the head is a fixed camera (it still sees);
`sudo apt install uvcdynctrl` and replugging the camera should map them (it
did on the Pi). Without the motors the camera is taken to be level (tilt 0;
Habitat's camera looks 20 degrees up, rover_world.CAM_PITCH_DEG), and a
person's head close by is above the picture.

## Check the hardware, piece by piece

```bash
.venv/bin/python -m robot.ugv --port /dev/ttyTHS1           # feedback: battery, wheels
.venv/bin/python -m robot.ugv --port /dev/ttyTHS1 --test    # + drives 0.1 m/s for 1 s: wheels up first!
.venv/bin/python -m robot.d500 --port /dev/ttyUSB0          # scans: put a box in front and check +0 is short
.venv/bin/python -m robot.detector --camera /dev/video0     # people in view, 10 s
PYTHONPATH=. .venv/bin/python -m robot.head --test          # the head process: person azimuth, size
```

To calibrate on the robot (constants at the top of each file):
- `robot/d500.py` `ZERO_DEG`, `CLOCKWISE`: a box straight ahead must give
  the short range at beam +0, one on the right at +90.
- `robot/ugv.py` `TRACK_M`: turn on the spot ten times; set the track so
  that odometry says 3600 degrees.
- The voltage's unit (volts or centivolts) is read automatically.

## Run

```bash
# the whole loop in a simulated room, real brain and real time (no hardware):
PYTHONPATH=. FLY_DATASET=merged FLY_NATIVE_LIB=$PWD/native/liblif_cuda.so \
  .venv/bin/python -m sim.habitat_bridge.brain_client --rover fake --lidar --avoid \
  --cortex pet --learning on --hfov 150 --seconds 60 --out /tmp/rover
# the same with the person from the real camera (--camera auto: the Orbit)
#   ... --rover fake --camera auto --lidar --avoid --cortex pet --learning on ...

# the robot, as a service (deploy/jetson/milo.service)
cp deploy/jetson/milo.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user start milo
journalctl --user -u milo -f
```

The run prints how many 100 ms steps took longer than 100 ms, the brain's
time per step and, with a camera, its frame rate and how long the person was
seen. On the Orin Nano Super (MAXN_SUPER, jetson_clocks), with the neocortex,
mushroom-body learning, lidar senses and steering, a step takes ~66 ms
(2026-09-30, 2-minute runs: mean 66 ms, 95th percentile 75-77 ms, no step
late), the same with the camera and the detector running (15 fps in a dim
room; the detector 3.8 ms a frame at 10 Hz on the GPU, the camera process 8%
of one core). The rover run pipelines the mushroom body's weight updates one
1 ms block later than the Habitat studies (`Session.pipeline_plasticity`),
which is what makes it real time.

Where the time goes (py-spy, 2026-09-30): the loop is CPU-bound; the GPU
waits for Python, not the other way round (3% of the loop waits on it). Each
1 ms block's Python work was ~0.8 ms: the sensory encoders (44%; they
recomputed rates that change at 10 Hz every 1 ms: now computed once per
change, results unchanged), the per-block readout, body and mushroom body
(28%), the neocortex (10%; one PyTorch thread, `FLY_TORCH_THREADS=1`, is as
fast as six on its small nets and leaves the cores free).

## Not yet

- The kit's 160-degree camera on the rover's pan-tilt (the ESP32's T133):
  the head process reads any V4L2 camera, but its pan/tilt and field of view
  are the Orbit's (`robot/head.py` HFOV_DEG, PanTilt). The detector finds
  people, not faces: a face detector (who is looking at Milo) comes later.
- The battery layer (robot/battery.py) from the base's voltage, the charging
  dock, bumpers (the rover has none: contacts are the lidar's).
- The face on the iPad (robot/face_server.py) and the voice.
