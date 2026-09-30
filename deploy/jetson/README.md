# Milo on the Jetson Orin Nano

The fly brain runs on the Jetson's GPU (`native/liblif_cuda.so`, ~2x real
time for the complete brain) and drives the Waveshare UGV Rover through the
same brain client as the Habitat studies (`sim/habitat_bridge/brain_client.py
--rover hw`, `robot/rover_world.py`).

```
brain_client --rover hw
  camera (not yet: the person is never seen)       robot/rover_world.py
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
`make -C native && make -C native cuda`, `mkdir -p ~/milo/runs`.

## Check the hardware, piece by piece

```bash
.venv/bin/python -m robot.ugv --port /dev/ttyTHS1           # feedback: battery, wheels
.venv/bin/python -m robot.ugv --port /dev/ttyTHS1 --test    # + drives 0.1 m/s for 1 s: wheels up first!
.venv/bin/python -m robot.d500 --port /dev/ttyUSB0          # scans: put a box in front and check +0 is short
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

# the robot, as a service (deploy/jetson/milo.service)
cp deploy/jetson/milo.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user start milo
journalctl --user -u milo -f
```

The run prints how many 100 ms steps took longer than 100 ms and the brain's
time per step. On the Orin Nano Super (MAXN_SUPER, jetson_clocks), with the
neocortex, mushroom-body learning, lidar senses and steering, a step takes
~80 ms (2026-09-30, a 15-minute run: 900 s in 903.7 s, mean 79 ms, 95th
percentile 99 ms, 6% of steps a little late; memory steady at ~1.3 GB). The
rover run pipelines the mushroom body's weight updates one 1 ms block later
than the Habitat studies (`Session.pipeline_plasticity`), which is what makes
it real time.

## Not yet

- The camera: the kit's 160-degree USB camera on the pan-tilt, and a person
  detector on the GPU (TensorRT), feeding the same fields the Habitat server
  sends (the person's azimuth, elevation and apparent size). Until then Milo
  sees no one, and its person-speed limit never engages.
- The battery layer (robot/battery.py) from the base's voltage, the charging
  dock, bumpers (the rover has none: contacts are the lidar's).
- The face on the iPad (robot/face_server.py) and the voice.
