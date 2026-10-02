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

### Faces, voice, ears and the dashboard

- `--faces` (with `--camera`): the camera process finds faces (YuNet), tells
  whether each one faces the robot, and recognises it (SFace;
  `robot/faces.py`, `robot/people.py`). Milo greets a person it knows who
  looks at it ("Hi Ben!", at most every 10 minutes) and asks one it does not
  know, out loud: "Hi! I don't think we've met. What's your name?". The answer
  is heard (`--listen`) and read by the language model (Gemma via PAIR:
  `--personality URL` or `$FLY_LLM_URL`, default `127.0.0.1:1234`); Milo
  checks it ("Did I get that right? Your name is Ben?") and only a yes learns
  the face. "No" is respected and nothing is kept. A face is saved only with
  the name its owner gave, in `~/milo/people/` on the robot
  (`python -m robot.people --list`, `--forget NAME`).
- `--voice`: Milo speaks from the rover's speaker (`robot/voice.py`: Pocket
  TTS in Stuart Bell's voice, CC0, made robotic: `FLY_VOICE_STYLE=tin`); the
  speaker is `$FLY_SPEAKER` or the first USB playback device.
- `--listen`: speech recognition on the robot (`robot/speech.py`: Parakeet
  TDT 0.6B v2 int8, sherpa-onnx, 2 CPU threads; muted while Milo speaks).
  Audio is never stored; heard words are shown live on the dashboard and go
  to the personality, but are not saved in the run's log.
- `--dashboard 8080`: a read-only dashboard (`robot/dashboard.py`): open
  `http://milo:8080/?key=<~/.milo_dashboard_key>` (the client prints it).

Models (not in the repository), in `~/milo/models`:

```bash
cd ~/milo/models   # faces (OpenCV Zoo: YuNet MIT, SFace Apache-2.0)
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
mkdir -p asr tts/voices && cd asr   # ears (NVIDIA Parakeet, CC-BY-4.0) and the voice detector
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
cd ../tts          # voice (Kyutai Pocket TTS, CC-BY-4.0) and the reference voice (CC0)
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xjf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
curl -L -o voices/stuart_bell.wav https://huggingface.co/kyutai/tts-voices/resolve/main/voice-zero/stuart_bell.wav
cd ~/fly-lab/fruit-fly-lab
uv pip install --python .venv/bin/python sherpa-onnx
.venv/bin/python -m robot.faces --build                 # TensorRT face engines (else OpenCV on the CPU)
.venv/bin/python -m robot.voice --say "Hello, I am Milo."
```

- `--see-danger` (with `--personality` and the camera): Gemma looks at the
  camera about once a second and says whether something is a real danger to
  Milo (`robot/appraise.py`); only that -- and the personality's `fear` for
  words like "watch out!" -- drives the fly's looming neurons, aimed at the
  danger's side (`robot/threat.py`). People and things coming closer are not
  danger. The brain's escape command then becomes a startle: Milo freezes,
  backs off a little facing the danger (not into anything behind it), watches
  it, beeps (`--voice`), and the console prints `startle (... mode): vision
  saw ...`. To watch the appraisals live and try things in front of the camera:
  `.venv/bin/python -m robot.appraise --camera /dev/video0 --seconds 120 --save /tmp/fear`.

The language model runs on the Mac Studio (PAIR); until PAIR runs on the
Jetson, a reverse tunnel from the Mac gives the Jetson's 127.0.0.1:1234:
`ssh -f -N -R 127.0.0.1:1234:127.0.0.1:1234 milo` (on the Mac).

Measured on the Orin (2026-10-01, max clocks): faces 6.5 ms a frame on the
GPU at 5 Hz; speech 0.2 s for "My name is Ben." (7.4 s of speech in 1.0 s);
everything at once (real lidar, camera, person and face recognition, ears,
voice, dashboard with the 3D brain, neocortex, learning): ~62 ms a 100 ms
step, none late.

### The robot's brain

The rover runs the fly brain without the optic lobes and the nerve cord
(`FLY_TRIM=robot`, set by `--rover`; 51,268 of 165,122 neurons): the whole
robot stack ~36 ms a 100 ms step instead of ~58 (`results/robot_brain_2026-10-01`).
`FLY_TRIM=` (empty) runs the whole brain; `--eye` needs it. Copy
`data/metadata/body_readout_merged_robot.json` with the rest of `data/metadata`.

### The eye: camera and lidar through the fly's optic lobes

`--eye`: what the camera and the lidar see goes through flyvis, the published
model of the fly's visual system (Lappalainen et al. 2024, MIT), into the
connectome's ~61,000 optic-lobe neurons (`robot/eye.py`,
`robot/flyvis_eye.py`; why this and not the connectome's own photoreceptors:
`experiments/vision_ab/README.md`). It runs in its own process on the GPU,
in real time; the dashboard's optic lobes light up when something moves (a
still room fades to rest within seconds, as the fly's visual neurons adapt).

```bash
cd ~/fly-lab/fruit-fly-lab
# flyvis without its training extras (and without touching torch):
uv pip install --python .venv/bin/python --no-deps flyvis==1.2.0 datamate==1.0.0
uv pip install --python .venv/bin/python python-dotenv pytz h5py ruamel.yaml xarray \
    matplotlib joblib toolz tqdm torchvision cachetools scikit-learn
# its pretrained model and connectome (110 MB, from the box or flyvis's download),
# in ~/milo/models/flyvis (FLYVIS_ROOT_DIR): connectome/ and results/flow/0000/000
PYTHONPATH=. FLY_DATASET=merged .venv/bin/python -m robot.eye --build   # once: the connectome mapping
```

Measured on the Orin (2026-10-01): flyvis 1.6 ms of GPU a 20 ms step (a
CUDA graph; PyTorch warns the GPU is unsupported, results match the CPU's to
0.001 Hz); a step of the whole run with the eye ~85 ms (without, ~62). The
brain's GPU blocks wait behind the eye's (two processes take turns on the
GPU), hence 20 ms eye steps, not 10. The robot's settings
(`FLY_EYE_DT=0.02`, `FLY_EYE_BASELINE_S=2`, `FLY_EYE_RATE_MAX=50`) and the
lidar steadying are explained in `robot/eye.py`.

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

Everything at once (2026-10-01, max clocks): the real D500 (`--rover fake
--lidar-port /dev/ttyUSB0`: the real lidar with the simulated base), the
Orbit with the person detector and face recognition (`--faces`, 6.5 ms a
frame on the GPU at 5 Hz), the face server, the neocortex and learning: a
step takes 62 ms (95th percentile 66 ms, no step late in 2 minutes). Since
the day before: the neocortex's critic runs in numpy on the rover
(`FLY_CRITIC=numpy`; 4.4 -> 0.7 ms a step, the same results), and sensory
inputs that did not change are recognised without recomputing (unchanged
encoders return the same read-only array; the session and the engine skip
the update; output identical). What is left: waiting for the GPU's brain
steps 30%, the per-millisecond readout, body and mushroom body 37%, sensory
input 12%, the neocortex 3%. Without `jetson_clocks` (after a reboot) the
same run took 84 ms and was late 13 times: install the unit (one-time setup).

## Not yet

- The kit's 160-degree camera on the rover's pan-tilt (the ESP32's T133):
  the head process reads any V4L2 camera, but its pan/tilt and field of view
  are the Orbit's (`robot/head.py` HFOV_DEG, PanTilt).
- The battery layer (robot/battery.py) from the base's voltage, the charging
  dock, bumpers (the rover has none: contacts are the lidar's).
- Language on the robot itself: the personality and the name reader use
  Gemma on the Mac Studio (PAIR) through a tunnel.
