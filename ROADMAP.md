# Roadmap: Milo

Milo is a small home robot, in the spirit of Johnny 5 and WALL-E. Its lower
brain is the complete wiring diagram (connectome) of a real fruit fly, a
once-living animal. On top of that sit a small learned "neocortex" and a
language-model personality.

Milo is a robot and says so. It never presents itself as a cat or any other
animal. The goal is that it feels **alive**: smooth, self-driven, responsive
movement. When a choice trades aliveness against the gracefulness side of
safety (occasional bumps), aliveness wins. The hard safety layers are never
part of that trade: the lidar brake, the speed limit near people and the
emergency return.

Hardware: a Waveshare UGV Rover with a Jetson Orin Nano, a D500 lidar, a
camera and a microphone, with an iPad or iPhone as its face.

Status tags: **now** = in progress, **next** = queued, **later** = planned,
**idea** = worth exploring.

## 1. Moving like something alive

- **done** Motor dynamics (`robot/motion.py`): a 0.3 s speed / 1.5 s turn lag
  on the brain's commands is now the default. On held-out seeds, heading
  reversals fell from ~185 to ~2 per minute and aliveness from 56% to 82%,
  with no significant safety cost.
- **done** Orienting reflex (`cortex/v0.py`, `FLY_ORIENT`). When someone comes
  into view, the neocortex points the fly's pursuit pathway at them for 2 s and
  the connectome turns. Held-out: turns toward the person 57/51% -> 69/67%,
  aliveness +2-3 points, no significant safety cost. On by default.
- **next** Getting unstuck: on a few days Milo stays pinned against furniture
  for the rest of the day (or starts the day pinned), which dominates the
  furniture-bump counts. Improve the pinned recovery (`robot/avoid.pivot`,
  `robot/safety.CollisionMonitor`).
- **next** Speed smoothness: with the motor lag, speed variation is 0.56-0.76
  across runs, around the top of the ideal band (0.3-0.7). A longer speed lag
  may settle it.
- **next** Push the aliveness score as high as it will go, one change at a
  time. Tune on tuning seeds, confirm on held-out seeds, and report with
  `sim/habitat_bridge/score_pets.py`.
- **idea** Saccadic turning. Flies turn in quick saccades separated by
  straight runs, and so do many lively robots. Compare with the smooth lag.
- **idea** "Looking around" while stopped: small scanning turns toward
  sounds and movement (camera attention), like Johnny 5's head.
- **idea** A leg-to-wheel model: read stride from the nerve cord's leg motor
  neurons instead of the steering neurons alone (DNa02 shortens ipsilateral
  strides; Yang et al. 2024).

## 2. The fly brain as the voice, the LLM as the language centre

The fly brain decides *when* Milo vocalizes and *what it is about*. The LLM
decides the *words*.

- **next** Find what drives the song circuit in simulation. The male CNS has
  the song command neurons (pIP10, 2) and the nerve-cord song pattern
  generator and wing premotor neurons (dPR1, vPR6, vPR9, TN1a: 41). Courtship
  normally needs a female's scent, and Milo has no nose. Look up P1, pC2 and
  vPN1 under their male-CNS names.
- **next** Milo "sings": when the song circuit is active, play a chirp or trill
  built from the brain's real pulse timing, so its robot sounds come from the
  fly brain rather than the LLM.
- **next** Vocal urge gates speech: the personality speaks when the fly brain
  produces an urge. Candidate triggers are the song command neurons, a startle
  (Giant Fibre), an arousal spike, or a strong reward or disappointment signal
  from the dopamine neurons (the equivalent of the periaqueductal gray gating
  vocalization in mammals).
- **next** A neural-activity readout for the LLM ("your escape neurons fired",
  "your mushroom body marks this spot as good", "your steering neurons pull
  toward your person"). The LLM describes the connectome's real state and does
  not invent feelings. Milo can explain itself truthfully to guests.
- **later** Hearing in the loop: a whistle or clap pattern at the fly's pulse
  rhythm (about 35 ms spacing) reaches the song-tuned hearing pathway through
  `robot/hearing.py`. It becomes an event for the LLM, like a secret handshake.
- **later** More channels from words into the fly brain: "come here" turns
  attention toward the person, alongside the existing praise and scolding,
  which already become dopamine.
- **later** Measure it: honesty and grounding checks in `cortex/llm_bench.py`,
  and in Habitat, whether speech lines up with real neural events.
- **idea** "Are you a cat?" currently gets silence ("what are you?" gets "I am
  a robot."). Make Milo answer briefly
  and cheerfully that it is a robot.

## 3. Seeing what Milo sees and "thinks" (iPhone/iPad dashboard)

- **later** A web app served by the Jetson and opened in Safari, added to the
  Home Screen, extending `robot/face_server.py` and `visualization/server.py`.
  Panels:
  - the camera with detections;
  - the live lidar scan and Milo's map;
  - drives and current goal;
  - live key neurons (steering, escape, feeding, song) and a whole-brain view;
  - the LLM's inner monologue;
  - learned places.
- **later** Home Wi-Fi first; remote access through Tailscale, not port
  forwarding. Watching is open on the home network; control needs the shared
  token.
- **idea** Prototype it against Habitat before the rover arrives.

## 4. The real rover (when the Jetson arrives)

- **later** Jetson setup; move the brain engine to the Jetson, which runs CUDA.
- **later** Hands-on lidar lessons: raw D500 scans, then occupancy grids, then
  SLAM. This is a learning project for the user, done together step by step.
- **later** Carry the motor dynamics, lidar safety layer, hearing and face
  over to the real hardware; compare simulated and real behaviour.
- **later** Calibrate in the home, in three kinds:
  1. **Robot side first:** record Milo at home (lidar, camera, commands, bumps)
     and fit the body interface (fly -> wheel scaling, the motor lag, lidar
     thresholds, camera/lidar -> visual-neuron encoders). The same logs tune
     Habitat toward reality (sim-to-real).
  2. **Grafted networks:** small PyTorch nets at the fly brain's edges, trained
     with RL or from demonstrations at home. Input grafts are learned encoders
     from the real sensors into the fly's own channels; output grafts are a
     cerebellum-like decoder from DN activity to wheels, compensating fragile
     spots and learning the rover's dynamics; neocortex grafts cover maps,
     navigation and attention. The connectome's weights stay fixed.
  3. **Brain dynamics stay calibrated against biology** (the fly exam), never
     tuned to make the robot behave.

  Guard: after each graft, run the ablations (brain disconnected, shuffled
  wiring). If Milo behaves the same without its real connectome, the graft has
  taken over. The hard safety layers stay hand-written, outside any learning.

## 5. Brain and simulation

- **later** A Metal (MLX) engine port for the Mac GPU. There is no CUDA on
  Apple silicon. This enables batched runs for reinforcement learning.
- **later** Reinforcement learning, training only grafted PyTorch networks
  (a future `graft/` package), never the connectome's own weights, plus a training environment.
- **later** Longer-horizon self-care tests: 120 s days never drain the
  battery, so "eats when hungry" (complete brain) vs "snacks" (FAFB) is untested.
- **later** Feeding readout parity: FAFB reads a 62-neuron label group, the
  male brains read MN9 alone. Compare both brains on the same readout.
- **later** Investigate FAFB's higher pet-caused person bumps with lidar steering.

## 6. Open biology limitations

- Odour steering is weak (DNa02 near threshold, and the male's VA1v pheromone
  channel is suppressed by fruit odour). Milo has no nose, so this is
  documented rather than urgent.
- Giant Fibre stray spikes at rest: the two DNg33 cells excite each other
  (~750 synapses each way in the male CNS, FAFB ~140) and latch at 130-140 Hz.
  These are single spikes, below the takeoff threshold, so there is no
  behaviour.
- **next** Latching: after a stimulus, a brain <-> nerve-cord loop (DNg33 <->
  AN09A005 <-> IN09A005, with FR1, optic-lobe or other DN pairs joining) can
  stay on. The exam's `no_latching` fails on some seed sets. It is structural:
  no calibration tried passes it on all of three seed sets, and
  descending-neuron adaptation alone does not fix it. Add the missing
  mechanism, most likely short-term depression at high-rate synapses, then
  recalibrate (and recover robustness, 0.85 vs FAFB 0.95).
- Heat-sense left/right imbalance (13%); the robot has no heat sense.

## Done recently (2026-09-27)

- Merge autapse bug fixed (255 mirror-created autapses: 85 moved to the next
  cell of the type; 170 within-type groups facing a single cell left unmirrored
  at raw strength; the raw data's 26 kept), and the dynamics refitted, now
  choosing on three seed sets: exam 32/32 on the fitting seeds, 30/32 on unseen
  seeds (no_latching and odour symmetry fail), robustness 0.85 (was 0.92).
- Male-CNS class names fixed: 1,314 descending neurons were invisible to the
  readout's "all DNs" list and the raster.
- Docs brought up to date: README quick start, environment variables, both
  brains, and the limitations.
- Stray startles fixed: the long-mode takeoff reads its looming DNs as a
  population. Two spikes of DNp11, driven by the pursuit pathway, launched
  Milo 2.7 times a day with no threat; real looms still trigger it.
- PR #2 merged: FAFB builds can no longer overwrite the merged brain; lab
  controls, route planner and telemetry fixes.
- The complete male-CNS brain became the default: gap-filled and left/right
  balanced.
- Proboscis readout fixed for male brains: the pet could not eat before.
- Real-world Habitat comparison against FAFB, scored 0-100: tied overall. The
  complete brain is far safer, FAFB more active.
- Milo's identity: name, robot face and sounds, an honest personality prompt,
  and honesty checks.
