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
- **done** Getting unstuck (`results/unstuck_2026-09-28/`). Milo sat against
  walls for ~25% of the time: the lidar's touch drove the fly's head bristles
  steadily and kept the brain in a backing / freezing touch response. Touch is
  now rapidly adapting, as bristles are, and an unstick reflex points the goal
  at the most open way. Held-out: stuck time 44 -> 15 s/day (lidar) and
  25 -> 2 s/day (lidar steering); furniture bumps 7x fewer.
- **next** With lidar alone (no steering, so no unstick reflex) Milo is stuck
  43 s/day now that it walks at a fly's pace. Decide whether the unstick
  reflex belongs to the lidar layer itself.
- **next** Walking: the complete brain now walks at a fly's pace at rest, but
  in the closed loop only half as far as FAFB (9-13 vs 19-29 m/day;
  `results/walking_latching_2026-09-28/`). Find what holds DNg100 below its
  resting rate in Habitat (P1 excitement, the goal and touch inputs), and make
  the lidar layer speed-aware.
- **done** Speed smoothness: 0.56-0.58 in the latest held-out runs (ideal
  0.3-0.7).
- **done** Why the complete brain walked so little
  (`results/walking_latching_2026-09-28/`). The body's walking-speed constant
  was FAFB's resting DNg100 rate (the male brain rests at ~3 Hz, so a quarter
  pace). Navigation goals also drove the male brain's pursuit neurons, which
  make it walk backward. Now each brain uses its own resting rate, and male
  navigation goes through the central complex.
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

- **done** What drives the song circuit: P1 (86 male pC1 cells named pMP4 /
  pMP-e) drives the song command pIP10 and the nerve cord's song pattern
  generator; nothing Milo senses does it alone (`experiments/song_test.py`,
  `results/milo_voice_2026-09-28/`).
- **done** Milo sings. The neocortex's excitement (you reappearing, petting,
  treats, praise) drives P1, capped below the courtship "lick" step and faded
  to zero within 0.3 m of you (P1 also drives pursuit).
  When pIP10 fires, the face plays the fly's song, slowed into chirps
  (`FLY_VOICE`).
- **done** Vocal urge gates speech. Words are spoken only after the song
  command, a startle, or being addressed, petted or given a treat
  (`FLY_URGE_GATE`); the urge is judged when the call is made.
- **done** Neural-activity readout for the LLM: escape, song, steering,
  walking and the critic's surprise. Grounding checks in `llm_bench` pass.
- **next** Mushroom-body valence in the readout ("this spot is good / bad"
  from the MBON outputs) and a separate reward / punishment dopamine line from
  the fly's own DANs.
- **later** Hearing in the loop: a whistle or clap pattern at the fly's pulse
  rhythm (about 35 ms spacing) reaches the song-tuned hearing pathway through
  `robot/hearing.py`. It becomes an event for the LLM, like a secret handshake.
- **later** More channels from words into the fly brain: "come here" turns
  attention toward the person, alongside the existing praise and scolding,
  which already become dopamine.
- **later** Measure in Habitat whether speech lines up with real neural events
  (a talking-pet run with the urge gate). The grounding and honesty checks in
  `cortex/llm_bench.py` are done.
- **done** "Are you a cat?" -> "I am a robot." (11 of 12).

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
- **done** Latching: brain-wide short-term depression (0.5% per spike, every
  non-sensory neuron). no_latching passes on all four seed sets tested, and
  robustness 0.85 -> 0.89, with no recalibration. Lazy decay keeps the CPU
  engine fast.
- Heat-sense left/right imbalance (13%); the robot has no heat sense.

## Done recently (2026-09-28)

- Milo's voice from the fly brain: excitement -> P1 -> the song command pIP10
  -> Milo's song; the LLM speaks on a vocal urge and describes the fly brain's
  live state; "are you a cat?" -> "I am a robot."
- Getting unstuck: rapidly adapting touch and the unstick reflex.
- Held-out total 75.3% -> 79.0%, aliveness 88 -> 90%, turning toward the
  person 66-70% -> 78-79%.

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
