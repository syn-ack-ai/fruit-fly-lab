# Fruit Fly Laboratory — Milo

> ## About this fork (syn-ack-ai)
> This fork extends the original Fruit Fly Laboratory toward **Milo, a small home
> robot whose lower brain is the connectome of a real fruit fly**. On top of it sit
> a small learned "neocortex" and an optional language-model personality. Milo is a
> robot and says so: it never presents itself as a cat or any other animal (robot
> face and sounds; honesty checks in `cortex/llm_bench.py`). Plans are in
> `ROADMAP.md`. Additions are MIT-licensed (see LICENSE); upstream code keeps its
> author's copyright.
>
> - **Default brain since 2026-09-27: our "complete" male CNS** (`FLY_DATASET=merged`,
>   `brain/connectivity/merge.py`). It is the Janelia MaleCNS v1.0 (brain + ventral
>   nerve cord; 165,122 neurons, 89.4 M synapses) with its demonstrated
>   reconstruction gaps filled and made left/right balanced, with dynamics refitted
>   to it. The filled gaps are the left-antenna smell neurons, Johnston's organ and
>   the head bristles; FlyWire FAFB decides what counts as a gap and sets the fill
>   targets. Fly exam: 30/32 on a seed set never used for fitting (21/21 held-out
>   tests, 3/3 constraints; only odour steering fails), 31/32 on the fitting seeds,
>   robustness 0.89 (FAFB 0.95). Odour steering is a documented limitation; the
>   robot has no nose.
>   It has no left/right steering bias (FAFB's wiring has one). See `results/complete_brain_2026-09-27/`. FlyWire FAFB (the original
>   female brain) remains available with `FLY_DATASET=fafb`.
> - **Robot behaviour** (`robot/`, `cortex/`):
>   - lidar obstacle avoidance that bends the fly brain's own steering toward
>     open space (`results/organic_avoidance_2026-09-27/`);
>   - a battery pet that charges at its dock when hungry and naps;
>   - an orienting reflex;
>   - motor dynamics for lifelike movement (`robot/motion.py`);
>   - a population readout for the long-mode escape, so Milo no longer startles
>     at nothing.
> - **Milo's voice from the fly brain** (`results/milo_voice_2026-09-28/`). The
>   neocortex's excitement drives the male P1 neurons; the connectome's song
>   command pIP10 fires; Milo plays the fly's song; the LLM speaks only on a
>   vocal urge and describes the fly brain's live state.
> - **Getting unstuck** (`results/unstuck_2026-09-28/`). Touch is now rapidly
>   adapting, and an unstick reflex frees Milo from walls: with lidar steering,
>   stuck time fell ~90%. (With lidar alone it later returned to ~40 s/day
>   once Milo walked faster; the unstick reflex needs the steering layer.)
> - **Walking and latching** (`results/walking_latching_2026-09-28/`). The body's
>   walking speed now uses each brain's own resting DNg100 rate (it used FAFB's:
>   the male brain walked at a quarter pace). Male navigation goes through the
>   central complex (its pursuit neurons made it walk backward). Brain-wide
>   short-term depression stops post-stimulus latching (robustness 0.89).
> - **Which head bristles are touch** (`results/touch_subtypes_2026-09-28/`). The
>   male brain's 69 untyped bristle neurons were sorted by wiring
>   (`brain/sensory/bm_subtypes.py`); lidar touch skips the fronto-orbital-like
>   ones, which made Milo back up and freeze at walls.
> - **The body's readouts, fitted to each brain** (`results/body_readout_2026-09-29/`).
>   They were tuned on FAFB, whose command neurons fire several times faster.
>   The male robot's proboscis flickered at its dock (it never finished a
>   meal), chance bursts of two MDN cells walked it backward 7.5% of the time,
>   and speed limits clipped its noisy DNg100. Now each brain is read relative
>   to its own rates, over the same expected number of spikes. Held-out: Milo
>   eats like FAFB, walks ~40% further with lidar steering, and the complete
>   brain ties FAFB overall (79.8% vs 79.8%). FAFB's resting rate itself was
>   stale (18.5 Hz, not 14.6).
> - **Real-world Habitat test** (no smell, camera and lidar only), scored 0-100 on
>   safety, self-care, life and "aliveness" (`sim/habitat_bridge/score_pets.py`).
>   See `results/habitat_real_world_2026-09-27/`.
> - `native/`: a C engine (multi-threaded) and a CUDA engine, bit-exact with the
>   published Python model.
> - Calibrated dynamics (`data/metadata/dynamics_calibrated*.json`, applied in
>   `simulation/engine/session.py`). They add resting ORN input (Hallem & Carlson
>   2006), ORN->PN compensation and normalisation, ORN depression (Olsen 2010),
>   ipsilateral release (Gaudry 2013), adaptation (including nerve cord and
>   descending neurons) and Giant Fibre corrections. Every change cites its source.
> - `cognition/exam/`: a 32-test validation battery (the "fly exam") with
>   shuffled-wiring controls and robustness curves.
> - Also: `brain/plasticity/` (dopamine-gated mushroom-body learning),
>   `brain/navigation/` (E-PG compass, FC2 -> PFL3 goal steering) and
>   `sim/habitat_bridge/` (the brain driving a robot in Meta Habitat 3.0 homes).
>
> The brain and the neocortex use no language model (connectome -> differential
> equations -> spikes; the cortex is a small learned map, drives and a TD critic).
> The optional personality layer (`cortex/personality.py`) uses a local language
> model for speech and intentions only. It never reaches the motors or the safety
> layer. The code in this fork was written with AI coding assistance (Anthropic's
> Claude) under human direction.

## Quick start (this fork)

```bash
python -m pip install -r requirements/requirements.txt   # + requirements-robot.txt for the camera
make -C native                                           # the C engine (the default brain needs it)
python -m pytest tests -q                                # the merged brain; FLY_DATASET=fafb for FAFB
```

**Building the brains.** Data: FlyWire FAFB v783 (section 3 below) and the Janelia
MaleCNS v1.0 release (`DATA_SOURCES.md` §1b).

```bash
python -m brain.connectivity.build_connectome                               # FAFB (always FAFB paths)
python -m brain.connectivity.build_malecns /path/to/malecns_v1.0            # MaleCNS
python -m brain.connectivity.malecns_annotations /path/to/malecns_v1.0      # labels, columns, dimorphism
python -m brain.sensory.crossmap                                            # MaleCNS <-> FlyWire modality map
python -m brain.connectivity.merge                                          # the complete brain (needs both)
```

**The fly exam:**

```bash
python -m cognition.exam --dynamics calibrated --workers 6
FLY_EXAM_SEED_OFFSET=200 python -m cognition.exam --dynamics calibrated --no-robustness
```

The second run uses a seed set that was never used for fitting.

**Milo in Habitat.** Habitat runs in its own conda environment
(`sim/habitat_bridge/README.md`). Split mode runs Habitat on a Linux GPU box and
the brains locally:

```bash
HAB_HOST=user@box REAL=1 BODY=rover CONDS="petlidar petsteer" SEEDS="1 2 3" \
    OUT=simulation/outputs/habitat/run bash sim/habitat_bridge/run_cortex_life.sh
python -m sim.habitat_bridge.score_pets milo=simulation/outputs/habitat/run
```

**Milo on the rover (Jetson).** The same brain client drives the Waveshare UGV
Rover in real time (`deploy/jetson/README.md`); `--rover fake` runs it in a
simulated room without hardware:

```bash
FLY_NATIVE_LIB=$PWD/native/liblif_cuda.so python -m sim.habitat_bridge.brain_client \
    --rover fake --lidar --avoid --cortex pet --learning on --hfov 150 --seconds 60
```

**Face (iPad / browser):** `python -m robot.face_server --port 8010`.

**Environment variables:**

| variable | meaning (default) |
|---|---|
| `FLY_DATASET` | `merged` (default), `malecns` (raw male), `fafb` (FlyWire female) |
| `FLY_DYNAMICS` | `published` (Session default) or `calibrated` (the robot, exam and lab use calibrated) |
| `FLY_ENGINE`, `FLY_NATIVE_LIB`, `FLY_THREADS` | engine choice, the CUDA library (`native/liblif_cuda.so`), threads |
| `FLY_GAIN` | override the dataset's calibrated synaptic gain (`calibration_<dataset>.json`) |
| `FLY_TORCH_DEVICE` | the neocortex's small networks (`cuda` if available; `cpu` on the rover) |
| `FLY_CUDA_NO_GRAPH` | CUDA engine: launch directly, not through CUDA graphs (for Nsight Compute) |
| `FLY_TORCH_THREADS` | the neocortex's PyTorch threads (1 on the rover) |
| `FLY_PERSON_ENGINE`, `FLY_TRT_LIB` | the camera's person detector on a Jetson: its TensorRT engine (`~/milo/models/yolox_tiny.engine`) and `native/libtrt_detect.so` (`robot/detector.py`) |
| `FLY_MOTOR_TAU` | robot motor lag `tau_v,tau_w[,stages[,tau_w_fast]]` in s (`0.3,1.5`; `0` = off) |
| `FLY_ORIENT` | the neocortex's orienting reflex for the pet (`1`) |
| `FLY_VOICE` | excitement -> P1 -> the fly's song command pIP10 -> Milo's "song" (`1`; male brains) |
| `FLY_URGE_GATE` | the personality speaks only after a vocal urge from the fly brain (`1`) |
| `FLY_UNSTICK` | the unstick reflex with lidar steering (`1`) |
| `FLY_UNSTICK_LIDAR` | the unstick reflex with lidar alone too (`0`: it made things worse, `results/touch_subtypes_2026-09-28/`) |
| `FLY_UNSTICK_CHANNEL` | the unstick reflex's channel: `goal` (male default) or `attend` (FAFB default) |
| `FLY_TOUCH_ADAPT` | lidar touch as rapidly adapting bristles (`1`; `0` = the old steady drive) |
| `FLY_TOUCH_BM` | male brains: lidar touch drives the antennal-like head bristles (`subtypes`, `brain/sensory/bm_subtypes.py`) or `all` 69 "BM" cells |
| `FLY_ATTEND_FROM_GOAL` | navigation goals also drive the pursuit neurons LC10a (FAFB `1`; male brains `0`: it made them walk backward) |
| `FLY_ORIENT_CHANNEL` | the orienting reflex's channel: `attend` (LC10a, default on every brain) or `goal` (central complex) |
| `FLY_PROBOSCIS_HOLD` | the proboscis follows its motor neurons' rate over ~1 s, scaled to FAFB's (`1`; `0` = the 50 ms readout before 2026-09-29) |
| `FLY_WALK_CMD_TAU` | s over which walking-direction commands must hold, with DNg100 read relative to rest (`0.2`; `0` = the readout before 2026-09-29) |
| `FLY_DNG100_TAU_SCALE` | DNg100 smoothing scaled to the brain's spike count (`1`; `0` = FAFB's times on every brain) |
| `FLY_POP_READOUT` | pooled descending-neuron readout for all channels (off; the long-mode escape is always pooled) |
| `FLY_EXAM_SEED_OFFSET` | shifts every exam seed (0; 100 = calibration's second set, 200 = report set) |
| `FLYWIRE_V783_DIR` | FlyWire files (default: a `flywire_v783/` folder next to the repository) |
| `FLY_FACE_KEY`, `FLY_HABITAT_KEY` | shared secrets for the face server and the Habitat socket |

Run-script variables (`sim/habitat_bridge/run_cortex_life.sh`): `HAB_HOST` (split
mode), `REAL=1` (real senses: no smell), `BODY=rover`, `DATASET`, `SEEDS`, `DAYS`,
`CONDS`, `THREADS`, `PORT0`, `OUT`, `SAFE_SPEED` (the near-person speed
limit, on by default; `0` turns it off).

---

## The original laboratory (FlyWire FAFB)

The sections below describe the original project, which this fork keeps working
with `FLY_DATASET=fafb`.

An interactive simulation of an adult female *Drosophila melanogaster* built on
the **real FlyWire FAFB v783 connectome** — 139,255 neurons, 3,732,460
connections, 50,666,648 synapses — running the **published whole-brain
leaky integrate-and-fire model** of Shiu et al. (2024, *Nature* 634:210–219).

Throw a rock at the fly and it escapes. The escape is not a rule anyone wrote:
it comes out of real LC4 and LPLC2 neurons driving the real Giant Fibre through
the real wiring diagram. Silence those 314 neurons and the escape disappears
entirely.

**No large language model is used in the brain simulation.** No transformer,
no chatbot, no agent, no RAG, no LLM-generated rules. The pipeline is
connectome → differential equations → spikes. (This fork's optional robot
personality layer is the one exception, and it sits outside the simulation.)

---

## The pipeline

```
looming object (exact geometry)
  → LC4 / LPLC2 firing rates      published tuning × FlyWire-derived receptive fields
  → Poisson drive onto 314 REAL FlyWire neurons
  → 139,255-neuron LIF simulation over 3,732,460 real connections
  → real descending neurons (DNp01 = Giant Fibre)
  → motor channels
  → digital fly body
```

---

## 1. Exact software to install

| Requirement | Version used |
|---|---|
| Python | 3.13.15 (3.11+ works) |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| pandas | 3.0.6 |
| fastapi | 0.141.1 |
| uvicorn | 0.53.0 |
| websockets | 17.1 |
| pytest | 9.1.1 |
| pyarrow | 25.0.1 (MaleCNS build) |
| torch | 2.14.0 (the neocortex's critic) |

Optional: `brian2`, to cross-check against the original reference
implementation. Not needed to run the laboratory.

**Engines.** `simulation/engine/lif_engine.py` is the reference Python engine;
`tests/test_lif_engine.py::test_matches_literal_reference_implementation` is
the equivalence test any backend must pass. This fork adds a multi-threaded C
engine (`make -C native`; it is the default when built, and the male brains
need it because they run with calibrated gain and dynamics) and a CUDA engine
(`make -C native cuda`, then `FLY_NATIVE_LIB=native/liblif_cuda.so`; on the
rover's Jetson Orin Nano it runs the complete brain at ~2x real time). Both are
verified against the Python engine (`native/verify_native.py`,
`native/verify_cuda.py`; `native/README_CUDA.md`).

## 2. Exact commands to install dependencies

```bash
python -m pip install -r requirements/requirements.txt
make -C native
```

Optional extras:

```bash
python -m pip install -r requirements/requirements-dev.txt
```

## 3. Exact commands to obtain the real FlyWire data

The project looks for the data in a `flywire_v783/` folder next to the
repository, or wherever `FLYWIRE_V783_DIR` points. Its SHA-256 checksums are
recorded in `data/metadata/flywire_v783_checksums.txt`.

To obtain it from scratch:

1. Create a free FlyWire account and accept the data licence at
   <https://flywire.ai> (the download requires sign-in; there is no anonymous
   URL, so this step cannot be scripted).
2. Go to <https://codex.flywire.ai/api/download>, select **data version 783**,
   and download at minimum these files:

   ```
   neurons.csv.gz
   classification.csv.gz
   consolidated_cell_types.csv.gz
   connections_princeton.csv.gz
   coordinates.csv.gz
   column_assignment.csv.gz
   labels.csv.gz
   visual_neuron_types.csv.gz
   cell_stats.csv.gz
   names.csv.gz
   ```

3. Put them in one directory and point the project at it:

   ```bash
   export FLYWIRE_V783_DIR="/path/to/FlyWire Brain Dataset (FAFB v783)"
   ```

   On Windows PowerShell:

   ```powershell
   $env:FLYWIRE_V783_DIR = "D:\Fruitfly\FlyWire Brain Dataset (FAFB v783)"
   ```

4. Verify the download matches what this project was built against:

   ```bash
   FLY_DATASET=fafb python -m pytest tests/test_data_provenance.py -q
   ```

5. Build the simulation-ready connectome (about 2 minutes):

   ```bash
   python -m brain.connectivity.build_connectome
   ```

   This command always writes the FAFB artefacts, regardless of `FLY_DATASET`.
   To run the FAFB experiments below, set `FLY_DATASET=fafb` (`export FLY_DATASET=fafb`
   on macOS/Linux; `$env:FLY_DATASET = "fafb"` in PowerShell). The default `merged`
   brain also requires a MaleCNS build and `python -m brain.connectivity.merge`;
   see [dataset provenance](DATA_SOURCES.md#1b-the-default-brain-since-2026-09-27-janelia-malecns-v10-gap-filled-merged).

   Expected output ends with:

   ```
   n_neurons        139255
   n_neuron_pairs  3732460
   n_synapses     50666648
   ```

## 4. Exact repository / files being used

| Purpose | Source |
|---|---|
| Connectome data | FlyWire FAFB **v783**, <https://codex.flywire.ai/api/download> |
| Neuron model | Shiu et al. 2024, [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9) |
| Reference code inspected | <https://github.com/philshiu/Drosophila_brain_model> — file `model.py` |
| Also inspected | <https://github.com/eonsystemspbc/fly-brain> |
| Not used | `snedea/flybrain`, `erojasoficial-byte/fly-brain` |

No third-party repository is vendored. `brain/neuron_models/lif.py` transcribes
the constants and equations from the authors' `model.py`, and
`tests/test_lif_engine.py` proves our engine is spike-for-spike identical to a
literal transcription of that Brian2 network. Full detail in
[`DATA_SOURCES.md`](DATA_SOURCES.md).

## 5. Hardware requirements

| | |
|---|---|
| CPU | x86-64 or ARM64 (Apple silicon, Jetson). The C engine is multi-threaded (`FLY_THREADS`). |
| RAM | **4 GB minimum, 8 GB comfortable.** The sparse connectome is ~45 MB in memory; the build step peaks around 3 GB while reading the 5.3M-row connectivity table. |
| Disk | 1.5 GB for the ten required FlyWire source files (the MaleCNS release is larger; see `DATA_SOURCES.md` §1b). (The full Codex release including meshes is 17 GB; most of it is unused — see `DATA_SOURCES.md`.) |
| GPU | Not required; optional CUDA engine. |

Performance: the reference Python engine takes about 1.0 ms of wall-clock time
per 0.1 ms simulated step on FAFB (10x slower than real time). The C engine runs
the complete male brain at about 1.7x real time on 1 thread and 4.5x on 4
threads (Apple M-series); CUDA runs the merged brain at about 12x on an RTX
3080 Ti and 2.6x on a Jetson Orin Nano (`native/README_CUDA.md`). Spike propagation is event-driven, so cost scales with
spike count.

## 6. First experiment to run

```bash
FLY_DATASET=fafb python -m experiments.02_escape_controls
```

This is the experiment that shows the escape is real. Expected output (FAFB):

```
condition                   LC4 spk  LPLC2 spk  DNp01 spk   peak DNp01     active
---------------------------------------------------------------------------------
looming                         312        573         37        210 Hz       1042
receding                          0          0          0          0 Hz          0
static                            0          0          0          0 Hz          0
looming, -LC4                   312        573         34        150 Hz        779
looming, -LPLC2                 312        573         28        210 Hz        642
looming, -LC4/-LPLC2            312        573          0          0 Hz        182
```

The last row is the point: with the 104 real LC4 and 210 real LPLC2 neurons
silenced, those neurons still fire (312 and 573 spikes) but send nothing — and
the Giant Fibre goes to **exactly zero**.

Then the other two:

```bash
FLY_DATASET=fafb python -m experiments.01_looming_escape      # time course of one escape
FLY_DATASET=fafb python -m experiments.03_touch_and_feeding   # feeding, touch, and honest refusals
```

And the interactive laboratory. There are two builds, and they run the same
simulation:

**Browser build** (what is deployed). The whole simulation runs client-side in a
Web Worker; the server is only a static file host.

```bash
FLY_DATASET=fafb python -m tools.export_web_connectome
```

```bash
python -m http.server 8001 --directory web
```

**Python server build** (original). Streams telemetry over a WebSocket.

```bash
python -m visualization.server
```

Open <http://127.0.0.1:8000>. The controls are:

| Control | What it does |
|---|---|
| **🪨 THROW ROCK** | Looming object at a chosen azimuth, speed and size → 104 LC4 + 210 LPLC2 |
| **🍎 PLACE FOOD** | Food odour (266 ORNs) + tarsal contact chemosensation (71 GRNs) + proboscis sugar (23 GRNs), together |
| Stimulus list | 15 further modalities, each labelled with how many real neurons it drives; 4 shown as *not modeled* with the reason |
| Lesion buttons | Silence LC4 / LPLC2 / DNp01 / JO-A / JO-B and repeat the experiment |
| **⟲ Replay experiment** | Scrub the whole recorded run frame by frame |
| Run / Pause / Reset | Transport |
| 3D neurons / Spike raster | All 139,255 neurons at true anatomical positions, or the raster |
| Circuit inspector | Real inputs and outputs of any cell type, with a link to its Codex page |
| Provenance | Every component tagged A / B / C / D with citations |

## 6b. Deploying

The Python server is stateful — a background thread holding 139,255 neurons of
simulation state, streaming over a WebSocket. That cannot run on Vercel, which
has no persistent processes, no WebSockets and no cross-request state.

So the deployed build moves the simulation into the browser. `web/` is a static
site: it downloads the real connectome as a binary asset (~12 MB gzipped, and
Vercel serves it Brotli-compressed) and runs the identical LIF model in a Web
Worker. Nothing is simplified for the web — all 139,255 neurons and all
3,732,460 connections are present, and the engine is verified against the Python
one:

```bash
FLY_DATASET=fafb python -m tools.verify_web_engine
```

That runs the same seed and stimulus through both engines with a shared
deterministic PRNG and asserts the spike counts are identical for every neuron.
All three scenarios (deterministic propagation, Poisson looming drive, and
lesioned) must match exactly before deploying.

Set the Vercel project's **Root Directory to `web`** — that is the whole
configuration. See [`DEPLOY.md`](DEPLOY.md). If it points anywhere else, every
route 404s, and running `npx vercel` from inside a subdirectory is the usual
cause.

```bash
npx vercel --cwd web --prod
``` Browser requirements: a modern browser with module Web Workers
(Chrome/Edge 91+, Firefox 114+, Safari 15+), about 200 MB of tab memory, and the
one-time connectome download.

## 7. How to verify the simulation uses real FlyWire neurons and connections

```bash
FLY_DATASET=fafb python -m pytest tests/ -q
```

The suite has about 213 tests (the FAFB provenance checks run only with
`FLY_DATASET=fafb`). The ones that matter for this question:

| Check | Test |
|---|---|
| Source files are byte-identical to the documented FlyWire release | `test_source_file_checksum` (SHA-256 of all 6 core files) |
| Exactly 139,255 neurons | `test_neuron_count_matches_published_v783` |
| Every root ID carries the FAFB `720575940` segmentation prefix | `test_all_root_ids_have_fafb_segmentation_prefix` |
| Exactly 50,666,648 synapses | `test_synapse_total_matches_published` |
| The published super-class census (77,873 optic, 1,305 descending, …) | `test_super_class_counts_match_v783` |
| The published neurotransmitter census | `test_neurotransmitter_census_matches_v783` |
| Real population sizes (LPLC2 = 210, LC4 = 104, DNp01 = 2) | `test_cell_type_population_sizes` |
| A fabricated root ID is *rejected*, not invented | `test_a_fabricated_root_id_is_rejected` |
| Neuron positions span a real fly brain (814 × 392 × 278 µm) | `test_neuron_positions_span_a_real_fly_brain` |
| Our engine == a literal transcription of the published Brian2 model | `test_matches_literal_reference_implementation` |

And the biological checks — facts discovered in wet labs that fall out of the
loaded wiring diagram, hard-coded nowhere:

| Known biology | Test |
|---|---|
| LC4 is the largest cell-type input to the Giant Fibre; LPLC2 is also top-5 | `test_lc4_and_lplc2_are_the_top_visual_inputs_to_the_giant_fibre` |
| The Giant Fibre also receives Johnston's-organ (JO-A/JO-B) input | `test_giant_fibre_receives_antennal_mechanosensory_input` |
| LPLC2 pools all four T4/T5 directional subtypes | `test_lplc2_receives_t4_t5_motion_input` |
| R1-6 photoreceptors target L1, L2, L3 (the lamina cartridge) | `test_photoreceptors_target_the_lamina_monopolar_cells` |
| Sugar GRNs drive proboscis motor neurons; bitter GRNs do not | `test_sugar_drives_proboscis_motor_neurons`, `test_bitter_does_not_drive_proboscis_motor_neurons` |
| Cutting LC4+LPLC2 abolishes the Giant Fibre response | `test_silencing_lc4_and_lplc2_abolishes_the_giant_fibre_response` |
| An unstimulated brain is completely silent | `test_an_unstimulated_brain_is_silent` |
| The body cannot move without descending activity | `test_body_does_nothing_without_descending_activity` |

**Verify a neuron by hand.** Every neuron in the UI's circuit inspector links to
its Codex page. For example the left Giant Fibre:
<https://codex.flywire.ai/app/cell_details?root_id=720575940622838154> —
compare the cell type, side, and partner list against what the app shows.

## 8. Known scientific limitations

Summarised here for the **FAFB reference brain**; the full treatment is in
[`BIOLOGICAL_ASSUMPTIONS.md`](BIOLOGICAL_ASSUMPTIONS.md). The default complete male
brain differs in several of these:

- it includes the ventral nerve cord (items 1-2);
- the MaleCNS treats histamine as inhibitory (item 3);
- 3,275 of its neurons are unsigned (item 7);
- it is one male (item 10).

The calibrated dynamics add resting input, adaptation and depression (items
5-6), and `brain/plasticity/` adds mushroom-body learning. See §11 of
`BIOLOGICAL_ASSUMPTIONS.md`.

1. **Brain only — no ventral nerve cord.** The leg and wing motor neurons are
   not in this dataset. The simulation ends at the descending neurons, which are
   genuinely the brain's only output. Proboscis motor neurons are the one
   exception; they are in the brain.
2. **Touch on thorax, abdomen and legs cannot be modelled** for the same reason,
   and is reported as "Not currently modeled" rather than faked.
3. **Histamine is missing from FlyWire's neurotransmitter vocabulary.**
   Photoreceptors are histaminergic and inhibitory; the dataset labels R1-6 as
   excitatory ACh. Driving light through photoreceptors would invert the sign of
   the whole early visual pathway, so it is **disabled**.
4. **No gap junctions.** The Giant Fibre's output onto motor neurons is largely
   electrical and is therefore absent.
5. **No spontaneous activity or inhibitory tone.** With no stimulus the network
   is silent. This makes the Giant Fibre fire at smaller angular sizes than a
   real fly would tolerate.
6. **Point neurons.** No dendritic computation, no plasticity, no adaptation, no
   neuromodulatory state (hunger, arousal, circadian phase).
7. **Neurotransmitters are predicted, not measured**, and 1,626 neurons (1.2%)
   have no resolvable sign and produce no output.
8. **Sensory transduction is modelled, not simulated.** The tuning curve shapes
   and their constants are ours; the neurons they drive are real. This is the
   honest weak point of the pipeline.
9. **Descending-neuron → behaviour assignments come from the literature**, not
   from the connectome, and cover 20 descending cell types (plus 3 pooled in the
   population readout) of ~473.
10. **This is one fly.** A single adult female, imaged once.
11. **The body is kinematic**, not biomechanical.

**This simulation does not reproduce the subjective experience, awareness, or
consciousness of a fruit fly, and no result here should be described that way.**

---

## Project layout

```
fruit-fly-lab/
├── config.py                    dataset selection (FLY_DATASET) and paths; refuses substituted data
├── data/
│   ├── derived/                 built connectomes: FAFB, malecns/, merged/ (+ fill logs)
│   ├── external/                Hallem & Carlson 2006 ORN responses
│   └── metadata/                checksums, build manifests, calibrated dynamics and gains
├── brain/
│   ├── connectivity/            build_connectome.py (FAFB), build_malecns.py, malecns_annotations.py, merge.py
│   ├── neurons/                 registry.py (queries), labels.py
│   ├── neuron_models/           lif.py: the published LIF parameters
│   ├── sensory/                 modalities, encoders, retinotopy, olfaction, crossmap, orn_side, camera
│   ├── motor/                   descending.py: DN readout, proboscis (MN9 on male brains), escapes
│   ├── navigation/              E-PG compass, FC2 -> PFL3 goal steering
│   └── plasticity/              dopamine-gated mushroom-body learning
├── simulation/                  engine/ (lif_engine.py, session.py), stimuli/, outputs/
├── native/                      C and CUDA engines + verifiers
├── fly/                         body/ (fly_body.py, foraging_body.py), world/ (closed-loop foraging)
├── cognition/                   exam/ (the fly exam), calibrate_merged.py, calibrate_dynamics.py
├── cortex/                      v0.py (neocortex), topdown.py, obstacle_map.py, personality.py, llm_bench.py
├── robot/                       safety, avoid, lidar, battery, motion, hearing, head, face (+ face_page/)
├── sim/habitat_bridge/          Habitat server/client, run scripts, score_pets.py, compare_* tools
├── experiments/                 01-05 plus symmetry, VNC turn, antenna touch, satiety and sleep tests
├── visualization/               server.py + static/: the Python server build
├── web/                         static browser build (what deploys)
├── tools/                       web export, shared PRNG, cross-engine verifier, lab_server.sh
├── tests/                       pytest suite (+ test_web_worker.mjs)
├── results/                     dated result write-ups
├── requirements/                requirements.txt, -dev, -robot
├── ROADMAP.md, DEPLOY.md, DATA_SOURCES.md, BIOLOGICAL_ASSUMPTIONS.md, LICENSE
```

## Citing

If you publish anything based on this, cite the data and the model, not this
code:

- Dorkenwald, S. et al. (2024) *Nature* **634**, 124–138. doi:10.1038/s41586-024-07558-y
- Schlegel, P. et al. (2024) *Nature* **634**, 139–152. doi:10.1038/s41586-024-07686-5
- Shiu, P.K. et al. (2024) *Nature* **634**, 210–219. doi:10.1038/s41586-024-07763-9

FlyWire data are CC BY-NC-SA 4.0.
