"""
The fly brain driving a robot in the Habitat 3.0 home (follow-a-person task).

Runs in the repo's .venv and talks to sim/habitat_bridge/habitat_server.py
(running in the "habitat" conda env). Every control step (default 100 ms of
simulated time):

  Habitat (ground-truth person geometry through the robot's head camera)
    -> the SAME encoders as the real robot head (robot/head.py):
       ObjectEncoder -> LC10a (pursuit), HeadLoomingStimulus + LoomingEncoder
       -> LC4/LPLC2 (the person approaching), RestingOlfaction (every ORN at its
       spontaneous rate)
    -> the connectome (simulation.engine.session.Session, calibrated dynamics)
    -> descending neurons -> fly/body/foraging_body.ForagingBody (the same
       motor readout as the closed-loop fly world: DNg100 forward walking,
       DNa01/DNa02 steering, MDN backward, plus a spontaneous walking rhythm
       standing in for the nerve cord's -- FAFB has none, and the male CNS's
       leg motor neurons are not read yet)
    -> robot velocity, SCALED from fly to robot (documented approximation):
       speed x 0.025 (fly walking 12 mm/s -> robot 0.3 m/s), limited to
       -0.3 .. +0.5 m/s; turn rate x 0.5 (deg/s), limited to +-120 deg/s.
       The robot cannot fly: a takeoff (escape) becomes a fast dash in the
       direction the body chose.
    -> motor dynamics (robot/motion.MotorLag, FLY_MOTOR_TAU, default 0.3 s speed
       / 1.5 s turn), then the robot's layers in order: battery (emergency
       return, dock approach), the speed governor near people, the lidar safety
       layer (--lidar; --avoid steers around obstacles, robot/avoid.py), a flat
       battery stops it.

Bodies: --body spot (default) or rover (the Waveshare UGV's footprint).
--real-senses: only what the real rover has (no smell, no petting or treats,
no plant; the dock's contact is its taste).

Modes: brain (the above), body_only (ForagingBody with neural=False: the same
spontaneous walking rhythm, no brain - the "brain disconnected" control),
still (the robot does not move).

--home adds rewards (sim/habitat_bridge/home.py): a food bowl that smells and
tastes sweet, a bitter plant, and an owner who pets the pet and gives treats;
the resting-ORN input is replaced by the home's odours (spontaneous rates
included). --learning on/off runs the mushroom body (dopamine-gated KC->MBON
plasticity, brain/plasticity/mushroom_body.py) with plasticity on or frozen;
--weights PATH keeps its synaptic weights across episodes (a lifetime).

    .venv/bin/python -m sim.habitat_bridge.brain_client --mode brain --episodes 0 1 2
"""
from __future__ import annotations

import argparse
import atexit
import collections
import json
import math
import os
import time
from pathlib import Path

import numpy as np

SPEED_SCALE = 0.3 / 12.0        # m/s per fly mm/s
V_MIN, V_MAX = -0.3, 0.5        # m/s
TURN_SCALE = 0.5
W_MAX_DEG = 120.0
MOTOR_TAU_DEFAULT = "0.3,1.5"   # robot/motion.MotorLag (s): speed, turn rate
# the fly's song (male brains): the song channel (pIP10, brain/motor/
# descending.py), averaged over the control step, at >= SONG_ON starts a song
# bout, which ends below SONG_OFF; each bout is voiced by the robot (robot/voice.py "song")
# and told to the personality -- the fly brain decides when Milo "sings".
# A bout must last SONG_MIN_S (single 100 ms blips of pIP10 noise: 5.6 a day
# with the voice off) and start SONG_GAP_S after the last one ended (tuning
# seeds 2026-09-28: 0 bouts/day with the voice off, 3.8 with it on).
SONG_ON, SONG_OFF, SONG_MIN_S, SONG_GAP_S = 0.30, 0.15, 0.3, 5.0


class SimClock:
    """Simulated time for robot.head's motion estimates (it reads time.monotonic)."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        pass


class SimFeed:
    """Stands in for robot.head.HeadFeed: the same state() keys, from Habitat."""

    def __init__(self, clock: SimClock):
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        """A new episode: no carried-over looming or motion estimate."""
        self.s = {"ready": False}
        self._prev_half = None
        self._exp = 0.0

    def update(self, obs: dict, dt: float) -> None:
        vis = obs["visible"]
        half = obs["half"]
        exp = 0.0
        if vis and self._prev_half is not None and dt > 0:
            exp = (half - self._prev_half) / dt              # deg/s, + = approaching
        self._exp = 0.5 * self._exp + 0.5 * exp
        self._prev_half = half if vis else None
        self.s = {"ready": True, "stale": False, "moving": 0.0, "fps": 1.0 / dt,
                  "person_active": 1.0 if vis else 0.0, "person_stale": False,
                  "person_t": self.clock.t, "person_az": obs["az"], "person_el": obs["el"],
                  "person_half_deg": half, "person_score": 1.0,
                  "obj_active": 1.0 if vis else 0.0, "obj_az": obs["az"], "obj_el": obs["el"],
                  "obj_half_deg": half, "obj_exp_deg_s": max(self._exp, 0.0),
                  "pan_deg": 0.0, "tilt_deg": 0.0}

    def state(self) -> dict:
        return self.s


LIDAR_BEAMS = 90


def build_brain(seed: int, home=None, learning: bool | None = None, nav: bool = False,
                topdown: tuple | None = None, lidar: bool = False, lidar_senses: bool = True,
                body: str = "spot"):
    import robot.head as rh
    from brain.neurons.registry import load_connectome
    from brain.sensory.encoders import LoomingEncoder
    from brain.sensory.retinotopy import load_retinotopy
    from fly.body.foraging_body import ForagingBody
    from simulation.engine.session import Session
    clock = SimClock()
    rh.time = clock                     # robot.head's motion estimate runs on simulated time
    c = load_connectome()
    ses = Session(c, seed=seed)
    feed = SimFeed(clock)
    loom = LoomingEncoder(c, load_retinotopy(c))
    obj = rh.ObjectEncoder(c, feed)
    rest = rh.RestingOlfaction(c)
    parts = [(loom, rh.HeadLoomingStimulus(feed)), (obj, obj)]
    ses.lidar = None
    if lidar:
        # 2D lidar as looming (LC4/LPLC2) and antennal touch (robot/lidar.py)
        from robot.lidar import LidarLooming, LidarScan, LidarTouch, beam_angles, body_profile
        from sim.habitat_bridge.bodies import BODIES
        ang = beam_angles(LIDAR_BEAMS)
        scan = LidarScan(ang, body_profile(ang, BODIES[body]))   # the robot's body, seen from its lidar
        lloom, ltouch = LidarLooming(scan), LidarTouch(c, scan)
        if lidar_senses:
            parts += [(LoomingEncoder(c, load_retinotopy(c)), lloom), (ltouch, ltouch)]
        ses.lidar = (scan, lloom, ltouch)
        ses.lidar_limit = True
        ses.body_dims = (BODIES[body]["half_len"], BODIES[body]["half_wid"])
    if home is not None:
        from sim.habitat_bridge.home import HomeSenses
        senses = HomeSenses(c, home)
        parts.append((senses, senses))
    else:
        parts.append((rest, rest))
    if learning is not None:
        # dopamine-gated KC->MBON plasticity, as in the closed-loop fly world
        # (Session.set_world); learning=False keeps the same circuit frozen
        from brain.plasticity.mushroom_body import MushroomBody
        ses.mb = MushroomBody(c, ses.engine, plastic=True, kc_mbon_gain=8.0, dan_modulatory=True)
        ses.mb.learning = bool(learning)
        if nav:
            # learned odour valence gates a central-complex goal (heading = the
            # robot's odometry yaw; brain/navigation/valence_goal.py)
            from brain.navigation.compass import Compass, CompassDrive
            from brain.navigation.goal import GoalCircuit, GoalDrive
            from brain.navigation.valence_goal import ValenceGoal
            cx = Compass(c)
            cd = CompassDrive(cx)
            gd = GoalDrive(GoalCircuit(c, cx))
            ses.nav = ValenceGoal(ses.mb, gd, cd)
            parts += [(cd, cd), (gd, gd)]
            if home is not None:
                ses.nav_inputs = lambda: (0.5 * float(home.conc["L"].sum() + home.conc["R"].sum()), None, 0.0)
    ses.topdown = None
    if topdown:
        # the neocortex's channels into the fly brain (cortex/topdown.py)
        from cortex.topdown import TopDown
        ses.topdown = TopDown(c, obj, channels=topdown)
    ses.body = ForagingBody(neural=True, seed=seed, spontaneous_takeoff_per_s=0.0)
    return ses, feed, obj, clock, parts


def robot_command(body_state) -> tuple:
    v = float(np.clip(body_state.speed_mm_s * SPEED_SCALE, V_MIN, V_MAX))
    w_deg = float(np.clip(body_state.turn_rate_deg_s * TURN_SCALE, -W_MAX_DEG, W_MAX_DEG))
    return v, math.radians(w_deg)


def run_episode(conn, mode: str, episode: int, seconds: float, period_ms: float,
                seed: int, video_dir: str | None, brain=None, home=None, cortex=None,
                safe_speed: bool = True, person=None, personality=None, voice=None,
                checkpoint=None, checkpoint_s: float = 600.0, social=None, ears=None,
                dashboard=None, eye=None) -> dict:
    from fly.body.foraging_body import ForagingBody
    from robot.safety import ProximityGovernor
    governor = ProximityGovernor() if safe_speed else None
    obs = _call(conn, {"cmd": "reset", "episode": episode})
    n_env = int(round(period_ms / 1000.0 * 120.0))
    # the real rover runs for hours: keep its last hour of steps
    log = collections.deque(maxlen=int(3600e3 / period_ms)) if obs.get("rover") else []
    if mode == "brain":
        ses, feed, obj, clock, parts = brain
        ses.reset(seed=seed)                      # clears stimuli: attach the head's encoders again
        clock.t = 0.0
        feed.reset()
        obj._prev_person, obj._pmove = None, 0.0
        if getattr(ses, "nav", None) is not None and hasattr(ses.nav, "reset"):
            ses.nav.reset()
        for enc, stim in parts:
            ses.add_stimulus(enc, stim)
        if ses.topdown is not None:
            ses.topdown.attach(ses)
        ses.body = ForagingBody(neural=True, seed=seed, spontaneous_takeoff_per_s=0.0)
        if ses.mb is not None:
            ses.mb.reset_activity()               # new day: ongoing activity gone, memories kept
        if getattr(ses, "lidar", None) is not None:
            ses.lidar[0].t = None                 # new day: no lidar history (review 2026-09-26)
            ses.lidar[1].reset()
            if hasattr(ses.lidar[2], "reset"):
                ses.lidar[2].reset()                 # touch adaptation
        if getattr(ses, "cmon", None) is not None:
            ses.cmon.reset()
        if getattr(ses, "avoid", None) is not None:
            ses.avoid.reset()
        if getattr(ses, "unstick", None) is not None:
            ses.unstick.reset()
    if home is not None:
        home.reset(seed)
    if cortex is not None:
        cortex.reset(episode, home, conn)
        if (home is not None and home.battery is not None and hasattr(cortex, "know_place")
                and getattr(cortex, "days", 1) == 0 and not getattr(cortex, "_dock_known", False)):
            from sim.habitat_bridge.home import BOWL_XZ
            cortex.know_place(BOWL_XZ)              # born on its dock: it knows where it is
            cortex._dock_known = True
    emergency = {"active": False, "s": 0.0, "events": 0}
    # motor dynamics (robot/motion.py): FLY_MOTOR_TAU="tau_v,tau_w[,stages[,tau_w_fast]]" (seconds;
    # one value = the same tau for both; "0" = off). Default 0.3 s speed, 1.5 s
    # turning: on held-out seeds (results/habitat_real_world_2026-09-27) heading
    # reversals fell from ~185 to ~2 per active minute and aliveness 56 -> 82%,
    # with no significant change in bumps, braking or pinned time.
    motor, mtau = None, None
    if mode == "brain":
        from robot.motion import MotorLag
        mtau = [float(x) for x in os.environ.get("FLY_MOTOR_TAU", MOTOR_TAU_DEFAULT).split(",")]
        if len(mtau) == 1:
            mtau = mtau * 2
        if max(mtau[:2]) > 0:
            motor = MotorLag(mtau[0], mtau[1], int(mtau[2]) if len(mtau) > 2 else 1,
                             mtau[3] if len(mtau) > 3 and mtau[3] > 0 else None)
    if person is not None:
        person.reset()
    if personality is not None:
        personality.reset(episode)
    said = []                                     # (t, personality output) when new
    social_failed = []                            # camera commands that did not get through
    voice_log = collections.deque(maxlen=12)      # (t, who, text) for the dashboard
    speaking_until = 0.0                          # monotonic time Milo's queued speech ends
    prev_events = {"pets": 0, "treats": 0, "bowl": False, "seen_t": -1e9}
    song = {"bouts": 0, "s": 0.0, "on": False, "off_t": -1e9, "onset": False, "peak_hz": 0.0,
            "start_t": 0.0, "counted": False}
    if mode == "body_only":
        body = ForagingBody(neural=False, seed=seed, spontaneous_takeoff_per_s=0.0)
    t_sim, k, wall0 = 0.0, 0, time.time()
    fr_prev = None
    found_t = None
    bumps = []                                    # (t, person speed m/s, pet speed m/s, wanted company)
    last_v = 0.0
    while t_sim < seconds and not obs["over"]:
        dn = {}
        if home is not None:
            prob = ses.body.state.proboscis_extension if mode == "brain" else 0.0
            resting = mode == "brain" and ses.body.state.behaviour.startswith("resting")
            home.step(obs, period_ms / 1000.0, prob, speed=last_v, resting=resting)
        if mode == "brain":
            clock.t = t_sim
            feed.update(obs, period_ms / 1000.0)
            if getattr(ses, "lidar", None) is not None and obs.get("lidar"):
                # the pose when the scan was taken (the real rover's odometry at
                # the end of the lidar's revolution; Habitat's scan is instantaneous)
                ses.lidar[0].update(obs["lidar"], t_sim, pose=tuple(obs.get("lidar_pose") or obs["robot"]))
                ses.lidar[1].update()
            if getattr(ses, "nav", None) is not None:
                ses.body.state.heading_deg = float(obs["robot"][2]) % 360.0   # odometry yaw
            if personality is not None and cortex is not None:
                pout = personality.step(t_sim, _digest(cortex, ses, obs, home, fr_prev))
                cortex.set_personality(pout["intent"], pout["feedback"] if pout["new"] else 0)
                if pout["new"]:
                    said.append((round(t_sim, 1), {k: pout[k] for k in ("intent", "sound", "say", "mood", "feedback")}))
            if cortex is not None:
                if getattr(cortex, "obstacles", None) is not None and obs.get("lidar"):
                    cortex.observe_scan(tuple(obs["robot"]), obs["lidar"], ses.lidar[0].angles)
                cmd = cortex.act(obs, home, t_sim, fr_prev)
                if getattr(ses, "avoid", None) is not None:
                    ses.avoid.last = {"active": False}   # this step's state only (review 2026-09-27)
                if getattr(ses, "unstick", None) is not None:
                    ses.unstick.pulling = False           # this step's state only
                    if getattr(cortex, "resting", False):
                        ses.unstick.cancel()              # a nap is not being stuck
                if (getattr(ses, "avoid", None) is not None and ses.lidar[0].t is not None
                        and not getattr(cortex, "resting", False)):
                    # steer around obstacles before reaching them (robot/avoid.py)
                    from robot.safety import person_distance, scan_points_robot
                    td = ta = None
                    if ((cmd.get("goal_is_person") or (getattr(cortex, "goal", None) and cortex.goal[1] == "owner"))
                            and obs["visible"]):
                        td = person_distance(obs["half"])      # the person is the target, not an obstacle
                    if cmd.get("attend_explicit") and obs["visible"]:
                        ta = person_distance(obs["half"])      # the orienting reflex points at the person
                    pts = scan_points_robot(ses.lidar[0].ranges, ses.lidar[0].angles)
                    cmd = ses.avoid.adjust(cmd, float(obs["robot"][2]) % 360.0, pts, last_v, td, attend_dist=ta)
                un = getattr(ses, "unstick", None)
                if (un is not None and ses.lidar[0].t is not None
                        and not getattr(cortex, "resting", False)):
                    # stuck near something and not moving: pull toward the most
                    # open way (robot/avoid.Unstick); the fly brain turns. Not
                    # while it means to stand still (review 2026-09-28). With
                    # lidar steering or, FLY_UNSTICK_LIDAR, lidar alone.
                    from cortex.topdown import goal_azimuth, reflex_channel
                    from robot.safety import person_distance, scan_points_robot
                    pts = scan_points_robot(ses.lidar[0].ranges, ses.lidar[0].angles)
                    h = float(obs["robot"][2]) % 360.0
                    d_person = person_distance(obs["half"]) if obs["visible"] else None
                    near_person = d_person is not None and d_person < 1.0
                    still = bool((home is not None and home.taste == "sweet")
                                 or (home is not None and home.battery is not None and home.battery.meal)
                                 or getattr(cortex, "manner", None) == "yield"
                                 or (near_person and (cmd.get("goal_is_person")
                                                      or (getattr(cortex, "goal", None) and cortex.goal[1] == "owner"))))
                    ug = un.step(t_sim, obs["robot"][:2], float(ses.lidar[0].clearance().min()), pts,
                                 period_ms / 1000.0, h, may_trigger=not still)
                    if ug is not None:
                        uaz = goal_azimuth(ug, h)
                        # the pull replaces the goal AND any attention (the orienting
                        # reflex's, or avoid's bend of it): nothing may pull elsewhere
                        cmd = {k: v_ for k, v_ in cmd.items() if k not in ("attend_az", "attend_gain", "attend_explicit")}
                        cmd.update(goal_deg=ug, goal_gain=1.0, goal_is_person=False)
                        if reflex_channel("unstick") == "attend":     # not on male brains (cortex/topdown.py)
                            cmd.update(attend_az=uaz, attend_gain=1.0, attend_explicit=True)
                        else:
                            cmd.update(attend_az=uaz, attend_gain=0.0, attend_explicit=True)   # LC10a off
                        un.pulling = True
                        if getattr(ses, "avoid", None) is not None:
                            # the pivot reflex must turn the same way (review 2026-09-28)
                            ses.avoid.last.update(active=True, chosen=round(uaz, 1), source="unstick")
                ses.topdown.apply(float(obs["robot"][2]) % 360.0, cmd)
            fr = fr_prev = ses.advance(period_ms)[-1]
            # the song channel over the whole step (every 1 ms block's 50 ms
            # window, Session.STEP_MEAN_KEYS): pIP10 is one cell per side, and a
            # single window moves in steps of 1 spike (reviews 2026-09-28)
            sm = fr.get("step_means") or {}
            fr["song_step"] = float(sm.get("song", fr["channels"].get("song", 0.0)))
            fr["hz_pIP10_step"] = float(sm.get("hz_pIP10", fr["channels"].get("hz_pIP10", 0.0)))
            v, w = robot_command(ses.body.state)
            raw = (v, w)                          # what the brain asked for, before the robot's layers
            if motor is not None and not str(ses.body.state.behaviour).startswith("escape"):
                v, w = motor(v, w, period_ms / 1000.0)      # a startle escape is not smoothed
            dn = {kk: round(vv, 1) for kk, vv in fr["dn_rates"].items()
                  if kk.startswith(("DNa01", "DNa02", "DNg100", "DNp09", "DNp01", "DNp02", "DNp04", "DNp11", "MDN", "DNge078"))}
            tb = round(fr["channels"].get("turn_bias", 0.0), 3)
        elif mode == "body_only":
            for i in range(int(period_ms)):
                body.update(1.0, {}, t_sim * 1000 + i)
            v, w = robot_command(body.state)
            tb = 0.0
        else:
            v, w, tb = 0.0, 0.0, 0.0
        if mode != "brain":
            raw = (v, w)                          # what the body asked for, before the robot's layers
        # robot layers, in order: battery (emergency return / docking), then the
        # safety limits, which nothing above may override (review 2026-09-26:
        # the emergency return used to bypass them)
        if home is not None and home.battery is not None:
            v, w = _emergency_return(conn, obs, home, emergency, v, w, period_ms / 1000.0)
            if not emergency["active"]:
                seeking = bool(cortex is not None and getattr(cortex, "goal", None) and cortex.goal[1] == "food")
                v = _dock_approach_limit(obs, home, v, seeking_food=seeking)
        if governor is not None:
            governor.observe(obs["visible"], obs["half"], t_sim)
            v = governor.limit(v, t_sim, period_ms / 1000.0)
        v_pre = v                                 # before the lidar safety layer: how often it must brake
        if (mode == "brain" and getattr(ses, "lidar", None) is not None and ses.lidar[0].t is not None
                and getattr(ses, "lidar_limit", True)):
            if getattr(ses, "lidar_ttc", False):
                from robot.safety import scan_points_robot
                if getattr(ses, "cmon", None) is None:
                    from robot.safety import CollisionMonitor
                    ses.cmon = CollisionMonitor(*ses.body_dims)
                pts = scan_points_robot(ses.lidar[0].ranges, ses.lidar[0].angles)
                v, w = ses.cmon(pts, v, w, period_ms / 1000.0)              # footprint TTC + recovery
            else:
                from robot.safety import obstacle_limit
                v = obstacle_limit(ses.lidar[0].clearance(), ses.lidar[0].angles, v)   # forward cone
                if getattr(ses, "avoid", None) is not None and not emergency["active"]:
                    from robot.avoid import pivot
                    w, pivoted = pivot(ses.avoid.last, v_pre, v, w)       # pinned: turn in place to the open side
                    ses.avoid.last["pivot"] = pivoted
        v_lidar = v                               # after the lidar layer (the "brake" it applied)
        if home is not None and home.battery is not None and home.battery.flat:
            v, w = 0.0, 0.0                       # a flat battery: the robot stops
        from robot.safety import finite_command
        v, w = finite_command(v, w)                # a NaN would pass every limit above
        if motor is not None:
            # the lag continues from what the wheels actually do: after a brake,
            # the emergency return or a pivot it starts up smoothly again instead
            # of jumping to the brain's command (review 2026-09-27: anti-windup)
            motor.sync(v, w)
        last_v = v
        prev_obs = obs
        obs = _call(conn, {"cmd": "step", "v": v, "w": w, "n": n_env,
                           "frame": bool(video_dir) and k % 5 == 0})
        bumped = obs.get("collisions", 0) > prev_obs.get("collisions", 0)
        if eye is not None:
            eye.put_lidar(obs.get("lidar_fine"))  # the eye draws the lidar's obstacles around the camera's view
        if person is not None:
            heard = person.step(t_sim, period_ms / 1000.0, obs, prev_obs, bumped, v)
            if heard and personality is not None:
                personality.event(t_sim, f'your person said: "{heard}"', urge=True)
        if mode == "brain" and fr_prev is not None:
            _song(song, fr_prev, t_sim, period_ms / 1000.0)
            if song["onset"] and personality is not None:
                personality.event(t_sim, "your fly brain's courtship-song command (pIP10) switched on "
                                         f"({song['peak_hz']:.0f} Hz): you are singing", urge=True)
        if personality is not None and mode == "brain":
            _events(personality, t_sim, obs, home, fr_prev, prev_events)
        heard_replies = []
        if ears is not None:
            # words heard on the robot (robot/speech.py): an answer to Milo's
            # open question, or, as in Habitat, something its person said.
            # Heard words are shown live (dashboard) but not saved in the run's log.
            now_m, now_w = time.monotonic(), time.time()
            for h in ears.heard():
                text = h["heard"][:200]
                voice_log.append({"t": round(t_sim, 1), "who": "heard", "text": text})
                # when the utterance began, on the monotonic clock
                began = now_m - (now_w - (h.get("t", now_w) - h.get("ms", 0.0) / 1000.0 - h.get("dur_s", 0.0)))
                if social is not None and social.pending is not None:
                    # an answer only if it began after Milo finished asking (not a "yeah"
                    # said to someone else before the question)
                    if began >= max(social.pending["t"], speaking_until) - 0.2:
                        heard_replies.append({"id": social.pending["id"], "text": text})
                    continue                      # while asking, nothing goes to the personality
                if personality is not None:
                    who = ", ".join(sorted({f["who"] for f in obs.get("faces") or []
                                            if f.get("facing") and f.get("who") not in (None, "?")}))
                    personality.event(t_sim, f'{who or "your person"} said: "{text}"', urge=True)
        soc = None
        if social is not None and mode == "brain":
            # faces: greet the people Milo knows, ask new ones their name (robot/people.py);
            # on the wall clock, which never goes back (episodes restart t_sim)
            cam_ev = list(obs.get("camera_events") or []) + social_failed
            social_failed.clear()
            soc = social.step(time.monotonic(), obs.get("faces") or [],
                              heard_replies, cam_ev)
            for c in soc["commands"]:
                if not _call(conn, {"cmd": "camera", "send": c}).get("ok") and c.get("cmd") == "enroll":
                    social_failed.append({"event": "enroll_failed", "track": c.get("track"),
                                          "name": c.get("name"), "why": "camera"})
            for e in soc["events"]:
                if personality is not None:
                    personality.event(t_sim, e, urge=True)
            if soc["say"]:
                said.append((round(t_sim, 1), {"social": soc["say"]}))
        if voice is not None and mode == "brain":
            for line in _speak(voice, personality, soc, song["onset"]):
                voice_log.append({"t": round(t_sim, 1), "who": "milo", "text": line})
        if voice is not None:
            # the voice reports how long its queued speech will still play: the
            # ears do not listen meanwhile (not to hear Milo's own voice)
            for d in voice.speaking():
                speaking_until = max(speaking_until, time.monotonic() + d)
                if ears is not None:
                    ears.mute(d + 0.6)
        if social is not None:
            # questions only while Milo can say them and hear the answer
            social.can_ask = bool(social.reader is not None and ears is not None and ears.alive
                                  and voice is not None and voice.alive)
        if dashboard is not None and mode == "brain" and k % 2 == 0:
            pub, jpeg_fn, bm = dashboard            # 5 Hz, in the background (never waits)
            try:
                tel = _telemetry(t_sim, obs, ses, fr_prev, cortex, personality, social, v, w, voice_log)
            except Exception as ex:               # display only: never the end of the run
                tel = {"t": round(t_sim, 1), "error": str(ex)[:200]}
            if eye is not None:
                tel["eye"] = eye.status()
            pub.publish(tel, jpeg_fn, lambda: _activity(bm, ses))
        if bumped:
            hs = math.hypot(obs["human"][0] - prev_obs["human"][0],
                            obs["human"][1] - prev_obs["human"][1]) / (period_ms / 1000.0)
            wanted = None if cortex is None or not hasattr(cortex, "social") else bool(cortex.social >= 0.35)
            bumps.append((round(t_sim, 1), round(hs, 2), round(v, 3), wanted))
        t_sim += period_ms / 1000.0
        k += 1
        if checkpoint is not None and k % max(1, int(round(checkpoint_s * 1000.0 / period_ms))) == 0:
            checkpoint()                          # the real rover: learning survives a crash or a stop
        if found_t is None and obs["visible"] and obs["dist"] < 2.0:
            found_t = t_sim
        log.append({"t": round(t_sim, 2), "dist": round(obs["dist"], 3), "visible": obs["visible"],
                    "robot": [round(x, 4) for x in obs["robot"]], "human": [round(x, 2) for x in obs["human"]],
                    "az": round(obs["az"], 1), "v": round(v, 3), "w_deg": round(math.degrees(w), 1),
                    "turn_bias": tb, "dn": dn,
                    "lc10a_hz": (obj.last.get("drive_hz") if mode == "brain" else None)})
        if mode == "brain" and getattr(ses, "lidar", None) is not None:
            L = ses.lidar
            log[-1]["lidar"] = {"loom": L[1].last.get("active", False), "loom_az": round(L[1].last.get("azimuth_deg", 0.0)),
                                "touch": L[2].last, "min_clear": round(float(L[0].clearance().min()), 2),
                                "raw": [round(raw[0], 3), round(math.degrees(raw[1]), 1)],
                                "recoveries": getattr(getattr(ses, "cmon", None), "recoveries", None),
                                "brake": round(max(v_pre - v_lidar, 0.0), 3)}
            if getattr(ses, "avoid", None) is not None:
                log[-1]["lidar"]["avoid"] = dict(ses.avoid.last)       # a copy: later steps must not rewrite it
            if getattr(ses, "unstick", None) is not None:
                # pulling THIS step (t_sim has already advanced: review 2026-09-28)
                log[-1]["lidar"]["unstick"] = bool(getattr(ses.unstick, "pulling", False))
        if home is not None:
            log[-1].update({"odour": [round(float(home.conc["L"].sum()), 3), round(float(home.conc["R"].sum()), 3)],
                            "taste": home.taste, "pet": home.petting, "treat": home.treating,
                            "proboscis": round(float(ses.body.state.proboscis_extension), 2) if mode == "brain" else 0.0,
                            "song": round(float(fr_prev.get("song_step", 0.0)), 2) if (mode == "brain" and fr_prev) else 0.0,
                            "groom": round(float(fr["channels"].get("groom", 0.0)), 2) if mode == "brain" else 0.0,
                            "behaviour": ses.body.state.behaviour if mode == "brain" else ""})
        if cortex is not None:
            log[-1]["cortex"] = {**ses.topdown.last, **cortex.log_state()}
    video = None
    if video_dir:
        video = _call(conn, {"cmd": "video", "path": str(Path(video_dir) / f"{mode}_ep{episode}.mp4")})["path"]
    d = np.array([r["dist"] for r in log]) if log else np.zeros(1)
    vis = np.array([r["visible"] for r in log]) if log else np.zeros(1)
    res = {"mode": mode, "episode": episode, "sim_s": round(t_sim, 1), "wall_s": round(time.time() - wall0, 1),
           "visible_frac": round(float(vis.mean()), 3), "mean_dist": round(float(d.mean()), 2),
           "in_follow_band_frac": round(float(((d > 0.8) & (d < 3.0) & vis.astype(bool)).mean()), 3),
           "found_s": found_t, "collisions": obs.get("collisions", 0),
           "scene_contacts": obs.get("scene_contacts"), "scene_bumps": obs.get("scene_bumps"),
           "blocked_s": obs.get("blocked_s"), "motor_tau": mtau if motor is not None else None, "habitat_stats": obs["stats"],
           "video": video, "log": list(log),
           "song": {"bouts": song["bouts"], "s": round(song["s"], 1)},
           "unstick": ({"events": ses.unstick.events, "s": round(ses.unstick.active_s, 1)}
                       if mode == "brain" and getattr(ses, "unstick", None) is not None else None)}
    if home is not None:
        res["home"] = dict(home.stats)
        if home.battery is not None:
            res["home"]["battery"] = home.battery.summary()
            res["home"]["emergency_return"] = {"events": emergency["events"], "s": round(emergency["s"], 1)}
        res["home"]["mean_bowl_dist"] = round(float(np.mean([math.hypot(r["robot"][0] - home.sources[0][1][0],
                                                                         r["robot"][1] - home.sources[0][1][1])
                                                              for r in log])), 2) if log else None
        res["events"] = home.events
    if cortex is not None:
        res["cortex"] = cortex.summary()
    if person is not None:
        res["speech"] = person.summary()
    if social is not None:
        res["social"] = {"stats": dict(social.stats), "met": sorted(social.met)}
    if ears is not None:
        res["ears"] = {"utterances": ears.stats["utterances"], "alive": ears.alive}
    if personality is not None:
        personality.wait(30.0)
        plog = personality.log
        if ears is not None:                       # its prompts quote what was heard: not saved
            plog = [{k: v for k, v in e.items() if k != "asked"} for e in plog]
        res["personality"] = {"said": said, "stats": dict(personality.stats), "log": plog}
        personality.save()
    # bumps into the person, by kind (ground truth; for evaluation only)
    res["bumps"] = {"n": len(bumps), "into_walking": sum(b[1] > 0.25 for b in bumps),
                    "fast": sum(b[2] > 0.15 for b in bumps),
                    "unwanted": sum(b[3] is False for b in bumps), "list": bumps,
                    "speed_limited_s": round(governor.limited_s, 1) if governor is not None else None}
    if mode == "brain" and brain is not None and getattr(brain[0], "nav", None) is not None:
        res["navigation"] = brain[0].nav.state()
    if mode == "brain" and brain is not None and brain[0].mb is not None:
        res["learning"] = brain[0].mb.summary()
        res["learning"]["enabled"] = bool(brain[0].mb.learning)
    return res


EMERGENCY_SOC, EMERGENCY_DONE = 0.10, 0.30


DOCK_SLOW_M, DOCK_SLOW_V, DOCK_PIVOT_DEG = 1.2, 0.15, 60.0


def _dock_approach_limit(obs, home, v, seeking_food: bool = False):
    """Robot-level slow final approach to the dock (any robot docks slowly):
    within DOCK_SLOW_M of the dock while going to it (a meal, or the neocortex's
    goal is food), forward speed
    is capped so the brain has time to turn and line up, and with the dock
    more than DOCK_PIVOT_DEG to the side it does not drive forward at all (a
    smoke test showed the pet orbiting the dock at 1 m). The dock position is
    the robot's own knowledge (it starts on its dock), not a sensed cue."""
    from sim.habitat_bridge.home import BOWL_XZ
    b = home.battery
    if b is None or not (b.meal or seeking_food):
        # only while the pet is going to its dock (a meal, or the neocortex's
        # goal is food); otherwise it can still get out of the way (review
        # 2026-09-26: "hunger > 0.5" froze pets beside the dock)
        return v
    x, z, yaw = obs["robot"]
    if math.hypot(x - BOWL_XZ[0], z - BOWL_XZ[1]) < DOCK_SLOW_M:
        from cortex.topdown import bearing_deg
        off = ((bearing_deg((x, z), BOWL_XZ) - yaw + 180.0) % 360.0) - 180.0
        if abs(off) > DOCK_PIVOT_DEG:
            return min(v, 0.0)        # dock to the side: turn on the spot (the brain steers), don't orbit
        return min(v, DOCK_SLOW_V)
    return v


def _emergency_return(conn, obs, home, em, v, w, dt):
    """Robot-level safety net, not the pet's choice: below EMERGENCY_SOC the
    navigation layer drives to the dock and docks (as a robot vacuum returns to
    base; on the real robot, Nav2 docking). The brain keeps running but its
    motor output is ignored until the charge is back to EMERGENCY_DONE. Every
    return is recorded as a FAILURE of the pet's own behaviour."""
    from sim.habitat_bridge.home import BOWL_XZ
    b = home.battery
    if not em["active"] and b.soc < EMERGENCY_SOC and not b.flat:
        em["active"] = True
        em["events"] += 1
    if em["active"] and b.soc >= EMERGENCY_DONE:
        em["active"] = False
    home.docked_by_nav = False
    if not em["active"]:
        return v, w
    em["s"] += dt
    x, z, yaw = obs["robot"]
    if math.hypot(x - BOWL_XZ[0], z - BOWL_XZ[1]) < home.eat_r:
        home.docked_by_nav = True                 # on the contacts: charges without "eating"
        return 0.0, 0.0
    p = _call(conn, {"cmd": "path", "goal": list(BOWL_XZ), "ahead": 0.6})
    from cortex.topdown import bearing_deg
    err = ((bearing_deg((x, z), p["waypoint"]) - yaw + 180.0) % 360.0) - 180.0   # + = turn left (CCW)
    w = math.radians(max(-90.0, min(90.0, 2.0 * err)))
    v = 0.3 * max(0.0, math.cos(math.radians(err)))
    return v, w


def brain_readout(fr, cortex=None) -> str:
    """The fly brain's live state in words, for the personality (the LLM as
    the language centre: it describes what the connectome is doing and must
    not invent it). From the descending-neuron channels of the last window."""
    if fr is None:
        return "no readout yet"
    ch = fr["channels"]
    parts = []
    esc = max(ch.get("escape_takeoff", 0.0), ch.get("escape_long_mode", 0.0))
    parts.append("escape neurons FIRING (startled)" if esc > 0.5 else "escape neurons quiet")
    hz = fr.get("hz_pIP10_step", ch.get("hz_pIP10"))
    if "hz_pIP10" in ch:
        # the step mean, as the song bouts use (one 50 ms window flickers)
        singing = fr.get("song_step", ch.get("song", 0.0)) >= SONG_ON
        parts.append(f"courtship-song command pIP10 {hz:.0f} Hz" + (" (singing)" if singing else ""))
    tb = ch.get("turn_bias", 0.0)
    parts.append("steering neurons pull " + ("right" if tb > 0.05 else "left" if tb < -0.05 else "straight"))
    fw = ch.get("forward_walk", 0.0)
    parts.append("walking drive " + ("strong" if fw > 0.4 else "weak" if fw > 0.1 else "off"))
    rpe = float(getattr(cortex, "recent_rpe", getattr(cortex, "last_rpe", 0.0)) or 0.0) if cortex is not None else 0.0
    if rpe > 0:
        parts.append("neocortex: better than expected (reward dopamine to the fly's learning centre)")
    elif rpe < 0:
        parts.append("neocortex: worse than expected (punishment dopamine to the fly's learning centre)")
    return "; ".join(parts)


def _digest(cortex, ses, obs, home=None, fr=None) -> dict:
    """What the personality layer is told each call (cortex/personality.py)."""
    from robot.safety import person_distance
    d = person_distance(obs["half"]) if obs["visible"] else None
    side = "ahead" if abs(obs["az"]) < 20 else ("to the right" if obs["az"] > 0 else "to the left")
    goal = cortex.goal[1] if getattr(cortex, "goal", None) else "nothing in particular"
    # what the neocortex remembers: the nearest place where it has tasted food
    x, z = obs["robot"][0], obs["robot"][1]
    food = [cortex.map.food_point(c) for c, n in cortex.map.nodes.items() if n["food"] > 0.1]
    fd = min((math.hypot(fx - x, fz - z) for fx, fz in food), default=None)
    # who it is, when the camera knows their face (robot/people.py)
    known = ", ".join(sorted({f["who"] for f in obs.get("faces") or [] if f.get("who") not in (None, "?")}))
    return {"hunger": cortex.hunger, "social": cortex.social,
            # sleepiness only means something to a pet that naps (review 2026-09-26)
            "sleepy": cortex.sleepy if getattr(cortex, "naps", False) else None,
            "battery": getattr(getattr(home, "battery", None), "soc", None) if home is not None else None,
            "behaviour": ses.body.state.behaviour,
            "person": ((f"{known}, " if known else "") + f"{d:.1f} m {side}") if d is not None else "not in view",
            "food": f"you remember food {fd:.1f} m away" if fd is not None else "you do not know where food is yet",
            "doing": f"heading for {goal}" + (f"; manners: {cortex.manner}" if getattr(cortex, "manner", None) else ""),
            "brain": brain_readout(fr, cortex)}


def _events(pers, t, obs, home, fr, prev):
    """Things worth a reaction, for the personality layer."""
    if home is not None:
        if home.stats["pets"] > prev["pets"]:
            pers.event(t, "your person petted you", urge=True)
        if home.stats["treats"] > prev["treats"]:
            pers.event(t, "your person gave you a treat", urge=True)
        prev["pets"], prev["treats"] = home.stats["pets"], home.stats["treats"]
        at_bowl = home.taste == "sweet"
        if at_bowl and not prev["bowl"]:
            pers.event(t, "you found food and tasted it")
        prev["bowl"] = at_bowl
    if obs["visible"]:
        if t - prev["seen_t"] > 20.0:
            pers.event(t, "your person came into view")
        prev["seen_t"] = t
    if fr is not None and max(fr["channels"].get("escape_takeoff", 0.0), fr["channels"].get("escape_long_mode", 0.0)) > 0.5:
        if t - prev.get("startle_t", -1e9) > 5.0:
            pers.event(t, "something startled you (your escape neurons fired)", urge=True)
        prev["startle_t"] = t


def _song(song, fr, t, dt):
    """Song bouts from the song channel: on at SONG_ON, off below SONG_OFF; a
    bout counts when it has lasted SONG_MIN_S. One that began SONG_GAP_S after
    the previous counted bout ended is a new bout (song["onset"] is set, once);
    one that resumed sooner continues that bout -- its time counts, no new
    onset (review 2026-09-28: it was never counted). Blips count neither way."""
    act = float(fr.get("song_step", fr["channels"].get("song", 0.0)))
    song["onset"] = False
    if act >= SONG_ON or (song["on"] and act >= SONG_OFF):
        if not song["on"]:
            song["on"], song["start_t"], song["counted"] = True, t, False
            song["resumed"] = t - song["off_t"] < SONG_GAP_S
            song["peak_hz"] = 0.0
        if song["counted"]:
            song["s"] += dt                              # (blips are not song)
        song["peak_hz"] = max(song["peak_hz"], float(fr.get("hz_pIP10_step", fr["channels"].get("hz_pIP10", 0.0))))
        if not song["counted"] and t - song["start_t"] + dt >= SONG_MIN_S - 1e-9:
            song["counted"] = True
            song["s"] += t - song["start_t"] + dt
            if not song.get("resumed"):
                song["onset"] = True
                song["bouts"] += 1
    elif song["on"]:
        song["on"] = False
        if song["counted"]:
            song["off_t"] = t


def _speak(voice, personality, social, song=False):
    """Milo's words and sounds from the rover's speaker (robot/voice.py): its
    own greeting or question (robot/people.Social) first, then the
    personality's words (queued, in order); the fly brain's courtship-song
    command (pIP10) voiced as its song."""
    fresh = personality.take_fresh() if personality is not None else None
    lines = [x for x in ((social or {}).get("say"), fresh.get("say") if fresh else None) if x]
    for line in lines:
        voice.say(line)
    snd = "song" if song else (fresh.get("sound") if fresh else None)
    if snd and snd != "none":
        voice.sound(snd)
    return lines


def _activity(bm, ses):
    """The brain's activity for the dashboard (robot/brainmap.py), in its
    background thread: the systems' rates and the 3D view's levels (the
    neurons' positions are sent once, DashboardPublisher.put_static)."""
    if bm is None:
        return {}, {}
    blobs = {}
    systems, levels = bm.update(ses.engine.spike_counts)
    if levels is not None:
        blobs["/activity"] = levels
        blobs["/flow"] = bm.flow or b"\0" * 8      # the wiring the activity travels along
    return ({"activity": systems} if systems else {}), blobs


def _telemetry(t, obs, ses, fr, cortex, personality, social, v, w, voice_log) -> dict:
    """What the dashboard shows (robot/dashboard.py): small, JSON-able."""
    ch = (fr or {}).get("channels", {})
    tel = {"t": round(t, 1), "pose": [round(float(x), 3) for x in obs["robot"]],
           "cmd": [round(float(v), 3), round(math.degrees(w), 1)], "battery_v": obs.get("battery_v"),
           "stale": obs.get("stale") or [], "body": ses.body.state.behaviour,
           "brain": {k: round(float(ch[k]), 3) for k in ("forward_walk", "turn_bias", "escape_takeoff",
                                                           "escape_long_mode", "groom", "backward_walk", "song")
                     if k in ch},
           "readout": brain_readout(fr, cortex) if fr is not None else None,
           "person": {"visible": bool(obs["visible"]), "az": round(float(obs["az"]), 1),
                      "dist": None if obs["dist"] != obs["dist"] else round(float(obs["dist"]), 2)},
           "faces": [{"who": f.get("who"), "facing": bool(f.get("facing")), "facing_s": f.get("facing_s", 0.0)}
                     for f in obs.get("faces") or []],
           "voice": list(voice_log)}
    st = obs.get("stats") or {}
    if st:
        tel["loop"] = {"mean_ms": st.get("compute_ms_mean"), "p95_ms": st.get("compute_ms_p95"),
                       "late": int(st.get("overruns", 0))}
        tel["camera_fps"] = st.get("camera_fps")
    if cortex is not None and hasattr(cortex, "hunger"):
        g = getattr(cortex, "goal", None)
        tel["cortex"] = {"hunger": round(float(cortex.hunger), 2), "social": round(float(getattr(cortex, "social", 0.0)), 2),
                         "sleepy": round(float(cortex.sleepy), 2) if getattr(cortex, "naps", False) else None,
                         "goal": g[1] if g else None, "intent": getattr(cortex, "intent", None),
                         "manner": getattr(cortex, "manner", None),
                         "mood": personality.current["mood"] if personality is not None else None}
    if getattr(ses, "lidar", None) is not None:
        scan = ses.lidar[0]
        fine = obs.get("lidar_fine")
        if fine:                                  # the D500's 1-degree scan (display only)
            n = len(fine)
            tel["lidar"] = {"angles": [round(-180.0 + 360.0 * k / n, 1) for k in range(n)], "ranges": fine}
        else:
            tel["lidar"] = {"angles": [round(float(a), 1) for a in scan.angles],
                            "ranges": [round(float(r), 2) if r == r and r < 8.0 else 8.0 for r in scan.ranges]}
        tel["lidar"]["moves"] = obs.get("lidar_moves", True)
    mb = getattr(ses, "mb", None)
    tel["learning"] = {"depressed": (mb.summary() or {}).get("depressed_synapses") if mb is not None else None,
                       "people": len(social.met) if social is not None else None}
    return tel


def _call(conn, msg):
    conn.send(msg)
    return conn.recv()


def main():
    from multiprocessing.connection import Client
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--mode", choices=("brain", "body_only", "still"), default="brain")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--period-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", default="simulation/outputs/habitat")
    ap.add_argument("--home", action="store_true", help="food bowl, bitter plant, owner rewards")
    ap.add_argument("--battery", type=float, default=None, metavar="SOC",
                    help="battery mode: the bowl is a charging dock and hunger is the charge "
                         "(robot/battery.py); SOC = starting charge, kept across days")
    ap.add_argument("--dock-reflex", action="store_true",
                    help="battery-mode control: sense the dock at full strength whatever the charge")
    ap.add_argument("--pet-rate", type=float, default=0.15)
    ap.add_argument("--treat-rate", type=float, default=0.05)
    ap.add_argument("--learning", choices=("on", "off"), default=None,
                    help="mushroom-body plasticity on or frozen (default: no mushroom-body module)")
    ap.add_argument("--weights", default=None, help="keep mushroom-body weights here across episodes")
    ap.add_argument("--nav", action="store_true", help="learned valence gates a central-complex goal")
    ap.add_argument("--topdown", default=None, help="save the house's top-down map here (npz)")
    ap.add_argument("--cortex", choices=("none", "oracle", "v0", "v0_amnesic", "v0_manners", "pet"), default="none",
                    help="neocortex driving the fly brain top-down (cortex/)")
    ap.add_argument("--channels", default="goal,attend,dopamine,arousal",
                    help="top-down channels the cortex may use (cortex/topdown.py)")
    ap.add_argument("--cortex-state", default=None, help="keep the cortex's memory here across episodes")
    ap.add_argument("--speech", action="store_true",
                    help="a scripted person calls, praises and scolds the pet (sim/habitat_bridge/speech.py)")
    ap.add_argument("--personality", default=None, metavar="URL",
                    help="LLM personality layer (cortex/personality.py), e.g. http://127.0.0.1:1234/v1/chat/completions (PAIR)")
    ap.add_argument("--llm-model", default="gemma-4-e4b-it-mlx")
    ap.add_argument("--pet-name", default="Milo")
    ap.add_argument("--dashboard", type=int, default=None, metavar="PORT",
                    help="--rover: serve Milo's read-only dashboard on this port (robot/dashboard.py), e.g. 8080")
    ap.add_argument("--eye", action="store_true",
                    help="--rover: what the camera and the lidar see goes through the fly's optic lobes "
                         "(robot/eye.py: the flyvis eye model, on the GPU, in its own process)")
    ap.add_argument("--voice", action="store_true",
                    help="--rover: Milo speaks from the rover's speaker (robot/voice.py): the personality's "
                         "words and sounds, greetings and questions")
    ap.add_argument("--speaker", default=None,
                    help="--voice: the ALSA playback device (default: $FLY_SPEAKER, else the first USB speaker)")
    ap.add_argument("--lidar", action="store_true",
                    help="simulated 2D lidar -> looming (LC4/LPLC2) and antennal touch (robot/lidar.py) "
                         "and the robot's obstacle speed limit (robot/safety.py)")
    ap.add_argument("--lidar-use", choices=("both", "senses", "limit", "ttc"), default="both",
                    help="ablation: lidar feeds the fly's senses, the speed limit, or both")
    ap.add_argument("--real-senses", action="store_true",
                    help="only what the real rover can sense: camera, lidar, odometry, battery and the dock's "
                         "charging contacts (as taste); no odour, no owner petting or treats, no bitter plant")
    ap.add_argument("--avoid", action="store_true",
                    help="with --lidar: steer around obstacles through the goal / pursuit channels (robot/avoid.py)")
    ap.add_argument("--route", action="store_true",
                    help="with --lidar: the neocortex remembers obstacles and plans routes around them "
                         "(cortex/obstacle_map.py)")
    ap.add_argument("--body", choices=("spot", "rover"), default="spot",
                    help="the robot's footprint; must match the server's --body (sim/habitat_bridge/bodies.py)")
    ap.add_argument("--rover", choices=("fake", "hw"), default=None,
                    help="drive the real rover in real time instead of Habitat (robot/rover_world.py): "
                         "hw = the Waveshare base and D500 lidar, fake = a simulated room, base and person "
                         "with the real brain and timing")
    ap.add_argument("--ugv-port", default="/dev/ttyTHS1", help="--rover hw: the base's serial port")
    ap.add_argument("--lidar-port", default=None, help="--rover hw: the D500's serial port (required); --rover fake: the real D500 "
                         "with the simulated base (a bench test)")
    ap.add_argument("--hfov", type=float, default=53.0, help="--rover: the head camera's horizontal field of view")
    ap.add_argument("--listen", action="store_true",
                    help="--rover: speech recognition on the robot (robot/speech.py): words reach the "
                         "personality and answer Milo's questions")
    ap.add_argument("--listen-wav", default=None, help="--listen: a 16 kHz WAV instead of the microphone (tests)")
    ap.add_argument("--faces", action="store_true",
                    help="--rover: recognise faces; greet the people Milo knows and ask new ones their name "
                         "(robot/faces.py, robot/people.py; speaks with --voice, hears the answer with --listen)")
    ap.add_argument("--camera", default=None,
                    help="--rover: the head camera (robot/head.py) with the person detector: a V4L2 device, "
                         "auto (the Orbit if plugged in; default with hw) or none (default with fake: the "
                         "simulated person)")
    ap.add_argument("--safe-speed", action=argparse.BooleanOptionalAction, default=True,
                    help="robot safety layer: slow down near the person (robot/safety.py); on by "
                         "default (review 2026-09-28), --no-safe-speed for brain-only studies")
    a = ap.parse_args()
    os.environ.setdefault("FLY_DYNAMICS", "calibrated")
    if a.rover:
        if a.home or a.video_dir or a.topdown or a.speech:
            raise SystemExit("--rover: no --home, --video-dir, --topdown or --speech (Habitat only)")
        if a.rover == "hw" and not (a.lidar and a.lidar_port):
            # the lidar safety layer (brake, steering) must never be silently off on the robot
            raise SystemExit("--rover hw needs --lidar and --lidar-port")
        a.body = "rover"
        if a.rover == "fake" and a.lidar_port and not a.lidar:
            raise SystemExit("--rover fake --lidar-port: add --lidar (the real lidar is read only with it)")
        if a.faces and (a.camera or ("auto" if a.rover == "hw" else "none")) == "none":
            raise SystemExit("--faces needs a camera (--camera auto or a device)")
        os.environ.setdefault("FLY_TORCH_DEVICE", "cpu")   # the neocortex's small nets; the brain has the GPU
        os.environ.setdefault("FLY_TORCH_THREADS", "1")    # ...on one core (Jetson: same speed, the rest free)
        os.environ.setdefault("FLY_CRITIC", "numpy")       # ...or rather numpy: PyTorch's overhead was ~10 ms a step
    home = None
    if a.home:
        from sim.habitat_bridge.home import HomeWorld
        battery = None
        if a.battery is not None:
            from robot.battery import Battery
            battery = Battery(a.battery)
        rs = a.real_senses
        home = HomeWorld(pet_rate=0.0 if rs else a.pet_rate, treat_rate=0.0 if rs else a.treat_rate,
                         plant=not rs, seed=a.seed, battery=battery,
                         hunger_senses=not a.dock_reflex, body=a.body, odour=not rs)
    import config
    learning = None if a.learning is None else a.learning == "on"
    chans = tuple(x for x in a.channels.split(",") if x) if a.cortex != "none" else None
    if a.cortex == "pet" and "rest" not in chans:
        chans += ("rest",)       # naps drive ER5 (attached only here: an idle input would shift the RNG stream)
    if a.cortex == "pet" and "excite" not in chans and config.env_flag("FLY_VOICE"):
        # excitement -> P1 -> song. Attached only with the voice on: a stimulus
        # adds Poisson targets (RNG stream, no refractory period), so FLY_VOICE=0
        # leaves the brain's inputs as they were before the voice (not the earlier
        # runs' results: the dynamics and body readout changed since)
        chans += ("excite",)
    brain = (build_brain(a.seed, home=home, learning=learning, nav=a.nav, topdown=chans, lidar=a.lidar,
                         lidar_senses=a.lidar_use in ("both", "senses", "ttc"), body=a.body)
             if a.mode == "brain" else None)
    if brain is not None and a.rover:
        # real time on the Jetson: the mushroom body's engine writes wait for
        # the gap between 1 ms blocks, so blocks pipeline (Session.advance)
        brain[0].pipeline_plasticity = True
        brain[0].HISTORY_MAX = 600                # telemetry frames kept: hours of running
        brain[0].raster_on = False                # no display reads the spike raster on the robot
    if (a.avoid or a.route) and not a.lidar:
        raise SystemExit("--avoid / --route need --lidar")
    if brain is not None and a.lidar:
        brain[0].lidar_limit = a.lidar_use in ("both", "limit", "ttc")
        brain[0].lidar_ttc = a.lidar_use == "ttc"          # senses + footprint time-to-collision
        if a.avoid:
            if not chans or "goal" not in chans:
                raise SystemExit("--avoid steers through the neocortex's goal channel: needs --cortex")
            from robot.avoid import Avoid
            brain[0].avoid = Avoid(*brain[0].body_dims)
        # the unstick reflex pulls through the neocortex's goal channel: with
        # lidar steering, and with lidar alone when FLY_UNSTICK_LIDAR is on
        if (config.env_flag("FLY_UNSTICK") and chans and "goal" in chans
                and (a.avoid or config.env_flag("FLY_UNSTICK_LIDAR", False))):
            from robot.avoid import Unstick
            brain[0].unstick = Unstick(*brain[0].body_dims)
    cortex = None
    if a.cortex != "none" and brain is not None:
        from cortex.agent import make_cortex
        cortex = make_cortex(a.cortex, state_path=a.cortex_state, seed=a.seed)
        td = getattr(brain[0], "topdown", None)
        if getattr(cortex, "voice", False) and (td is None or "excite" not in td.channels or not len(td.excite.indices)):
            cortex.voice = False     # no P1 to drive (FAFB) or the channel is off: no excitement logged
        if a.route:
            from sim.habitat_bridge.bodies import BODIES
            cortex.enable_route(BODIES[a.body]["half_wid"])
    mb = brain[0].mb if brain is not None else None
    if mb is not None and a.weights and os.path.exists(a.weights):
        w = np.load(a.weights)
        mb.weights[:] = w
        mb.mult[mb.edge_pos] = (mb.kc_mbon_gain * w).astype(np.float32)
        getattr(mb.e, "commit_plastic", lambda *_: None)(mb.edge_pos)
        print("loaded mushroom-body weights from", a.weights, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    person = personality = voice = dashboard = eye = None
    if a.speech:
        from sim.habitat_bridge.speech import ScriptedPerson
        person = ScriptedPerson(name=a.pet_name, seed=a.seed)
    if a.personality:
        if cortex is None or not hasattr(cortex, "set_personality"):
            raise SystemExit("--personality needs --cortex v0 / v0_manners")
        from cortex.personality import Personality
        personality = Personality(url=a.personality, model=a.llm_model, name=a.pet_name,
                                  state_path=a.cortex_state,
                                  urge_gate=config.env_flag("FLY_URGE_GATE"))
    if a.voice:
        if not a.rover:
            raise SystemExit("--voice: the robot's speaker, with --rover")
        from robot.voice import Voice
        voice = Voice(a.speaker)
        atexit.register(voice.close, 1.0)
    ears = None
    if a.listen:
        if not a.rover:
            raise SystemExit("--listen: the robot's microphone, with --rover")
        from robot.speech import Ears
        ears = Ears(wav=a.listen_wav)
        atexit.register(ears.close)
    social = None
    if a.faces:
        if not a.rover:
            raise SystemExit("--faces: the real camera, with --rover")
        if not a.voice:
            raise SystemExit("--faces needs --voice: Milo greets and asks out loud")
        from robot.people import AnswerReader, Social
        reader = None
        if a.listen:
            # the language model reads the answers (Gemma via PAIR, as the personality)
            reader = AnswerReader(a.personality or os.environ.get("FLY_LLM_URL",
                                                                  "http://127.0.0.1:1234/v1/chat/completions"),
                                  model=a.llm_model, robot=a.pet_name)
            if not reader.available():
                print(f"faces: the language model ({reader.url}) does not answer: Milo will not ask names", flush=True)
                reader = None
        # names are asked only if Milo can hear the answer and understand it
        social = Social(a.pet_name, can_ask=a.listen, reader=reader)
    if a.rover:
        import signal
        import sys
        # systemctl stop: leave through the with-block, which stops the wheels
        # (if the process dies instead, the base's heartbeat stops them)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        from robot.rover_world import LocalConn, make_world
        cam = a.camera or ("auto" if a.rover == "hw" else "none")
        world = make_world(a.rover, a.ugv_port, a.lidar_port, LIDAR_BEAMS, a.hfov, camera=cam, faces=a.faces)
        if a.eye:
            if brain is None:
                raise SystemExit("--eye: the fly brain's optic lobes, with --mode brain")
            from robot.eye import EyeFeed, EyeRates
            head_shm = getattr(getattr(world.camera, "shm", None), "name", None)
            if head_shm is None and not a.lidar_port:
                raise SystemExit("--eye needs the camera (--camera) or the lidar (--lidar-port)")
            eye = EyeFeed(head_shm)
            atexit.register(eye.close)
            eye_rates = EyeRates(eye)
            brain[4].append((eye_rates, eye_rates))   # attached with the other senses each day
            print(f"eye: flyvis drives {eye.n} optic-lobe neurons (camera {'on' if head_shm else 'off'}, "
                  f"lidar {'on' if a.lidar_port else 'off'})", flush=True)
        if a.dashboard:
            from robot.dashboard import DashboardPublisher
            pub = DashboardPublisher(a.dashboard)
            atexit.register(pub.close)
            jpeg = getattr(world.camera, "preview_jpeg", None) if world.camera is not None else None
            from robot.brainmap import BrainActivity
            bm = BrainActivity(brain[0].c) if brain is not None else None
            if bm is not None:
                pub.put_static("/geometry", bm.geometry())
                pub.put_static("/circuit", bm.circuit())     # every neuron's strongest partners (tap one)
            dashboard = (pub, jpeg, bm)
            import socket
            print(f"dashboard: http://{socket.gethostname()}:{a.dashboard}/?key={pub.key}", flush=True)
        connection = LocalConn(world)
    else:
        from sim.habitat_bridge.authkey import authkey
        connection = Client(("127.0.0.1", a.port), authkey=authkey())
    with connection as conn:
        srv_body = _call(conn, {"cmd": "body"}).get("name", "spot")
        if srv_body != a.body:
            raise SystemExit(f"server body is {srv_body}, client --body {a.body}")
        if a.topdown:
            print("top-down map:", _call(conn, {"cmd": "topdown", "path": a.topdown}), flush=True)
        if a.lidar:
            rl = _call(conn, {"cmd": "lidar", "beams": LIDAR_BEAMS})
            if "error" in rl:
                raise SystemExit(f"lidar: {rl['error']}")

        def save_learning(final: bool = False):
            """The mushroom body's weights and the neocortex's memory, written
            atomically (a crash mid-write must not leave a broken file)."""
            if mb is not None and a.weights:
                tmp = a.weights + ".tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, mb.weights)
                os.replace(tmp, a.weights)
            if cortex is not None:
                if final:
                    cortex.save()
                elif hasattr(cortex, "checkpoint"):
                    cortex.checkpoint()
        for day, ep in enumerate(a.episodes):
            try:
                r = run_episode(conn, a.mode, ep, a.seconds, a.period_ms, a.seed + ep,
                                a.video_dir, brain, home, cortex, safe_speed=a.safe_speed,
                                person=person, personality=personality, voice=voice,
                                checkpoint=save_learning if a.rover else None, social=social, ears=ears,
                                dashboard=dashboard, eye=eye)
            except BaseException:
                if a.rover:
                    save_learning()               # stopped or crashed: keep what was learned
                if ears is not None:
                    ears.close()
                if voice is not None:
                    voice.close(wait_s=1.0)
                if dashboard is not None:
                    dashboard[0].close()
                raise
            r["day"] = day
            (out / f"{a.mode}_day{day}_ep{ep}.json" if a.home else out / f"{a.mode}_ep{ep}.json").write_text(json.dumps(r))
            save_learning(final=True)
            if a.home:
                h, L = r["home"], r.get("learning") or {}
                print(f"day {day} (ep {ep}): " + (f"battery {h['battery']} | " if "battery" in h else "")
                      + f"bowl first {h['first_bowl_s']} s, near {h['near_bowl_s']:.1f} s, "
                      f"eating {h['eating_s']:.1f} s, mean bowl dist {h['mean_bowl_dist']} m | plant {h['at_plant_s']:.1f} s "
                      f"| pets {h['pets']} treats {h['treats']} near person {h['near_person_s']:.1f} s "
                      f"| bumps {r['collisions']} (walking {r['bumps']['into_walking']}, fast {r['bumps']['fast']}, "
                      f"unwanted {r['bumps']['unwanted']}) | MB depressed {L.get('depressed_synapses')} {L.get('changed_mbons')}"
                      + (f" | cortex {r['cortex']}" if cortex is not None else "")
                      + (f" | calls answered {r['speech']['answered']}/{r['speech']['calls']}, scolds {r['speech']['scolds']}"
                         if person is not None else "")
                      + (f" | LLM {r['personality']['stats']}" if personality is not None else ""),
                      flush=True)
                continue
            st = r["habitat_stats"] or {}
            if a.rover:
                lg = r["log"]
                print(f"rover ({a.rover}) {a.mode} ep{ep}: {r['sim_s']}s in {r['wall_s']}s wall "
                      f"| steps over the {a.period_ms:.0f} ms period: {st.get('overruns', 0):.0f} "
                      f"(late {st.get('late_s', 0)} s) | brain+layers per step: mean {st.get('compute_ms_mean')} "
                      f"p95 {st.get('compute_ms_p95')} max {st.get('compute_ms_max')} ms | visible {r['visible_frac']:.2f} "
                      f"| moving {np.mean([abs(x['v']) > 0.02 for x in lg]) if lg else 0:.2f} "
                      f"| pressed against something {r['blocked_s']} s"
                      + (f" | camera {st['camera_fps']} fps, stale {st['camera_stale_s']} s, person seen "
                         f"{st['person_seen_s']} s" if "camera_fps" in st else "")
                      + (f" | heard {r['ears']['utterances']} utterances" if "ears" in r else ""), flush=True)
                continue
            print(f"{a.mode} ep{ep}: {r['sim_s']}s sim in {r['wall_s']}s | visible {r['visible_frac']:.2f} "
                  f"| mean dist {r['mean_dist']:.2f} m | follow-band {r['in_follow_band_frac']:.2f} "
                  f"| found at {r['found_s']} | bumps {r['collisions']} | habitat found_human "
                  f"{st.get('has_found_human')} follow_ratio {st.get('follow_human_steps_ratio_after_frist_encounter')}",
                  flush=True)
    if ears is not None:
        ears.close()
    if voice is not None:
        voice.close()
    if dashboard is not None:
        dashboard[0].close()


if __name__ == "__main__":
    main()
