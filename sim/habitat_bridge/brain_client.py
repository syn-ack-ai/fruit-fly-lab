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
       DNa01/DNa02 steering, MDN backward, plus the spontaneous walking rhythm
       the missing nerve cord would supply)
    -> robot velocity, SCALED from fly to robot (documented approximation):
       speed x 0.025 (fly walking 12 mm/s -> robot 0.3 m/s), limited to
       -0.3 .. +0.5 m/s; turn rate x 0.5 (deg/s), limited to +-120 deg/s.
       The robot cannot fly: a takeoff (escape) becomes a fast dash in the
       direction the body chose.

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


def build_brain(seed: int, home=None, learning: bool | None = None, nav: bool = False,
                topdown: tuple | None = None):
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
                safe_speed: bool = False, person=None, personality=None, face=None) -> dict:
    from fly.body.foraging_body import ForagingBody
    from robot.safety import ProximityGovernor
    governor = ProximityGovernor() if safe_speed else None
    obs = _call(conn, {"cmd": "reset", "episode": episode})
    n_env = int(round(period_ms / 1000.0 * 120.0))
    log = []
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
    if person is not None:
        person.reset()
    if personality is not None:
        personality.reset(episode)
    if face is not None:
        face[0].reset()
    said = []                                     # (t, personality output) when new
    prev_events = {"pets": 0, "treats": 0, "bowl": False, "seen_t": -1e9}
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
            if getattr(ses, "nav", None) is not None:
                ses.body.state.heading_deg = float(obs["robot"][2]) % 360.0   # odometry yaw
            if personality is not None and cortex is not None:
                pout = personality.step(t_sim, _digest(cortex, ses, obs, home))
                cortex.set_personality(pout["intent"], pout["feedback"] if pout["new"] else 0)
                if pout["new"]:
                    said.append((round(t_sim, 1), {k: pout[k] for k in ("intent", "sound", "say", "mood", "feedback")}))
            if cortex is not None:
                ses.topdown.apply(float(obs["robot"][2]) % 360.0, cortex.act(obs, home, t_sim, fr_prev))
            fr = fr_prev = ses.advance(period_ms)[-1]
            v, w = robot_command(ses.body.state)
            dn = {kk: round(vv, 1) for kk, vv in fr["dn_rates"].items()
                  if kk.startswith(("DNa01", "DNa02", "DNg100", "DNp09", "DNp01", "MDN", "DNge078"))}
            tb = round(fr["channels"].get("turn_bias", 0.0), 3)
        elif mode == "body_only":
            for i in range(int(period_ms)):
                body.update(1.0, {}, t_sim * 1000 + i)
            v, w = robot_command(body.state)
            tb = 0.0
        else:
            v, w, tb = 0.0, 0.0, 0.0
        if governor is not None:
            governor.observe(obs["visible"], obs["half"], t_sim)
            v = governor.limit(v, t_sim, period_ms / 1000.0)
        if home is not None and home.battery is not None:
            v, w = _emergency_return(conn, obs, home, emergency, v, w, period_ms / 1000.0)
            if home.battery.flat:
                v, w = 0.0, 0.0                   # a flat battery: the robot stops
        last_v = v
        prev_obs = obs
        obs = _call(conn, {"cmd": "step", "v": v, "w": w, "n": n_env,
                           "frame": bool(video_dir) and k % 5 == 0})
        bumped = obs.get("collisions", 0) > prev_obs.get("collisions", 0)
        if person is not None:
            heard = person.step(t_sim, period_ms / 1000.0, obs, prev_obs, bumped, v)
            if heard and personality is not None:
                personality.event(t_sim, f'your person said: "{heard}"')
        if personality is not None and mode == "brain":
            _events(personality, t_sim, obs, home, fr_prev, prev_events)
        if face is not None and mode == "brain":
            _face(face, period_ms / 1000.0, obs, ses, fr_prev, home, cortex, personality)
        if bumped:
            hs = math.hypot(obs["human"][0] - prev_obs["human"][0],
                            obs["human"][1] - prev_obs["human"][1]) / (period_ms / 1000.0)
            wanted = None if cortex is None or not hasattr(cortex, "social") else bool(cortex.social >= 0.35)
            bumps.append((round(t_sim, 1), round(hs, 2), round(v, 3), wanted))
        t_sim += period_ms / 1000.0
        k += 1
        if found_t is None and obs["visible"] and obs["dist"] < 2.0:
            found_t = t_sim
        log.append({"t": round(t_sim, 2), "dist": round(obs["dist"], 3), "visible": obs["visible"],
                    "robot": [round(x, 2) for x in obs["robot"]], "human": [round(x, 2) for x in obs["human"]],
                    "az": round(obs["az"], 1), "v": round(v, 3), "w_deg": round(math.degrees(w), 1),
                    "turn_bias": tb, "dn": dn,
                    "lc10a_hz": (obj.last.get("drive_hz") if mode == "brain" else None)})
        if home is not None:
            log[-1].update({"odour": [round(float(home.conc["L"].sum()), 3), round(float(home.conc["R"].sum()), 3)],
                            "taste": home.taste, "pet": home.petting, "treat": home.treating,
                            "proboscis": round(float(ses.body.state.proboscis_extension), 2) if mode == "brain" else 0.0,
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
           "scene_contacts": obs.get("scene_contacts"), "habitat_stats": obs["stats"],
           "video": video, "log": log}
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
    if personality is not None:
        personality.wait(30.0)
        res["personality"] = {"said": said, "stats": dict(personality.stats), "log": personality.log}
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


def _emergency_return(conn, obs, home, em, v, w, dt):
    """Robot-level safety net, not the pet's choice: below EMERGENCY_SOC the
    navigation layer drives to the dock and docks (as a robot vacuum returns to
    base; on the real robot, Nav2 docking). The brain keeps running but its
    motor output is ignored until the charge is back to EMERGENCY_DONE. Every
    return is recorded as a FAILURE of the pet's own behaviour."""
    from sim.habitat_bridge.home import BOWL_XZ, EAT_R_M
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
    if math.hypot(x - BOWL_XZ[0], z - BOWL_XZ[1]) < EAT_R_M:
        home.docked_by_nav = True                 # on the contacts: charges without "eating"
        return 0.0, 0.0
    p = _call(conn, {"cmd": "path", "goal": list(BOWL_XZ), "ahead": 0.6})
    from cortex.topdown import bearing_deg
    err = ((bearing_deg((x, z), p["waypoint"]) - yaw + 180.0) % 360.0) - 180.0   # + = turn left (CCW)
    w = math.radians(max(-90.0, min(90.0, 2.0 * err)))
    v = 0.3 * max(0.0, math.cos(math.radians(err)))
    return v, w


def _digest(cortex, ses, obs, home=None) -> dict:
    """What the personality layer is told each call (cortex/personality.py)."""
    from robot.safety import person_distance
    d = person_distance(obs["half"]) if obs["visible"] else None
    side = "ahead" if abs(obs["az"]) < 20 else ("to the right" if obs["az"] > 0 else "to the left")
    goal = cortex.goal[1] if getattr(cortex, "goal", None) else "nothing in particular"
    # what the neocortex remembers: the nearest place where it has tasted food
    x, z = obs["robot"][0], obs["robot"][1]
    food = [cortex.map.food_point(c) for c, n in cortex.map.nodes.items() if n["food"] > 0.1]
    fd = min((math.hypot(fx - x, fz - z) for fx, fz in food), default=None)
    return {"hunger": cortex.hunger, "social": cortex.social,
            # sleepiness only means something to a pet that naps (review 2026-09-26)
            "sleepy": cortex.sleepy if getattr(cortex, "naps", False) else None,
            "battery": getattr(getattr(home, "battery", None), "soc", None) if home is not None else None,
            "behaviour": ses.body.state.behaviour,
            "person": f"{d:.1f} m {side}" if d is not None else "not in view",
            "food": f"you remember food {fd:.1f} m away" if fd is not None else "you do not know where food is yet",
            "doing": f"heading for {goal}" + (f"; manners: {cortex.manner}" if getattr(cortex, "manner", None) else "")}


def _events(pers, t, obs, home, fr, prev):
    """Things worth a reaction, for the personality layer."""
    if home is not None:
        if home.stats["pets"] > prev["pets"]:
            pers.event(t, "your person petted you")
        if home.stats["treats"] > prev["treats"]:
            pers.event(t, "your person gave you a treat")
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
            pers.event(t, "something startled you")
        prev["startle_t"] = t


def _face(face, dt, obs, ses, fr, home, cortex, personality):
    """The face (robot/face.py) from the brain, body and personality."""
    from robot.safety import person_distance
    from cortex.topdown import goal_azimuth
    model, pub = face
    ch = fr["channels"] if fr is not None else {}
    fresh = personality.take_fresh() if personality is not None else None
    goal = ses.topdown.last.get("goal_deg") if getattr(ses, "topdown", None) is not None else None
    st = model.update(dt, {
        "person_visible": obs["visible"], "person_az": obs["az"], "person_el": obs["el"],
        "person_d": person_distance(obs["half"]) if obs["visible"] else None,
        "goal_az": goal_azimuth(goal, float(obs["robot"][2])) if goal is not None else None,
        "startle": max(ch.get("escape_takeoff", 0.0), ch.get("escape_long_mode", 0.0)),
        "speed": ses.body.state.speed_mm_s * SPEED_SCALE, "behaviour": ses.body.state.behaviour,
        "eating": ses.body.state.proboscis_extension > 0.5, "grooming": ch.get("groom", 0.0) > 0.5,
        "petting": bool(home.petting) if home is not None else False,
        "mood": personality.current["mood"] if personality is not None else "calm",
        "say": fresh["say"] if fresh else None, "sound": fresh["sound"] if fresh else None,
        "hunger": getattr(cortex, "hunger", 0.0), "social": getattr(cortex, "social", 0.0)})
    pub.send(st)


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
    ap.add_argument("--pet-name", default="Mote")
    ap.add_argument("--face", default=None, metavar="URL",
                    help="send the face to robot/face_server.py, e.g. http://127.0.0.1:8010/state")
    ap.add_argument("--safe-speed", action="store_true",
                    help="robot safety layer: slow down near the person (robot/safety.py)")
    a = ap.parse_args()
    os.environ.setdefault("FLY_DYNAMICS", "calibrated")
    home = None
    if a.home:
        from sim.habitat_bridge.home import HomeWorld
        battery = None
        if a.battery is not None:
            from robot.battery import Battery
            battery = Battery(a.battery)
        home = HomeWorld(pet_rate=a.pet_rate, treat_rate=a.treat_rate, seed=a.seed, battery=battery,
                         hunger_senses=not a.dock_reflex)
    learning = None if a.learning is None else a.learning == "on"
    chans = tuple(x for x in a.channels.split(",") if x) if a.cortex != "none" else None
    if a.cortex == "pet" and "rest" not in chans:
        chans += ("rest",)       # naps drive ER5 (attached only here: an idle input would shift the RNG stream)
    brain = build_brain(a.seed, home=home, learning=learning, nav=a.nav, topdown=chans) if a.mode == "brain" else None
    cortex = None
    if a.cortex != "none" and brain is not None:
        from cortex.agent import make_cortex
        cortex = make_cortex(a.cortex, state_path=a.cortex_state, seed=a.seed)
    mb = brain[0].mb if brain is not None else None
    if mb is not None and a.weights and os.path.exists(a.weights):
        w = np.load(a.weights)
        mb.weights[:] = w
        mb.mult[mb.edge_pos] = (mb.kc_mbon_gain * w).astype(np.float32)
        getattr(mb.e, "commit_plastic", lambda *_: None)(mb.edge_pos)
        print("loaded mushroom-body weights from", a.weights, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    person = personality = face = None
    if a.speech:
        from sim.habitat_bridge.speech import ScriptedPerson
        person = ScriptedPerson(name=a.pet_name, seed=a.seed)
    if a.personality:
        if cortex is None or not hasattr(cortex, "set_personality"):
            raise SystemExit("--personality needs --cortex v0 / v0_manners")
        from cortex.personality import Personality
        personality = Personality(url=a.personality, model=a.llm_model, name=a.pet_name,
                                  state_path=a.cortex_state)
    if a.face:
        from robot.face import FaceModel, FacePublisher
        face = (FaceModel(), FacePublisher(a.face))
    from sim.habitat_bridge.authkey import authkey
    with Client(("127.0.0.1", a.port), authkey=authkey()) as conn:
        if a.topdown:
            print("top-down map:", _call(conn, {"cmd": "topdown", "path": a.topdown}), flush=True)
        for day, ep in enumerate(a.episodes):
            r = run_episode(conn, a.mode, ep, a.seconds, a.period_ms, a.seed + ep,
                            a.video_dir, brain, home, cortex, safe_speed=a.safe_speed,
                            person=person, personality=personality, face=face)
            r["day"] = day
            (out / f"{a.mode}_day{day}_ep{ep}.json" if a.home else out / f"{a.mode}_ep{ep}.json").write_text(json.dumps(r))
            if mb is not None and a.weights:
                np.save(a.weights, mb.weights)
            if cortex is not None:
                cortex.save()
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
            print(f"{a.mode} ep{ep}: {r['sim_s']}s sim in {r['wall_s']}s | visible {r['visible_frac']:.2f} "
                  f"| mean dist {r['mean_dist']:.2f} m | follow-band {r['in_follow_band_frac']:.2f} "
                  f"| found at {r['found_s']} | bumps {r['collisions']} | habitat found_human "
                  f"{st.get('has_found_human')} follow_ratio {st.get('follow_human_steps_ratio_after_frist_encounter')}",
                  flush=True)


if __name__ == "__main__":
    main()
