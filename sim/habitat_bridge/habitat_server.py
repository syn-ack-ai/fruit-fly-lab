"""
Habitat 3.0 social-navigation home, served to the fly brain over a local socket.

Runs in the "habitat" conda env (Python 3.9, habitat-sim 0.3.3, habitat-lab
0.3.3; see README.md). The brain runs in the repo's own .venv
(sim/habitat_bridge/brain_client.py) because the two environments differ; they
talk through multiprocessing.connection with plain Python values only (no numpy
objects cross the socket, so numpy versions need not match).

What the brain gets (GROUND TRUTH, not a detector - labelled as such):
  the person's bearing, elevation and angular size through a virtual head camera
  on the robot (0.35 m high, pitched up 20 deg, 53 deg field of view like the real
  Orbit head), and whether they are visible (inside that view and not occluded,
  by a ray cast from the camera to the person). The target is the person's head
  and shoulders (0.5 m wide), as on the real robot head (robot/head.py). The
  rendered video comes from Spot's own head camera (the sensor's pose is not
  used for the geometry: it did not track the base in kinematic mode).

Scene: the Habitat 3.0 benchmark subset of HSSD (hab3_bench_assets: a small,
medium and large house) with the Spot robot as the robot body (kinematic base
velocity control) and a humanoid that walks to random places at a human walking
pace (the task's own OracleNavRandCoordAction, slowed from its training speed).

    conda run -n habitat python -m sim.habitat_bridge.habitat_server --port 6010
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from sim.habitat_bridge.authkey import authkey   # before make_env changes directory

HOUSES = {"small": "small_small", "medium": "medium_medium", "large": "large_large"}
CTRL_HZ = 120.0                 # env step = 1/120 s (kinematic mode, ac_freq_ratio 1)
BASE_SPEED = 5.0                # m/s and rad/s at action +-1 (config values)
HUMAN_SPEED_SCALE = 0.07        # -> ~0.7 m/s, casual indoor walking (0.11 gave 1.1 m/s)
TARGET_HALF_WIDTH_M = 0.25      # head and shoulders
TARGET_HEIGHT_M = 1.45          # above the floor (humanoid is 1.66 m; base is at the floor)
CAM_HEIGHT_M = 0.35             # the pet's head camera above the floor


def make_env(house: str, max_seconds: float, habitat_lab_dir: str):
    import habitat
    from habitat.config import read_write
    from habitat.config.default import get_config
    os.chdir(habitat_lab_dir)                        # the configs use data/... paths
    hs = BASE_SPEED * HUMAN_SPEED_SCALE
    cfg = get_config("benchmark/multi_agent/hssd_spot_human_social_nav.yaml", overrides=[
        "habitat.dataset.data_path=data/hab3_bench_assets/episode_datasets/%s.json.gz" % HOUSES[house],
        "habitat.dataset.scenes_dir=data/hab3_bench_assets/hab3-hssd/",
        "habitat.simulator.scene_dataset=data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json",
        "habitat.environment.max_episode_steps=%d" % int(max_seconds * CTRL_HZ),
        # a pet exploring a home bumps into furniture: count it, don't end the episode
        "habitat.task.measurements.social_nav_reward.max_count_colls=-1",
        # both oracle-nav actions: they share one humanoid walk controller
        "habitat.task.actions.agent_1_oracle_nav_randcoord_action.lin_speed=%g" % hs,
        "habitat.task.actions.agent_1_oracle_nav_randcoord_action.ang_speed=%g" % hs,
        "habitat.task.actions.agent_1_oracle_nav_action.lin_speed=%g" % hs,
        "habitat.task.actions.agent_1_oracle_nav_action.ang_speed=%g" % hs,
    ])
    with read_write(cfg):
        cfg.habitat.dataset.split = "train"
    return habitat.Env(config=cfg)


class Server:
    LIDAR_H_M, LIDAR_MAX_M = 0.20, 8.0      # UGV Rover-like lidar height; range limit

    def __init__(self, house: str, max_seconds: float, hfov_deg: float, habitat_lab_dir: str,
                 cam_height: float = CAM_HEIGHT_M, cam_pitch: float = 20.0):
        self.env = make_env(house, max_seconds, habitat_lab_dir)
        self.hfov = hfov_deg
        self.cam_height, self.cam_pitch = cam_height, cam_pitch
        self.frames = []
        self._prev_half = None
        self.t = 0.0
        self.max_seconds = max_seconds
        self.collisions = 0
        self._colliding = False

    # --------------------------------------------------------------- geometry
    def summary(self) -> dict:
        """Ground-truth person geometry through the pet's VIRTUAL head camera:
        at the robot base, CAM_HEIGHT above the floor, facing the robot's heading,
        pitched up by cam_pitch (a small pet looks up at people), with the real
        robot head's horizontal field of view. Visible = inside that view and a
        ray from the camera to the person hits nothing else first."""
        import habitat_sim
        import magnum as mn
        sim = self.env.sim
        robot = sim.agents_mgr[0].articulated_agent
        human = sim.agents_mgr[1].articulated_agent
        rp, hp = np.array(robot.base_pos), np.array(human.base_pos)
        fwd = robot.base_transformation.transform_vector(mn.Vector3(1, 0, 0))
        yaw = math.atan2(-fwd.z, fwd.x)                  # CCW from +x seen from above (+y)
        cam = rp + np.array([0.0, self.cam_height, 0.0])
        tgt = hp + np.array([0.0, TARGET_HEIGHT_M, 0.0])
        d = tgt - cam
        # horizontal: (u, w) = (x, -z) is a standard CCW frame when seen from above
        du, dw = d[0], -d[2]
        ahead_h = du * math.cos(yaw) + dw * math.sin(yaw)
        right = du * math.sin(yaw) - dw * math.cos(yaw)
        az = math.degrees(math.atan2(right, ahead_h))    # + = person to the right
        el = math.degrees(math.atan2(d[1], math.hypot(du, dw))) - self.cam_pitch
        rng = float(np.linalg.norm(d))
        half = math.degrees(math.atan2(TARGET_HALF_WIDTH_M, max(rng, 0.05)))
        in_fov = ahead_h > 0 and abs(az) <= self.hfov / 2
        visible = False
        if in_fov:
            origin = mn.Vector3(*cam.tolist())
            dv = mn.Vector3(*d.tolist())
            hits = sim.cast_ray(habitat_sim.geo.Ray(origin, dv.normalized()),
                                max_distance=float(dv.length()) - 0.3)
            ignore = {human.sim_obj.object_id, robot.sim_obj.object_id}
            visible = not (hits.has_hits() and any(h.object_id not in ignore for h in hits.hits))
        m = self.env.get_metrics()
        out_lidar = self._lidar(sim, robot, rp, yaw) if getattr(self, "lidar_beams", 0) else None
        return {"t": round(self.t, 4), "dist": float(np.linalg.norm((hp - rp)[[0, 2]])),
                "lidar": out_lidar,
                "az": az, "el": el, "half": half, "visible": bool(visible), "in_fov": bool(in_fov),
                "robot": [float(rp[0]), float(rp[2]), math.degrees(yaw)],
                "human": [float(hp[0]), float(hp[2])],
                "over": bool(self.env.episode_over),
                "collided": self.collisions > 0, "collisions": self.collisions,
                "scene_contacts": int((m.get("robot_collisions") or {}).get("robot_scene_colls", 0) or 0),
                "stats": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else None)
                          for k, v in (m.get("social_nav_stats") or {}).items()}}

    def _lidar(self, sim, robot, rp, yaw) -> list:
        """A 2D lidar scan: ranges (m) for beams at lidar_angles (deg, + = right
        of the heading), in the horizontal plane at LIDAR_H_M; the robot's own
        body is ignored, the person is seen (their legs). yaw in radians."""
        import habitat_sim
        import magnum as mn
        own = {robot.sim_obj.object_id} | set(getattr(robot.sim_obj, "link_object_ids", {}).keys())
        origin = mn.Vector3(float(rp[0]), float(rp[1]) + self.LIDAR_H_M, float(rp[2]))
        out = []
        for a in self.lidar_angles:
            b = yaw - math.radians(a)                        # yaw is in RADIANS here (summary: atan2);
            # world bearing CCW from +x in (x, -z). (Review 2026-09-26: an extra
            # radians() left the beams fixed to the world axes.)
            d = mn.Vector3(math.cos(b), 0.0, -math.sin(b))
            hits = sim.cast_ray(habitat_sim.geo.Ray(origin, d), max_distance=self.LIDAR_MAX_M)
            r = self.LIDAR_MAX_M
            if hits.has_hits():
                for h in hits.hits:
                    if h.object_id not in own:
                        r = min(r, float(h.ray_distance))
            out.append(round(r, 3))
        return out

    def set_lidar(self, beams: int) -> dict:
        self.lidar_beams = int(beams)
        self.lidar_angles = [(-180.0 + 360.0 * k / beams) for k in range(beams)] if beams else []
        return {"beams": self.lidar_beams, "angles": self.lidar_angles}

    # ------------------------------------------------------------------ calls
    def reset(self, episode: int | None = None) -> dict:
        env = self.env
        if episode is not None:
            eps = env.episodes
            env.current_episode = eps[episode % len(eps)]
        env.reset()
        self.frames = []
        self.t = 0.0
        self.collisions = 0
        self._colliding = False
        return self.summary()

    def step(self, v: float, w: float, n: int, frame: bool = False) -> dict:
        act = {"action": ("agent_0_base_velocity", "agent_1_oracle_nav_randcoord_action"),
               "action_args": {"agent_0_base_vel": np.array([v / BASE_SPEED, w / BASE_SPEED], np.float32),
                               "agent_1_oracle_nav_randcoord_action": np.array([1.0], np.float32)}}
        obs = None
        for _ in range(int(n)):
            if self.env.episode_over:
                break
            obs = self.env.step(act)
            self.t += 1.0 / CTRL_HZ
            # The task ends an episode when robot and person touch; here a bump
            # is counted (once per contact) and the episode goes on.
            hit = bool(self.env.get_metrics().get("did_collide", False))
            if hit and not self._colliding:
                self.collisions += 1
            self._colliding = hit
            if self.env.episode_over and hit and self.t < self.max_seconds - 1.0 / CTRL_HZ:
                self.env._episode_over = False
                self.env.task.should_end = False
        if frame and obs is not None:
            self.frames.append(np.asarray(obs["agent_0_head_rgb"])[..., :3].copy())
        return self.summary()

    def topdown(self, path: str, mpp: float = 0.03) -> dict:
        """The house's navigable area from above, for drawing trajectories:
        row = (z - zmin) / mpp, col = (x - xmin) / mpp."""
        from habitat.utils.visualizations import maps
        pf = self.env.sim.pathfinder
        lo, hi = pf.get_bounds()
        floor_y = float(pf.snap_point(pf.get_random_navigable_point())[1])
        td = maps.get_topdown_map(pf, height=floor_y, meters_per_pixel=mpp)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, map=td, xmin=float(lo[0]), zmin=float(lo[2]), mpp=mpp)
        return {"path": path, "shape": list(td.shape)}

    def path(self, goal_xz, ahead_m: float = 0.8) -> dict:
        """Shortest navigable path from the robot to a floor point (x, z), for
        ORACLE baselines only (a real pet has no map of the house): the
        geodesic distance and the first path point at least ahead_m away."""
        import habitat_sim
        sim = self.env.sim
        pf = sim.pathfinder
        rp = np.array(sim.agents_mgr[0].articulated_agent.base_pos)
        start = pf.snap_point(rp)
        end = pf.snap_point(np.array([goal_xz[0], float(start[1]), goal_xz[1]], np.float32))
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = start, end
        if not pf.find_path(sp) or len(sp.points) < 2:
            return {"ok": False, "geodesic": None, "waypoint": [float(goal_xz[0]), float(goal_xz[1])]}
        pts = [np.array(q) for q in sp.points]
        wp, acc = pts[-1], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            acc += float(np.linalg.norm(b - a))
            if float(np.linalg.norm((b - rp)[[0, 2]])) >= ahead_m:
                wp = b
                break
        return {"ok": True, "geodesic": float(sp.geodesic_distance), "waypoint": [float(wp[0]), float(wp[2])]}

    def save_video(self, path: str, fps: float = 10.0) -> str:
        if not self.frames:
            return ""
        import imageio
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)
        strip = np.concatenate(self.frames[:: max(1, len(self.frames) // 6)][:6], axis=1)
        imageio.imwrite(path.rsplit(".", 1)[0] + "_strip.png", strip)
        return path


def main():
    from multiprocessing.connection import Listener
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--house", choices=list(HOUSES), default="small")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--hfov", type=float, default=53.0,
                    help="field of view the brain gets (deg); 53 = the real robot head")
    ap.add_argument("--cam-pitch", type=float, default=20.0, help="head tilted up (deg)")
    ap.add_argument("--habitat-lab", default=os.path.expanduser("~/habitat-lab"))
    a = ap.parse_args()
    srv = Server(a.house, a.max_seconds, a.hfov, a.habitat_lab, cam_pitch=a.cam_pitch)
    print("habitat server ready on port", a.port, flush=True)
    with Listener(("127.0.0.1", a.port), authkey=authkey()) as lst:
        while True:
            with lst.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    t0 = time.time()
                    try:
                        out = handle(srv, msg, conn)
                    except Exception as ex:          # keep serving; tell the client
                        out = {"error": repr(ex)}
                    if out is None:
                        return
                    out["server_s"] = round(time.time() - t0, 4)
                    conn.send(out)


def handle(srv, msg, conn):
    """One command -> its reply, or None to shut the server down."""
    cmd = msg.get("cmd")
    if cmd == "reset":
        return srv.reset(msg.get("episode"))
    if cmd == "step":
        return srv.step(msg["v"], msg["w"], msg.get("n", 12), msg.get("frame", False))
    if cmd == "lidar":
        return srv.set_lidar(msg.get("beams", 90))
    if cmd == "path":
        return srv.path(msg["goal"], msg.get("ahead", 0.8))
    if cmd == "topdown":
        return srv.topdown(msg["path"], msg.get("mpp", 0.03))
    if cmd == "video":
        return {"path": srv.save_video(msg["path"], msg.get("fps", 10.0))}
    if cmd == "close":
        conn.send({"ok": True})
        return None
    return {"error": "unknown command %r" % cmd}


if __name__ == "__main__":
    main()
