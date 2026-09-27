"""
The neocortex's memory of where things are: a fine occupancy map built from
the lidar, and routes planned around what it holds.

Animals remember obstacles and plan around them rather than rediscovering them
at every step: walking cats remember an obstacle they stepped over for
minutes and adjust their hind legs without seeing it again (McVea & Pearson
2007 J Neurosci 27:3052); rats have boundary-vector cells that code the
distance and direction to walls (Lever et al. 2009 J Neurosci 29:9771). The
topological map in cortex/v0.py (0.5 m places joined by moves made) knows
which places connect, not where the furniture is; this adds that.

  map      RES_M cells holding log-odds of "occupied": each lidar hit raises its
           cell, the cells a beam passed through are lowered (Moravec & Elfes
           1985). Clamped, so moved furniture is forgotten after a few scans.
  plan     A* over the map: occupied cells forbidden, cells within the body's
           radius + margin of one very costly (effectively forbidden unless the
           pet is already there), unknown cells allowed at extra cost (a cat
           will try a way it has not seen), within WINDOW_M of the pet
  route    the point ROUTE_AHEAD_M along the path, as the neocortex's goal
           direction (the fly brain's goal channel) instead of the straight line

On the robot the map comes from SLAM (slam_toolbox on the rover's lidar), or a
scan of the home (e.g. an iPhone / iPad LiDAR room scan) loaded at "birth" as
known ground. C. APPROXIMATIONS: pose = Habitat ground truth (odometry + SLAM
on the robot); 2D only, at the lidar's height.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

RES_M = 0.1
SIZE = 400                      # cells per side (40 m), centred on the first pose
L_HIT, L_MISS = 0.9, -0.35
L_MIN, L_MAX = -2.0, 3.5
OCC_L = 0.6                     # log-odds above this = occupied
MAX_RAY_M = 4.0                 # free space is only written this far along a beam
INFLATE_M = 0.06                # margin beyond the body's radius
UNKNOWN_COST = 1.5              # per-cell cost multiplier for unseen ground
MARGIN_COST = 25.0              # ... for cells within the body's radius of an obstacle
WINDOW_M = 5.0                  # plan within this of the pet
ROUTE_AHEAD_M = 0.6


class ObstacleMap:
    def __init__(self, body_radius_m: float):
        self.logodds = np.zeros((SIZE, SIZE), np.float32)
        self.seen = np.zeros((SIZE, SIZE), bool)
        self.origin = None                      # world (x, z) of cell (0, 0)
        self.body_radius = body_radius_m        # half the body's WIDTH: it faces along its path
        self.radius = body_radius_m + INFLATE_M
        self._inflated, self._dirty = None, True
        k = int(math.ceil(self.radius / RES_M))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        self._disk = (xx ** 2 + yy ** 2) * RES_M ** 2 <= self.radius ** 2
        self.last = {}

    # ------------------------------------------------------------- geometry
    def cell(self, x, z):
        return (np.floor((np.asarray(x) - self.origin[0]) / RES_M).astype(int),
                np.floor((np.asarray(z) - self.origin[1]) / RES_M).astype(int))

    def centre(self, i, j):
        return (self.origin[0] + (i + 0.5) * RES_M, self.origin[1] + (j + 0.5) * RES_M)

    def _inside(self, i, j):
        return (i >= 0) & (i < SIZE) & (j >= 0) & (j < SIZE)

    # --------------------------------------------------------------- update
    def update(self, pose, ranges, angles_deg, max_range: float = 8.0) -> None:
        """One scan. pose = (x, z, yaw_deg), yaw CCW in the (x, -z) plane;
        angles + = right of the heading (robot/lidar.py)."""
        x, z, yaw = pose
        if self.origin is None:
            self.origin = (x - SIZE * RES_M / 2, z - SIZE * RES_M / 2)
        r = np.asarray(ranges, float)
        b = np.radians(yaw - np.asarray(angles_deg, float))
        dx, dz = np.cos(b), -np.sin(b)
        hit = np.isfinite(r) & (r < max_range - 1e-3)
        rr = np.where(np.isfinite(r), np.minimum(r, max_range), max_range)
        # free space: samples along each beam short of its hit
        t = np.arange(0.0, MAX_RAY_M, RES_M / 2)
        T = t[None, :]
        ok = T < (rr[:, None] - RES_M)
        fx, fz = x + T * dx[:, None], z + T * dz[:, None]
        fi, fj = self.cell(fx[ok], fz[ok])
        # hits
        hi, hj = self.cell(x + rr[hit] * dx[hit], z + rr[hit] * dz[hit])
        hin = self._inside(hi, hj)
        hcells = set(zip(hi[hin].tolist(), hj[hin].tolist()))
        fin = self._inside(fi, fj)
        free = set(zip(fi[fin].tolist(), fj[fin].tolist())) - hcells
        if free:
            a = np.array(list(free))
            self.logodds[a[:, 0], a[:, 1]] += L_MISS
            self.seen[a[:, 0], a[:, 1]] = True
        if hcells:
            a = np.array(list(hcells))
            self.logodds[a[:, 0], a[:, 1]] += L_HIT
            self.seen[a[:, 0], a[:, 1]] = True
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)
        self._dirty = True

    def occupied(self) -> np.ndarray:
        return self.logodds > OCC_L

    def inflated(self) -> np.ndarray:
        """Cells the body's centre may not enter."""
        if self._dirty or self._inflated is None:
            from scipy.ndimage import binary_dilation
            self._inflated = binary_dilation(self.occupied(), structure=self._disk)
            self._dirty = False
        return self._inflated

    # ------------------------------------------------------------- planning
    def plan(self, start_xz, goal_xz):
        """A* path (world points) from start to goal, or None."""
        if self.origin is None:
            return None
        blocked = self.inflated()
        si, sj = (int(v) for v in self.cell(*start_xz))
        gx, gz = goal_xz
        w = int(WINDOW_M / RES_M)
        # a goal beyond the window: aim at the window's edge toward it
        d = math.hypot(gx - start_xz[0], gz - start_xz[1])
        if d > WINDOW_M * 0.9:
            f = WINDOW_M * 0.9 / d
            gx, gz = start_xz[0] + f * (gx - start_xz[0]), start_xz[1] + f * (gz - start_xz[1])
        gi, gj = (int(v) for v in self.cell(gx, gz))
        lo_i, hi_i = max(si - w, 0), min(si + w, SIZE - 1)
        lo_j, hi_j = max(sj - w, 0), min(sj + w, SIZE - 1)
        if not (lo_i <= gi <= hi_i and lo_j <= gj <= hi_j):
            return None
        occ = self.occupied()
        if blocked[gi, gj]:                            # goal inside an obstacle's margin: nearest free cell
            ii, jj = np.nonzero(~blocked[lo_i:hi_i + 1, lo_j:hi_j + 1])
            if not len(ii):
                return None
            k = int(np.argmin((ii + lo_i - gi) ** 2 + (jj + lo_j - gj) ** 2))
            gi, gj = int(ii[k] + lo_i), int(jj[k] + lo_j)
        seen = self.seen
        open_ = [(0.0, 0.0, si, sj)]
        came = {(si, sj): None}
        cost = {(si, sj): 0.0}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while open_:
            _, g, i, j = heapq.heappop(open_)
            if (i, j) == (gi, gj):
                break
            if g > cost[(i, j)]:
                continue
            for di, dj, c in steps:
                ii, jj = i + di, j + dj
                if not (lo_i <= ii <= hi_i and lo_j <= jj <= hi_j):
                    continue
                if occ[ii, jj]:
                    continue                           # the obstacle itself
                # within the body's margin: allowed only at a high cost (so a
                # pet already brushing something can still plan its way out)
                ng = g + c * (1.0 if seen[ii, jj] else UNKNOWN_COST) * (MARGIN_COST if blocked[ii, jj] else 1.0)
                if ng < cost.get((ii, jj), np.inf):
                    cost[(ii, jj)] = ng
                    came[(ii, jj)] = (i, j)
                    h = math.hypot(gi - ii, gj - jj)
                    heapq.heappush(open_, (ng + h, ng, ii, jj))
        if (gi, gj) not in came:
            return None
        path, k = [], (gi, gj)
        while k is not None:
            path.append(self.centre(*k))
            k = came[k]
        return path[::-1]

    def route_point(self, start_xz, goal_xz):
        """Where to head now: the planned path's point ROUTE_AHEAD_M along, or
        None (no map yet / no path: the neocortex keeps its straight line)."""
        path = self.plan(start_xz, goal_xz)
        if not path:
            self.last = {"route": False}
            return None
        acc, prev = 0.0, start_xz
        for p in path[1:]:
            acc += math.hypot(p[0] - prev[0], p[1] - prev[1])
            prev = p
            if acc >= ROUTE_AHEAD_M:
                self.last = {"route": True, "len_m": round(self._length(path), 2)}
                return p
        self.last = {"route": True, "len_m": round(self._length(path), 2)}
        return path[-1]

    @staticmethod
    def _length(path):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
