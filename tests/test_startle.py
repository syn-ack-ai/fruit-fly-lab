"""The wheeled body's startle (fly/body/foraging_body.py wheeled=True) and the
backing limit behind it (robot/safety.rear_limit)."""
import numpy as np

from fly.body.foraging_body import (STARTLE_BACK_MS, STARTLE_FREEZE_MS, STARTLE_WATCH_MS,
                                    ForagingBody)
from robot.safety import rear_limit


def run(body, ms, ch, lat=0.0, t0=0.0):
    out = []
    for i in range(int(ms)):
        s = body.update(1.0, ch, t0 + i, escape_laterality=lat)
        out.append((s.behaviour, s.speed_mm_s, s.turn_rate_deg_s, s.airborne))
    return out


def test_escape_becomes_a_startle_not_a_takeoff():
    b = ForagingBody(neural=True, seed=1, spontaneous_takeoff_per_s=0.0, wheeled=True)
    threat = {"escape_long_mode": 0.8}
    tr = run(b, 4000, threat, lat=0.5)                    # a threat on the right, held
    assert not any(a for *_, a in tr)                      # never airborne
    beh = [x[0] for x in tr]
    assert beh[10] == "startle (freeze)"
    k = int(STARTLE_FREEZE_MS + 100)
    assert beh[k] == "startle (back off)" and tr[k][1] < 0 and tr[k][2] < 0   # backs, turns right (cw)
    assert beh[int(STARTLE_FREEZE_MS + STARTLE_BACK_MS["long"] + 100)] == "startle (watch)"
    end = int(STARTLE_FREEZE_MS + STARTLE_BACK_MS["long"] + STARTLE_WATCH_MS) + 50
    assert not beh[end].startswith("startle")              # over, though the threat is still there
    assert b.startles == 1                                  # one threat, one startle
    run(b, 300, {}, t0=4000)                               # the command falls ...
    run(b, 100, threat, lat=-0.5, t0=4300)                 # ... and rises: a new startle
    assert b.startles == 2
    s = run(b, 600, threat, lat=-0.5, t0=4400)[-1]
    assert s[0] == "startle (back off)" and s[2] > 0       # threat on the left: turns left


def test_giant_fibre_startle_backs_off_longer_and_old_body_still_flies():
    b = ForagingBody(neural=True, seed=1, spontaneous_takeoff_per_s=0.0, wheeled=True)
    tr = run(b, 2000, {"escape_takeoff": 0.9})
    backing = sum(1 for x in tr if x[0] == "startle (back off)")
    assert abs(backing - STARTLE_BACK_MS["short"]) <= 2
    fly = ForagingBody(neural=True, seed=1, spontaneous_takeoff_per_s=0.0)
    assert any(a for *_, a in run(fly, 500, {"escape_takeoff": 0.9}))


def test_rear_limit():
    ang = np.arange(-180.0, 180.0, 4.0)
    clear = np.full(ang.size, 2.0)
    assert rear_limit(clear, ang, -0.2) == -0.2
    assert rear_limit(clear * 0 + 0.05, ang, 0.3) == 0.3     # forward: untouched
    clear[np.abs(ang) >= 170] = 0.05                         # something right behind
    assert rear_limit(clear, ang, -0.2) == 0.0
    clear[np.abs(ang) >= 170] = 0.19
    assert -0.2 < rear_limit(clear, ang, -0.2) < 0.0
