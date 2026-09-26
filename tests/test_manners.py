"""Cat-like manners around the person (cortex/v0.py) and the robot's
near-person speed limit (robot/safety.py)."""
import math

import pytest

from cortex.topdown import bearing_deg
from cortex.v0 import CortexV0, SEEK_SOCIAL
from robot.safety import (CONTACT_M, FREE_M, HEAD_ABOVE_CAM_M, TARGET_HALF_WIDTH_M, V_CONTACT, V_FREE,
                          ProximityGovernor, person_distance, speed_limit)


def half_for(d):
    return math.degrees(math.atan2(TARGET_HALF_WIDTH_M, math.hypot(d, HEAD_ABOVE_CAM_M)))


def test_person_distance_inverts_apparent_size():
    for d in (0.4, 1.0, 2.5):
        assert person_distance(half_for(d)) == pytest.approx(d, abs=1e-6)
    assert person_distance(0.0) is None


def test_speed_limit_crawls_at_contact_and_is_free_far_away():
    assert speed_limit(None) == V_FREE
    assert speed_limit(0.2) == V_CONTACT
    assert speed_limit(CONTACT_M) == V_CONTACT
    assert speed_limit(FREE_M + 1) == V_FREE
    xs = [speed_limit(d) for d in (0.6, 1.0, 1.5, 1.9)]
    assert xs == sorted(xs) and V_CONTACT < xs[0] < xs[-1] < V_FREE


def test_governor_limits_forward_only_and_forgets():
    g = ProximityGovernor()
    g.observe(True, half_for(0.4), 0.0)
    assert g.limit(0.4, 0.1) == V_CONTACT
    assert g.limit(-0.3, 0.1) == -0.3                  # backing away is never limited
    assert g.limit(0.4, 5.0) == 0.4                    # an old sighting no longer counts


def manners_cortex(social, person_xz, person_vel=(0.0, 0.0)):
    cx = CortexV0.__new__(CortexV0)                    # no critic needed for manners
    cx.social = social
    cx.person_seen = (person_xz[0], person_xz[1], 10.0)
    cx.person_vel = person_vel
    cx.day_log = {"manners_s": {}}
    cx.intent = "none"
    return cx


def test_content_pet_steps_back_from_a_still_person():
    cx = manners_cortex(SEEK_SOCIAL - 0.1, (0.5, 0.0))
    cmd = cx._manners({"goal_deg": None, "goal_gain": 0.0}, 0.0, 0.0, 10.1, 0.1)
    assert cx.manner == "give_space"
    assert cmd["goal_deg"] == pytest.approx(bearing_deg((0, 0), (-1, 0)))   # away from the person
    assert cmd["arousal"] < 0.5


def test_pet_that_wants_company_may_come_close():
    cx = manners_cortex(SEEK_SOCIAL + 0.2, (0.3, 0.0))
    cmd = cx._manners({"goal_deg": 0.0, "goal_gain": 0.8}, 0.0, 0.0, 10.1, 0.1)
    assert cx.manner == "greet"
    assert cmd["goal_deg"] == 0.0 and cmd["goal_gain"] == 0.8 and cmd["arousal"] == 1.0


def test_never_in_front_of_a_walking_person():
    # person at x=1 walking toward the pet (-x): step out sideways, stop chasing
    for social in (0.1, 0.9):
        cx = manners_cortex(social, (1.0, 0.1), person_vel=(-0.7, 0.0))
        cmd = cx._manners({"goal_deg": 0.0, "goal_gain": 1.0}, 0.0, 0.0, 10.1, 0.1)
        assert cx.manner == "yield"
        assert cmd["arousal"] == 0.0
        # sideways, to the side the pet is already on (z < person's z -> -z)
        assert cmd["goal_deg"] == pytest.approx(bearing_deg((0, 0), (0, -1)))


def test_following_behind_a_walking_person_is_fine():
    cx = manners_cortex(0.9, (1.0, 0.0), person_vel=(0.7, 0.0))   # walking away from the pet
    cmd = cx._manners({"goal_deg": 0.0, "goal_gain": 1.0}, 0.0, 0.0, 10.1, 0.1)
    assert cx.manner == "follow" and cmd["goal_gain"] == 1.0
