"""Who bumped whom (sim/habitat_bridge/bump_analysis.py) reads the contact step.
Regression (review 2026-09-26): it used the pet's speed from two steps before."""
from sim.habitat_bridge.bump_analysis import classify


def _log():
    # brain_client: a step starts at t_sim, sends v, then stamps its log entry
    # with t_sim + 0.1 and the poses after the step. Person still at x = 1.
    # The pet stands until step 42 (t = 4.2 .. 4.3), where it drives at 0.3 m/s
    # toward the person (+x, yaw 0) and contact begins.
    log = []
    for k in range(50):
        v = 0.3 if k == 42 else 0.0
        log.append({"t": round(0.1 * (k + 1), 2), "v": v, "robot": [0.3, 0.0, 0.0],
                    "human": [1.0, 0.0], "az": 0.0})
    return log


def test_pet_speed_is_taken_from_the_contact_step():
    c = classify(_log(), 4.2)                   # bump stored at the step's start
    assert c["pet_closing"] > 0.25 and c["pet_fault"]


def test_a_standing_pet_is_not_blamed():
    log = _log()
    for e in log:
        e["v"] = 0.0
    log[42]["human"] = [0.9, 0.0]               # the person steps into the pet
    c = classify(log, 4.2)
    assert not c["pet_fault"] and c["person_closing"] > 0.5
