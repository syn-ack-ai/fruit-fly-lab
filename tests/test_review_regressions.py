"""Small reproductions of episode, dataset, and hardware lifecycle failures."""
import json
from unittest.mock import Mock

import numpy as np
import pytest

from cognition.sweep import complete_dataset
from cortex.v0 import CortexV0, FOOD_GIVE_UP_S
from fly.body.fly_body import FlyBody
from fly.world.world import Fruit, World, WorldConfig
from robot.head import HeadFeed, _F, _I
from sim.habitat_bridge.speech import ScriptedPerson
from simulation.engine.session import Session


def test_backward_command_does_not_keep_previous_turn():
    body = FlyBody()
    body.update(1, {"turn_bias": 0.5}, 1)
    assert body.state.turn_rate_deg_s != 0
    body.update(1, {"backward_walk": 0.8}, 2)
    assert body.state.turn_rate_deg_s == 0


def test_world_reset_counts_first_fruit_contact_again():
    world = World(WorldConfig(n_fruit=0, predators=False))
    world.fruits = [Fruit(0, 0, kind="ripe")]
    body = FlyBody()
    world.step(1, 1, body)
    assert world.stats["fruit_visits"] == 1
    world.reset()
    world.fruits = [Fruit(0, 0, kind="ripe")]
    world.step(1, 1, body)
    assert world.stats["fruit_visits"] == 1


def test_pending_speech_call_is_reported_without_mutating_history():
    person = ScriptedPerson()
    person.call_t = 58.0
    assert person.summary()["pending"] == 1
    assert person.summary()["calls"] == 0 and person.summary()["answered"] == 0
    assert person.calls == []


def test_head_close_requests_graceful_shutdown():
    feed = HeadFeed.__new__(HeadFeed)
    feed._h = np.zeros(len(_F), dtype=np.float64)
    feed.proc = Mock()
    feed.proc.poll.return_value = None
    feed.shm = Mock()
    header = feed._h
    feed.close()
    assert header[_I["cmd_stop"]] == 1.0
    assert feed.proc.wait.call_args.kwargs["timeout"] >= 6
    feed.proc.terminate.assert_not_called()
    assert feed.shm.close.called and feed.shm.unlink.called


def test_head_close_falls_back_if_worker_stalls():
    import subprocess

    feed = HeadFeed.__new__(HeadFeed)
    feed._h = np.zeros(len(_F), dtype=np.float64)
    feed.proc = Mock()
    feed.proc.poll.return_value = None
    feed.proc.wait.side_effect = [subprocess.TimeoutExpired("head", 6), None]
    feed.shm = Mock()
    feed.close()
    feed.proc.terminate.assert_called_once()
    feed.proc.kill.assert_not_called()


def test_sweep_rejects_missing_trial_shard(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "meta.json").write_text(json.dumps({"trials": 4}))
    np.savez(ds / "driven.npz", sensory=np.array([1]))
    np.savez(ds / "shard_000.npz", seed=np.array([0, 2]))
    with pytest.raises(ValueError, match="missing or duplicated"):
        complete_dataset(ds, 4)
    np.savez(ds / "shard_001.npz", seed=np.array([1, 3]))
    assert complete_dataset(ds, 4)


def test_timed_out_food_goal_is_not_selected_again():
    cx = CortexV0.__new__(CortexV0)
    cx.goal = ((1, 0), "food", 1.0, 0.0)
    cx.food_avoid = {}
    cx._utilities = lambda here, t: ({(1, 0): (2.0, "food"), here: (0.1, "explore")}, {})
    cx.day_log = {"goals": {}}
    cx._plan((0, 0), FOOD_GIVE_UP_S + 1)
    assert cx.goal is None or cx.goal[0] != (1, 0)
    assert (1, 0) in cx.food_avoid


def test_session_reset_clears_episode_activity_but_keeps_learning(monkeypatch):
    monkeypatch.setattr("simulation.engine.session.SpikeRecorder", Mock())
    session = Session.__new__(Session)
    session._native = False
    session.engine = Mock()
    session.c = Mock()
    session.readout = Mock()
    session.window_ms = 50
    session._total_spikes = 10
    session.history = [{}]
    session._raster = [1]
    session.body_track = [1]
    session.encoders = []
    session.body = Mock()
    session.world = Mock()
    session.world_senses = Mock()
    session.world_senses._jo_adapt = np.ones(4)
    session.mb = Mock()
    session.nav = Mock()
    session.reset(seed=7)
    session.body.reset.assert_called_once()
    session.mb.reset_activity.assert_called_once()
    session.nav.reset.assert_called_once()
    session.world.reset.assert_called_once_with(7)
    assert session.world_senses._B is None
    assert not session.world_senses._jo_adapt.any()


def test_give_up_blocks_food_goal_only():
    cx = CortexV0.__new__(CortexV0)
    cx.goal = None
    cx.food_avoid = {(1, 0): 100.0}
    cx._utilities = lambda here, t: ({(1, 0): (2.0, "owner"), here: (0.1, "explore")}, {})
    cx.day_log = {"goals": {}}
    cx._plan((0, 0), 10.0)
    assert cx.goal is not None and cx.goal[0] == (1, 0) and cx.goal[1] == "owner"
