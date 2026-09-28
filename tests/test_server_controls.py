"""Exercise the laboratory controls without a connectome or camera hardware."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from brain.sensory.camera import CameraLoomingStimulus
from visualization import server


def _runner():
    runner = server.Runner.__new__(server.Runner)
    runner.lock = threading.Lock()
    runner.running = False
    runner.camera = runner._camera_stim = runner.head = None
    runner.connectome = object()
    runner.session = SimpleNamespace(encoders=[], engine=Mock(), paused=False)
    runner.session.add_stimulus = lambda enc, stim: runner.session.encoders.append((enc, stim))
    runner.session.clear_stimuli = runner.session.encoders.clear
    return runner


def _post(path):
    async def request():
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await server.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                          "method": "POST", "scheme": "http", "path": path,
                          "raw_path": path.encode(), "query_string": b"", "root_path": "",
                          "headers": [], "client": ("127.0.0.1", 1234), "server": ("test", 80)},
                         receive, send)
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
        return status, json.loads(body)

    return asyncio.run(request())


def test_clear_endpoint_removes_stimuli_without_starting_simulation(monkeypatch):
    runner = _runner()
    runner.session.add_stimulus(object(), object())
    monkeypatch.setattr(server, "RUNNER", runner)

    assert _post("/api/stimulus/clear") == (200, {"ok": True})
    assert runner.session.encoders == []
    assert not runner.running
    assert _post("/api/stimulus/unknown")[1]["ok"] is False


def _dead_camera(runner):
    old = Mock()
    old.alive = False
    runner.camera = old
    runner._camera_stim = CameraLoomingStimulus(old)
    runner.session.add_stimulus(object(), runner._camera_stim)
    return old


def test_restarting_dead_camera_removes_old_feed_and_encoder(monkeypatch):
    runner = _runner()
    old = _dead_camera(runner)
    unrelated = (object(), object())
    runner.session.encoders.append(unrelated)
    new = Mock()
    new.alive = True
    new.state.return_value = {"frames": 1}
    monkeypatch.setattr("brain.sensory.camera.CameraFeed", Mock(return_value=new))
    monkeypatch.setattr("brain.sensory.encoders.LoomingEncoder", Mock(return_value=object()))
    monkeypatch.setattr("brain.sensory.retinotopy.load_retinotopy", Mock())

    assert runner.camera_on()["ok"]
    old.close.assert_called_once()
    assert len(runner.session.encoders) == 2
    assert runner._camera_stim.feed is new
    runner.camera_off()
    assert runner.session.encoders == [unrelated]
    new.close.assert_called_once()


def test_failed_camera_restart_still_releases_dead_feed(monkeypatch):
    runner = _runner()
    old = _dead_camera(runner)
    new = Mock()
    new.alive = False
    new.error.return_value = "camera unavailable"
    monkeypatch.setattr("brain.sensory.camera.CameraFeed", Mock(return_value=new))

    assert not runner.camera_on()["ok"]
    old.close.assert_called_once()
    new.close.assert_called_once()
    assert runner.camera is None and runner._camera_stim is None
    assert runner.session.encoders == []
