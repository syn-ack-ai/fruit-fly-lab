"""robot/seeing.py: the memory of seen views, novelty and habituation, the
curiosity target; and its place in robot/head.ObjectEncoder's choice."""
import numpy as np

import robot.seeing as S
from tests.test_head_motion import _encoder


def unit(seed, base=None, mix=0.0):
    v = np.random.default_rng(seed).standard_normal(S.DIM).astype(np.float32)
    if base is not None:
        v = mix * base + (1 - mix) * v / np.linalg.norm(v)
    return v / np.linalg.norm(v)


def look(cur, t, seq, emb, pan=0.0, pers=None):
    return cur.step(t, {"seq": seq, "emb": emb, "pan": pan, "el": 0.0}, pers)


class _Pers:
    def __init__(self):
        self.ev = []

    def event(self, t, text, urge=False):
        self.ev.append(text)


class _Labels:
    def top(self, e):
        return "a cat", 0.2


def test_novelty_habituates_and_comes_back_for_something_new():
    room = unit(1)
    cur = S.Curiosity(60.0)
    emb = np.stack([room] * 4)
    nov = []
    for k in range(40):
        look(cur, 0.5 * k, k + 1, emb)
        nov.append(cur.last["novelty"]["middle"])
    assert nov[0] == 1.0 and nov[5] > S.NOVEL_MIN and nov[-1] < 0.2          # interest for a few s, then used to it
    assert len(cur.mem) == 1                                                  # four alike views: one memory
    cat = unit(2)
    t = look(cur, 21.0, 100, np.stack([room, cat, room, room]), pan=10.0)    # something new on the left
    assert cur.last["side"] == "left" and t is not None
    assert abs(t["az"] - (10.0 - 0.25 * 60.0)) < 1e-6 and t["novelty"] > 0.9
    assert look(cur, 21.0, 100, np.stack([room, cat, room, room])) is t       # the same look: unchanged
    near = unit(3, base=room, mix=0.97)                                        # the room, a little changed
    look(cur, 22.0, 101, np.stack([near] * 4))
    assert cur.last["novelty"]["middle"] < 0.3


def test_personality_told_once_in_a_while_and_memory_saved(tmp_path):
    cur = S.Curiosity(60.0, labels=_Labels(), state_path=str(tmp_path))
    pers = _Pers()
    look(cur, 0.0, 1, np.stack([unit(k) for k in range(4)]), pers=pers)
    look(cur, 1.0, 2, np.stack([unit(k + 10) for k in range(4)]), pers=pers)
    assert len(pers.ev) == 1 and "a cat" in pers.ev[0]                        # at most every TELL_EVERY_S
    cur.save()
    again = S.Curiosity(60.0, state_path=str(tmp_path))
    assert len(again.mem) == len(cur.mem) == 8
    look(again, 0.0, 1, np.stack([unit(k) for k in range(4)]))
    assert max(again.last["novelty"].values()) < 1.0                          # remembered across days


def test_memory_capacity_forgets_the_longest_unseen():
    m = S.Memory(capacity=3)
    for k in range(5):
        m.learn(unit(k), float(k))
    assert len(m) == 3 and m.novelty(unit(0)) == 1.0 and m.novelty(unit(4)) < 1.0


def test_novel_target_ranks_after_person_and_toy():
    enc = _encoder()
    enc.feed.s = {"ready": True, "stale": False, "moving": 0.0, "novel_active": 1.0, "novel_az": 30.0,
                  "novel_el": 0.0, "novel_half_deg": 8.0, "novelty": 0.8}
    r = enc.rates_hz(0.0)
    assert enc.last["target"] == "novel" and r[2] > r[0]
    enc.curiosity = 0.0
    assert not enc.rates_hz(0.0).any()
    enc.curiosity = 1.0
    enc.feed.s.update(person_active=1.0, person_stale=False, person_t=0.0, person_az=-30.0,
                      person_el=0.0, person_half_deg=6.0)
    enc.rates_hz(0.0)
    assert enc.last["target"] == "person"
