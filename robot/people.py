"""
The people Milo knows, by face, and how it meets new ones.

    camera process (robot/head.py)          brain client (sim/habitat_bridge/brain_client.py)
      faces (robot/faces.py)                  Social: greet a known face that looks at Milo;
      -> FaceTracker: tracks, who is who        ask an unknown one its name (spoken on the
         (PeopleBook), facing for how long       face page), learn the face from the answer
      <- "enroll track 3 as Ben", "snooze"  <-

PeopleBook keeps each person's face embeddings (robot/faces.py, SFace) in
~/milo/people/ on the robot only; nothing about a face is saved unless that
person gave their name. `python -m robot.people --list`, `--forget NAME`
(a running robot reloads the folder when it changes).

A face is someone Milo knows when most of its recent embeddings match one
person (cosine >= MATCH, SFace's published threshold) and unknown when several
good embeddings matched no one. Milo asks only when one face is facing it, an
unknown one, for ASK_AFTER_S, at most once per ASK_GAP_S. The answer (typed or
spoken on the face page) is read for a name ("I'm Ben", "my name is Ben",
"Ben"). "No" / "skip", or no answer, snoozes that face (kept in memory only,
for an hour / 15 minutes): Milo does not ask it again meanwhile. A name Milo
already knows is only added to if the face matches that person; otherwise it
asks for a full name.

C. APPROXIMATIONS: tracks follow a face by box overlap between detections
(~5 Hz), split when the embedding changes; "looking at Milo" is the head
facing the camera (robot/faces.facing). Names reach the language-model
personality only as one or two checked words.
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import secrets
import time

import numpy as np

PEOPLE_DIR = os.path.expanduser(os.environ.get("FLY_PEOPLE_DIR", "~/milo/people"))
MATCH = 0.363              # SFace cosine threshold for the same person (OpenCV Zoo)
SAME_TRACK = 0.2           # a face this unlike a track's recent faces is someone else
MAX_EMB = 40               # embeddings kept per person
WINDOW = 8                 # recent identifications a track decides from
VOTES = 3                  # ...of which this many agreeing ones name it
UNKNOWN_VOTES = 4          # ...or this many matching no one make it unknown
TRACK_GAP_S = 1.5          # a face unseen this long starts a new track
TRACK_KEEP_S = 60.0        # ...but its embeddings stay this long (to learn its name)
ASK_AFTER_S = 2.0          # an unknown face facing Milo this long: ask its name
ASK_GAP_S = 120.0          # at most one question per this long
ANSWER_S = 40.0            # wait this long for an answer
RETRIES = 2                # "didn't catch that" this many times, then give up
SNOOZE_DECLINED_S = 3600.0  # "no": do not ask that face again for this long
SNOOZE_UNANSWERED_S = 900.0
GREET_AFTER_S = 0.6        # a known face facing Milo this long: greet
GREET_GAP_S = 600.0        # greet each person at most once per this long
NAME_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|['\-]){0,19}$")   # letters (any script), ' and -
ANSWER_RE = re.compile(r"^(?:[^\W\d_]|[\s'\-.,!])*$")           # what an answer may contain (no "?")
LEAD = re.compile(r"\b(?:my name is|my name's|name's|name is|i am|i'm|im|it's|its|this is|call me|"
                  r"they call me)\b\s*(.*)$")
DECLINE = {"no", "nope", "nah", "none", "nobody", "skip", "pass", "cancel", "never mind", "nevermind",
           "no thanks", "no thank you", "not telling", "rather not", "i'd rather not", "stop", "go away",
           "none of your business", "i don't want to say", "i dont want to say"}
DECLINE_START = ("no ", "not ", "nope", "i don't", "i dont", "i do not", "rather not", "i'd rather",
                 "none of", "i won't", "i wont", "i'm not telling", "im not telling")
FILLER = {"the", "a", "an", "just", "here", "hi", "hello", "hey", "yes", "yeah", "ok", "okay", "um", "uh",
          "erm", "well", "so", "and", "is", "am", "it", "its", "it's", "me", "my", "name", "i", "i'm", "im",
          "call", "oh", "mr", "mrs", "ms", "dr"}
NOT_NAMES = {"what", "who", "why", "where", "when", "how", "which", "sorry", "pardon", "know", "there",
             "nothing", "none", "nobody", "someone", "somebody", "anyone", "everyone", "robot", "you", "your",
             "yours", "thanks", "thank", "please", "don't", "dont", "can't", "cant", "cannot", "not", "of",
             "business", "idea", "maybe", "later", "unknown", "guess", "secret", "busy", "fine", "good",
             "bye", "goodbye", "huh", "eh", "test", "testing", "stupid", "shut", "up"}


def slug(name: str) -> str:
    """A file name for a person: readable, and unique per exact name."""
    s = re.sub(r"[\W_]+", "-", name.lower()).strip("-") or "person"
    return f"{s}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def name_ok(name: str) -> bool:
    """One or two words of letters (as parse_name returns)."""
    w = str(name or "").split()
    return 1 <= len(w) <= 2 and all(NAME_RE.match(x) for x in w)


def parse_name(text: str, robot: str = "Milo"):
    """A name from an answer, "decline", or None (not understood)."""
    text = str(text or "").strip()
    low = " ".join(re.sub(r"[.,!]+", " ", text.lower()).split())
    if not low:
        return None
    if low in DECLINE or low.startswith(DECLINE_START):
        return "decline"
    if not ANSWER_RE.match(text):
        return None                                   # digits, symbols, a question
    m = LEAD.search(low)
    rest = (m.group(1) if m else low).split()
    if m and (" ".join(rest) in DECLINE or " ".join(rest).startswith(DECLINE_START)):
        return "decline"                              # "I'm not telling"
    if len(rest) > 6:
        return None
    words = [w for w in rest if w not in FILLER][:2]
    if (not words or not all(NAME_RE.match(w) for w in words)
            or any(w in NOT_NAMES or w == robot.lower() for w in words)):
        return None
    return " ".join(re.sub(r"(^|['\-])(\w)", lambda q: q.group(1) + q.group(2).upper(), w) for w in words)


def medoid(embs) -> np.ndarray | None:
    """The embedding most like the others."""
    e = np.asarray(embs, np.float32).reshape(-1, 128)
    if not len(e):
        return None
    return e[int(np.argmax((e @ e.T).sum(1)))]


# ------------------------------------------------------------------ the book
class PeopleBook:
    """Known people: name -> L2-normalised face embeddings (N x 128)."""

    def __init__(self, path: str = PEOPLE_DIR):
        self.path = path
        self._load()

    def _mtime(self) -> int:
        try:
            return os.stat(self.path).st_mtime_ns
        except FileNotFoundError:
            return 0

    def _load(self) -> None:
        self.people = {}                         # name -> {"emb", "created", "seen", "file"}
        self._seen_mtime = self._mtime()
        self._checked = time.monotonic()
        if os.path.isdir(self.path):
            for f in sorted(os.listdir(self.path)):
                if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                    try:
                        z = np.load(os.path.join(self.path, f), allow_pickle=False)
                        self.people[str(z["name"])] = {"emb": z["emb"].astype(np.float32), "file": f,
                                                       "created": float(z["created"]), "seen": float(z["seen"])}
                    except Exception:
                        continue

    def reload_if_changed(self, every_s: float = 2.0) -> bool:
        """Reload when the folder changed outside this process (--forget)."""
        now = time.monotonic()
        if now - self._checked < every_s:
            return False
        self._checked = now
        if self._mtime() != self._seen_mtime:
            self._load()
            return True
        return False

    def names(self) -> list:
        return sorted(self.people)

    def identify(self, e) -> tuple:
        """(name or None, best similarity) for one embedding."""
        best, sim = None, -1.0
        for n, p in self.people.items():
            s = float(np.max(p["emb"] @ np.asarray(e, np.float32)))
            if s > sim:
                best, sim = n, s
        return (best if sim >= MATCH else None), sim

    def matches(self, name: str, e) -> bool:
        p = self.people.get(name)
        return p is not None and float(np.max(p["emb"] @ np.asarray(e, np.float32))) >= MATCH

    def add(self, name: str, embs) -> int:
        """Add embeddings to a person (new or known); returns how many are kept."""
        embs = np.asarray(embs, np.float32).reshape(-1, 128)
        p = self.people.get(name)
        now = time.time()
        allv = embs if p is None else np.concatenate([p["emb"], embs])
        while len(allv) > MAX_EMB:               # drop the most redundant embedding
            sim = allv @ allv.T
            np.fill_diagonal(sim, -1.0)
            allv = np.delete(allv, int(np.argmax(sim.max(1))), axis=0)
        self.people[name] = {"emb": allv, "created": now if p is None else p["created"], "seen": now,
                             "file": p["file"] if p is not None else slug(name) + ".npz"}
        self._save(name)
        return len(allv)

    def forget(self, name: str) -> bool:
        p = self.people.pop(name, None)
        if p is None:
            return False
        try:
            os.remove(os.path.join(self.path, p["file"]))
        except FileNotFoundError:
            pass
        self._seen_mtime = self._mtime()
        return True

    def _save(self, name: str) -> None:
        os.makedirs(self.path, mode=0o700, exist_ok=True)
        p = self.people[name]
        dst = os.path.join(self.path, p["file"])
        tmp = dst[:-4] + ".tmp.npz"
        np.savez(tmp, name=np.array(name), emb=p["emb"], created=p["created"], seen=p["seen"])
        os.replace(tmp, dst)
        self._seen_mtime = self._mtime()


# ------------------------------------------------------------------ tracks (camera process)
def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class FaceTracker:
    """Faces over time: who each one is and how long it has faced the camera."""

    def __init__(self, book: PeopleBook):
        self.book = book
        self.tracks = {}
        self._next = 1
        self.snoozed = []                        # (embedding, until t): faces not to ask, in memory only

    def update(self, faces: list, t: float) -> list:
        """faces: robot.faces.Faces() dicts (box, facing, embedding, ...).
        Returns the live tracks: {id, box, who (name, "?" unknown, or None
        undecided), sim, facing, facing_s, snoozed}."""
        if self.book.reload_if_changed():
            for tr in self.tracks.values():      # someone forgotten: decide afresh
                if tr["who"] not in (None, "?") and tr["who"] not in self.book.people:
                    tr.update(who=None, named=False)
                    tr["window"].clear()
        self.snoozed = [(e, u) for e, u in self.snoozed if u > t]
        live = [k for k, tr in self.tracks.items() if t - tr["t"] <= TRACK_GAP_S]
        used = set()
        for f in sorted(faces, key=lambda q: -q["score"]):
            e = f.get("embedding")
            best, bi = None, 0.25
            for k in live:
                if k in used:
                    continue
                tr = self.tracks[k]
                v = _iou(f["box"], tr["box"])
                if v > bi and (e is None or not tr["embs"]
                               or float(np.mean(np.asarray(tr["embs"])[-5:] @ e)) >= SAME_TRACK):
                    best, bi = k, v
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = {"id": best, "t0": t, "embs": collections.deque(maxlen=20),
                                     "window": collections.deque(maxlen=WINDOW), "who": None,
                                     "sim": 0.0, "facing_since": None, "named": False, "snoozed": False}
            used.add(best)
            tr = self.tracks[best]
            tr.update(box=f["box"], t=t, facing=bool(f["facing"]))
            if f["facing"]:
                tr["facing_since"] = tr["facing_since"] if tr["facing_since"] is not None else t
            else:
                tr["facing_since"] = None
            if e is not None and f["facing"]:
                e = np.asarray(e, np.float32)
                tr["embs"].append(e)
                if any(float(s @ e) >= MATCH for s, _ in self.snoozed):
                    tr["snoozed"] = True
                if not tr["named"]:
                    name, sim = self.book.identify(e)
                    tr["sim"] = sim
                    tr["window"].append(name)
                    tr["who"] = self._decide(tr)
        for k in list(self.tracks):
            if t - self.tracks[k]["t"] > TRACK_KEEP_S:
                del self.tracks[k]
        return [self.view(tr, t) for tr in self.tracks.values() if t - tr["t"] <= TRACK_GAP_S]

    @staticmethod
    def _decide(tr):
        w = list(tr["window"])
        named = collections.Counter(x for x in w if x is not None)
        if named:
            name, n = named.most_common(1)[0]
            if n >= VOTES and n >= 0.6 * len(w):
                return name
        if w.count(None) >= UNKNOWN_VOTES and (not named or named.most_common(1)[0][1] < VOTES):
            return "?"
        return None

    @staticmethod
    def view(tr, t) -> dict:
        fs = 0.0 if tr["facing_since"] is None else t - tr["facing_since"]
        return {"id": tr["id"], "box": [round(float(x), 1) for x in tr["box"]], "who": tr["who"],
                "sim": round(float(tr["sim"]), 3), "facing": tr["facing"], "facing_s": round(fs, 2),
                "embs": len(tr["embs"]), "snoozed": tr["snoozed"]}

    def _consistent(self, tr) -> list:
        """The track's embeddings that agree with its main face (a track that
        briefly took in someone else's face does not teach that face)."""
        m = medoid(tr["embs"])
        return [] if m is None else [e for e in tr["embs"] if float(e @ m) >= MATCH]

    def enroll(self, track_id: int, name: str) -> dict:
        """Learn a track's face as `name` (the answer to Milo's question)."""
        fail = {"event": "enroll_failed", "track": track_id, "name": name}
        tr = self.tracks.get(int(track_id))
        if tr is None:
            return dict(fail, why="face gone")
        embs = self._consistent(tr)
        if len(embs) < 2:
            return dict(fail, why="too few views")
        if name in self.book.people and not self.book.matches(name, medoid(embs)):
            return dict(fail, why="name taken")   # someone else already has this name
        n = self.book.add(name, embs)
        tr.update(who=name, named=True)
        return {"event": "enrolled", "track": int(track_id), "name": name, "kept": n}

    def snooze(self, track_id: int, seconds: float, t: float) -> dict:
        """Do not ask this face again for a while (in memory only)."""
        tr = self.tracks.get(int(track_id))
        m = medoid(tr["embs"]) if tr is not None else None
        if m is not None:
            self.snoozed.append((m, t + float(seconds)))
            tr["snoozed"] = True
        return {"event": "snoozed", "track": int(track_id), "ok": m is not None}

    def forget(self, name: str) -> dict:
        ok = self.book.forget(name)
        for tr in self.tracks.values():
            if tr["who"] == name:
                tr.update(who=None, named=False)
                tr["window"].clear()
        return {"event": "forgotten", "name": name, "ok": ok}

    def command(self, c: dict, t: float) -> dict | None:
        """A command from the brain client (robot/head.py's stdin)."""
        cmd = c.get("cmd")
        name = str(c.get("name", ""))
        if cmd == "enroll":
            if not name_ok(name):
                return {"event": "enroll_failed", "track": c.get("track"), "name": name, "why": "bad name"}
            return self.enroll(int(c.get("track", -1)), name)
        if cmd == "snooze":
            return self.snooze(int(c.get("track", -1)), min(float(c.get("s", 600.0)), 86400.0), t)
        if cmd == "forget":
            return self.forget(name)
        return None


# ------------------------------------------------------------------ social (brain client)
class Social:
    """Greets the people Milo knows and asks the ones it does not. Each step:
    step(t, faces, replies, camera_events) -> {"say", "ask", "commands",
    "events"}: words for the face page to speak, the open question (shown with
    an answer box; its id is random, so only the page showing it can answer),
    commands for the camera process (enroll, snooze), and lines for the
    personality. t is a clock that never goes back (time.monotonic())."""

    Q = "Hi! I don't think we've met. What's your name?"

    def __init__(self, robot_name: str = "Milo"):
        self.robot = robot_name
        self.pending = None                      # {"id", "track", "t", "tries"}
        self.last_ask_t = -1e9
        self.greeted = {}                        # name -> t
        self.met = set()                         # names learned this run
        self.stats = collections.Counter()
        self._rng = secrets.SystemRandom()

    def _ask(self, t, track, tries=0) -> dict:
        self.pending = {"id": self._rng.randrange(1, 2 ** 31 - 1), "track": track, "t": t, "tries": tries}
        return {"id": self.pending["id"], "q": "What's your name?"}

    def step(self, t: float, faces: list, replies: list = (), camera_events: list = ()) -> dict:
        out = {"say": None, "ask": None, "commands": [], "events": []}
        for ev in camera_events:
            if ev.get("event") == "enrolled":
                self.met.add(ev.get("name"))
                self.greeted[ev.get("name")] = t
                self.stats["learned"] += 1
                out["events"].append(f"you just met {ev.get('name')} and learned their face")
            elif ev.get("event") == "enroll_failed":
                self.stats["enroll_failed"] += 1
                if ev.get("why") == "name taken" and self.pending is None:
                    out["say"] = f"I already know someone called {ev.get('name')}. What's your full name?"
                    out["ask"] = self._ask(t, ev.get("track"))
                    return out
                out["say"] = "Sorry, I didn't get a good look at you. Ask me again in a little while!"
                self.last_ask_t = t - ASK_GAP_S + 30.0          # not straight away
        if self.pending is not None:
            for r in replies:
                if r.get("id") != self.pending["id"]:
                    continue
                name = parse_name(r.get("text", ""), self.robot)
                if name == "decline":
                    out["commands"].append({"cmd": "snooze", "track": self.pending["track"], "s": SNOOZE_DECLINED_S})
                    out["say"] = "Okay, no problem."
                    self.stats["declined"] += 1
                    self.pending = None
                elif name:
                    out["commands"].append({"cmd": "enroll", "track": self.pending["track"], "name": name})
                    out["say"] = f"Nice to meet you, {name}!"
                    self.stats["named"] += 1
                    self.pending = None
                elif self.pending["tries"] < RETRIES:
                    out["say"] = "Sorry, I didn't catch that. What's your name?"
                    out["ask"] = self._ask(t, self.pending["track"], self.pending["tries"] + 1)
                    return out
                else:
                    out["commands"].append({"cmd": "snooze", "track": self.pending["track"], "s": SNOOZE_UNANSWERED_S})
                    out["say"] = "Never mind. Nice to see you anyway!"
                    self.stats["not_understood"] += 1
                    self.pending = None
                break
            if self.pending is not None and t - self.pending["t"] > ANSWER_S:
                out["commands"].append({"cmd": "snooze", "track": self.pending["track"], "s": SNOOZE_UNANSWERED_S})
                self.stats["unanswered"] += 1
                self.pending = None
            if self.pending is not None:
                out["ask"] = {"id": self.pending["id"], "q": "What's your name?"}
                return out
        if out["say"]:
            return out
        facing = [f for f in faces if f.get("facing")]
        for f in sorted(facing, key=lambda q: -q.get("facing_s", 0.0)):
            who, fs = f.get("who"), f.get("facing_s", 0.0)
            if who not in (None, "?") and fs >= GREET_AFTER_S and t - self.greeted.get(who, -1e9) > GREET_GAP_S:
                self.greeted[who] = t
                out["say"] = f"Hi {who}!"
                out["events"].append(f"{who} is looking at you")
                self.stats["greeted"] += 1
                return out
        # ask only when exactly one face is facing Milo (so it is clear who is asked)
        if len(facing) == 1:
            f = facing[0]
            if (f.get("who") == "?" and f.get("facing_s", 0.0) >= ASK_AFTER_S and not f.get("snoozed")
                    and t - self.last_ask_t > ASK_GAP_S):
                self.last_ask_t = t
                out["say"] = self.Q
                out["ask"] = self._ask(t, f["id"])
                out["events"].append("someone you do not know is looking at you; you asked their name")
                self.stats["asked"] += 1
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="the people Milo knows (faces kept on this robot only)")
    ap.add_argument("--dir", default=PEOPLE_DIR)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--forget", metavar="NAME")
    a = ap.parse_args()
    book = PeopleBook(a.dir)
    if a.forget:
        print("forgotten" if book.forget(a.forget) else "not known:", a.forget)
    for n in book.names():
        p = book.people[n]
        print(f"{n}: {len(p['emb'])} views, met {time.strftime('%Y-%m-%d', time.localtime(p['created']))}, "
              f"last seen {time.strftime('%Y-%m-%d %H:%M', time.localtime(p['seen']))}")
    if not book.names():
        print("nobody yet")


if __name__ == "__main__":
    main()
