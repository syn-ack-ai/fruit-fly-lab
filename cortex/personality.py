"""
The pet's personality: a local language model on top of the neocortex.

    heard speech, events, drives ─> digest ─> LLM (slow, event-driven) ─> intention
                                                                       ─> sound / words
                                                                       ─> praise or scolding
    intention  -> biases the neocortex's goal choice (cortex/v0.py); never motors
    feedback   -> reward / punishment for the neocortex's critic, whose error is the
                  dopamine sent to the fly's mushroom body: words can teach the connectome

Layers and who wins:
  fly brain (connectome)  moves the body, eats, startles, pursues   (10 kHz)
  neocortex v0            map, drives, critic, manners              (10 Hz)
  personality (this)      speech, intentions, voice, diary          (~ one call per event, ~2 s)
  safety layers           lidar brake, speed limit near people, emergency return; nothing above changes them
An intention only nudges the neocortex's choice, weighted by the matching drive:
a pet that is full is not made to eat because the model said so.

The model is reached through an OpenAI-compatible endpoint: NVIDIA's Personal
AI Router (PAIR) on the machine running the brain (port 1234; it used 1236 while another program held 1234) routes it to the
cluster (LM Studio on the Mac Studio, so the fly brain keeps the local GPU).
Default model: Gemma 4 E4B (MLX 4-bit): 0.6 s per decision, praise and scolding
read correctly in every test (cortex/llm_bench.py, 2026-09-26: vs Qwen3.5-9B
1.2 s and Qwen3.8-27B 3.7 s, all 54-56/60 checks).
Calls run in a background thread; while one is pending the pet carries on on
its drives. If the model is unreachable the pet simply has no personality layer.

Safety note: speech and the model's own diary notes (saved, fed back on later
days) reach the prompt, so whoever the microphone hears can steer intentions
and praise/scold (dopamine). Replies are restricted to fixed lists and short
phrases (parse), intentions only nudge the drives, and nothing here reaches the
motors or the safety layer; still, treat the diary as untrusted text.

C. APPROXIMATIONS: not a fly structure. The analogy is a cortex-like layer for
language and social behaviour (Habitat: speech arrives as ground-truth text).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

INTENTS = ("seek_person", "follow", "rest", "explore", "eat", "give_space", "none")
MOODS = ("happy", "eager", "content", "curious", "sleepy", "hungry", "grumpy", "startled", "playful", "calm")
SOUNDS = ("beep", "boop", "chirp", "trill", "whirr", "buzz", "none")   # robot sounds (robot/face_page)
INTENT_S = 20.0              # an intention lasts this long unless replaced
IDLE_S = 30.0                # with nothing happening, reconsider this often
MIN_GAP_S = 1.0              # at most one call per this (simulated) time
URGE_S = 6.0                 # words only within this long of a vocal urge (urge_gate)

SYSTEM = """You are {name}, the personality of a small home robot. You are a robot and
say so plainly: never claim or pretend to be a cat, a dog, a fly, a person or any
living animal. What makes you unusual is your lower brain: a simulation of the complete
wiring diagram (connectome) of a real fruit fly, a once-living animal, runs your body:
walking, steering, docking to charge, startle. A neocortex layer keeps a map and your
drives. You sense the world with a camera, a lidar and bumpers; you have no sense of
smell, so never speak or write of smells. If someone asks what you are, or whether you are an
animal ("are you a cat?"), answer in a few words that you are a robot. You only choose intentions, sounds and a few words; the body decides the details.
Your status uses the fly brain's words for robot things: "food" is your charging dock,
"hunger" is how empty your battery is, and "eat" means dock and charge (the person's
"dinner" is charging too). Hungry and at the food: eat.
Your DRIVES decide what you want (hunger, company, sleep). The "Fly brain" line reports
what your real fly-brain neurons are doing right now (escape, song, steering, walking,
the learning centre's surprise signal): let it colour your mood and words -- startled when
the escape neurons fire, bright when your song command fires (you are singing) -- and
never claim a feeling or neuron it does not show.
Character: {character}
Stay true to the drives you are given: with a full battery you do not want to charge; if
"wants company" is low (below 0.2) you do not seek or follow your person -- give space,
rest or explore -- even when petted. You do not always obey. Your person rarely
takes you to the charger: when hungry, go to where you remember food ("eat").
Reply with ONLY one JSON object, no other text:
{{"intent": one of {intents},
  "sound": one of {sounds},
  "say": a very short phrase (<= 6 words) or null,
  "mood": one of {moods},
  "feedback": -1 if the person just scolded you in words, +1 if they praised you in words, else 0
              (being petted or given a treat is not praise: those are rewarded already),
  "note": a few words for your diary, or null}}"""

DEFAULT_CHARACTER = ("affectionate but independent, endlessly curious about new places and things, "
                     "a little vain; talks rarely, mostly beeps, chirps and whirrs")


class Personality:
    def __init__(self, url: str = "http://127.0.0.1:1234/v1/chat/completions",
                 model: str = "gemma-4-e4b-it-mlx", name: str = "Milo",
                 character: str = DEFAULT_CHARACTER, state_path: str | None = None,
                 timeout_s: float = 20.0, urge_gate: bool = False):
        self.url, self.model, self.name, self.timeout = url, model, name, timeout_s
        self.urge_gate = urge_gate
        self.system = SYSTEM.format(name=name, character=character, intents=list(INTENTS),
                                    sounds=list(SOUNDS), moods=list(MOODS))
        self.state_path = state_path
        self.diary = self._load_diary()
        self._lock = threading.Lock()
        self._pending = None                 # thread
        self._reply = None                   # (t_asked, dict, latency, ok) waiting to be taken
        self._gen = 0                        # day counter: replies to an earlier day are dropped
        self.log = []                        # every exchange this day
        self.reset(0)

    # ---------------------------------------------------------------- memory
    def _diary_file(self):
        return os.path.join(self.state_path, "diary.json") if self.state_path else None

    def _load_diary(self):
        f = self._diary_file()
        if f and os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)
        return []

    def reset(self, day: int) -> None:
        """A new day. A reply still in flight from the day before is dropped
        (its simulated clock restarts at 0, so it would otherwise never be due
        and would block every later call)."""
        with self._lock:
            self._gen += 1
            self._reply = None
        self.day = day
        self.current = {"intent": "none", "sound": "none", "say": None, "mood": "calm",
                        "feedback": 0, "t": -1e9}
        self.last_call_t = -1e9
        self._fresh = None                   # the newest reply, until the face takes it
        self.events = []                     # (t, text) since the last call
        self.log = []
        self.stats = {"calls": 0, "failures": 0, "latency_s": 0.0, "praise": 0, "scold": 0, "gated": 0}
        self.urge_t = -1e9

    def save(self) -> None:
        notes = [e["reply"].get("note") for e in self.log if e.get("reply") and e["reply"].get("note")]
        self.diary.append({"day": self.day, "notes": notes[-8:]})
        self.diary = self.diary[-14:]        # the last two weeks
        f = self._diary_file()
        if f:
            os.makedirs(self.state_path, exist_ok=True)
            with open(f, "w") as fh:
                json.dump(self.diary, fh, indent=1)

    # ------------------------------------------------------------------ loop
    def event(self, t_s: float, text: str, urge: bool = False) -> None:
        """Something worth reacting to (heard speech, petted, startled, found
        food). urge: a vocal urge -- the fly brain's song command or startle,
        or being spoken to / petted: words are allowed for URGE_S after it."""
        self.events.append((round(t_s, 1), text))
        if urge:
            self.urge_t = t_s

    def step(self, t_s: float, digest: dict) -> dict:
        """Every control step: collect a finished reply, maybe start a new call,
        and return the current personality output (fresh keys marked new=True)."""
        out = dict(self.current, new=False)
        with self._lock:
            rep = self._reply
            # in simulation (slower than real time) a reply takes effect only
            # once the model's real latency has passed in simulated time
            if rep is not None and t_s >= rep[0] + rep[2]:
                self._reply = None
            else:
                rep = None
        if rep is not None:
            t_asked, r, latency, ok, *more = rep
            rep_urge = more[0] if more else True
            self.stats["latency_s"] += latency
            if ok and r is not None:
                r = dict(r, t=t_s)                    # (the log keeps the model's own reply)
                if self.urge_gate and r.get("say") and not rep_urge:
                    # the fly brain decides WHEN Milo speaks: without a vocal
                    # urge when the call was made, the words stay unsaid (sounds
                    # and mood still apply). Judged at ASK time: a reply lands
                    # after the model's latency (review 2026-09-28)
                    r["gated_say"], r["say"] = r["say"], None
                    self.stats["gated"] += 1
                self.current = r
                self.stats["praise"] += r["feedback"] > 0
                self.stats["scold"] += r["feedback"] < 0
                out = dict(r, new=True)
                self._fresh = out
            else:
                self.stats["failures"] += 1
        if out["intent"] != "none" and t_s - self.current["t"] > INTENT_S:
            self.current["intent"] = out["intent"] = "none"
        with self._lock:
            waiting = self._reply is not None
        busy = (self._pending is not None and self._pending.is_alive()) or waiting
        due = self.events or t_s - self.last_call_t > IDLE_S
        if not busy and due and t_s - self.last_call_t >= MIN_GAP_S:
            self.last_call_t = t_s
            msg = self._message(t_s, digest)
            self.events = []
            urge = t_s - self.urge_t <= URGE_S
            self._pending = threading.Thread(target=self._ask, args=(t_s, msg, self._gen, urge), daemon=True)
            self._pending.start()
        return out

    def take_fresh(self) -> dict | None:
        """The newest reply once (for the face: words and sounds play once)."""
        f, self._fresh = self._fresh, None
        return f

    def wait(self, timeout: float | None = None) -> None:
        """Block until a pending call finishes (for simulated time, which does
        not wait for the model on its own)."""
        if self._pending is not None:
            self._pending.join(timeout)

    # --------------------------------------------------------------- the model
    def _message(self, t_s, d):
        recent = "; ".join(f"t={t} {x}" for t, x in self.events[-6:]) or "nothing new"
        past = " | ".join(f"day {e['day']}: " + ", ".join(e["notes"][-3:]) for e in self.diary[-3:] if e["notes"])
        lines = [f"Time {t_s:.0f} s into day {self.day}.",
                 f"Drives: hunger {d.get('hunger', 0):.2f}, wants company {d.get('social', 0):.2f}"
                 + (f", sleepy {d['sleepy']:.2f}" if d.get("sleepy") is not None else "") + " (0-1)."
                 + (f" Battery {d['battery']:.0%}." if d.get("battery") is not None else ""),
                 f"Body (fly brain): {d.get('behaviour', '?')}.",
                 f"Person: {d.get('person', 'not in view')}.",
                 f"Food: {d.get('food', 'unknown')}.",
                 f"Doing: {d.get('doing', '?')}. Your last mood: {self.current['mood']}.",
                 f"Fly brain: {d.get('brain', 'no readout')}.",
                 f"Just now: {recent}."]
        if past:
            lines.append(f"Diary, earlier days: {past}.")
        return "\n".join(lines)

    def _ask(self, t_s, msg, gen, urge=True):
        body = request_body(self.model, self.system, msg)
        t0 = time.time()
        reply, ok = None, False
        try:
            req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                text = json.load(fh)["choices"][0]["message"]["content"]
            reply = parse(text)
            ok = reply is not None
        except Exception as ex:                       # unreachable, timeout, bad reply
            text = f"error: {ex}"
        lat = time.time() - t0
        with self._lock:
            if gen != self._gen:              # asked on an earlier day
                return
            self.stats["calls"] += 1
            self.log.append({"t": round(t_s, 1), "asked": msg, "answer": text[:400], "reply": reply,
                             "latency_s": round(lat, 2)})
            self._reply = (t_s, reply, lat, ok, urge)


def request_body(model: str, system: str, msg: str) -> dict:
    """An OpenAI-style chat request with the model's reasoning switched off.
    LM Studio (2026-09) ignores enable_thinking / reasoning_effort for the small
    Qwen3.5 models, which then spend the whole reply thinking; starting the
    answer with an empty think block skips it (0.6 s instead of > 3 s with no
    answer). Gemma 4 only thinks when asked in its system prompt."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": msg}]
    if "qwen" in model.lower():
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
    return {"model": model, "temperature": 0.7, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            "messages": messages}


def parse(text: str) -> dict | None:
    """The model's JSON, validated; anything outside the allowed values is dropped."""
    a = text.find("{")
    if a < 0:
        return None
    try:
        r, _ = json.JSONDecoder().raw_decode(text[a:])      # the first object only
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(r, dict):
        return None
    say = r.get("say")
    say = str(say)[:60] if say not in (None, "", "null") else None
    try:
        fb = round(float(r.get("feedback", 0)))
    except (TypeError, ValueError, OverflowError):
        fb = 0
    note = r.get("note")
    return {"intent": r.get("intent") if r.get("intent") in INTENTS else "none",
            "sound": r.get("sound") if r.get("sound") in SOUNDS else "none",
            "say": say,
            "mood": r.get("mood") if r.get("mood") in MOODS else "calm",
            "feedback": max(-1, min(1, fb)),
            "note": str(note)[:80] if note not in (None, "", "null") else None}
