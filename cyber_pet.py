"""
Cyber-Pet Simulator v2 (FSM) - PyGame edition
=============================================

Setup (once):   pip install pygame
Run:            python cyber_pet.py      (or press ▶ "Run Python File" in VS Code)

Controls
  Click a button or press 1-8   ->  Kick, Hug, Play, Food, Candy, Beer, Medicine, Bed
  Hover a button                ->  preview where that action would take the pet right now
  Space                         ->  pause / resume time
  R                             ->  reset
  Esc / close window            ->  quit

How the state machine works
---------------------------
The pet is a finite state machine extended with four internal stats
(food, energy, fun, health; 0-100, higher is better). Three things move it:

  1. INPUTS (your actions). The transition function
         resolve_transition(state, action, stats) -> (next_state, guard)
     looks up TRANSITIONS[state][action]. An entry is either
       * a plain state name:            "Angry"
       * a list of guarded rules:       [("tired", "Sleepy"), (None, "Playful")]
         The first rule whose guard is true wins; None is the default.
     Guards are checked on the stats BEFORE the action's effects are applied.
     Then ACTION_EFFECTS change the stats.

  2. AUTO RULES (stat thresholds). Every tick, the AUTO_RULES are checked in
     priority order, e.g. "health < 20 -> Sick", "food < 15 -> Hungry".

  3. TIMERS. Some moods wear off: Drunk -> Sleepy after 10 s, Sleepy dozes off
     into Asleep after 15 s, and so on (TIMEOUTS).

validate_fsm() checks the whole table when the program starts, so a typo in a
state, action or guard name is caught immediately.
"""

import math
import random
import sys
from dataclasses import dataclass

try:
    import pygame
except ImportError:
    sys.exit("PyGame is required. Install it with:  pip install pygame")


# ===========================================================================
# 1. FSM MODEL (no PyGame in this section - easy to test or reuse)
# ===========================================================================
STATES = ["Happy", "Playful", "Bored", "Hungry", "Angry", "Sad", "Sleepy", "Asleep", "Sick", "Drunk"]
ACTIONS = ["Kick", "Hug", "Play", "Food", "Candy", "Beer", "Medicine", "Bed"]
START_STATE = "Happy"
START_STATS = {"food": 70, "energy": 80, "fun": 70, "health": 100}

# Named guard conditions (on the stats before the action is applied).
GUARDS = {
    "tired": lambda s: s["energy"] < 30,
    "rested": lambda s: s["energy"] > 70,
    "full": lambda s: s["food"] > 80,
    "starving": lambda s: s["food"] < 25,
    "weak": lambda s: s["health"] < 40,
}

# TRANSITIONS[state][action] = "Next" | [(guard | None, "Next"), ...]
TRANSITIONS = {
    "Happy": {
        "Kick": "Angry", "Hug": "Happy",
        "Play": [("tired", "Sleepy"), (None, "Playful")],
        "Food": [("full", "Sleepy"), (None, "Happy")],          # food coma
        "Candy": "Playful", "Beer": "Drunk", "Medicine": "Angry",
        "Bed": [("tired", "Asleep"), (None, "Bored")],          # not tired yet
    },
    "Playful": {
        "Kick": "Angry", "Hug": "Happy",
        "Play": [("tired", "Sleepy"), (None, "Playful")],
        "Food": "Happy",
        "Candy": [("full", "Sick"), (None, "Playful")],         # sugar overload
        "Beer": "Drunk", "Medicine": "Angry",
        "Bed": [("tired", "Asleep"), (None, "Angry")],          # stopped playtime!
    },
    "Bored": {
        "Kick": "Angry", "Hug": "Happy",
        "Play": [("tired", "Sleepy"), (None, "Playful")],
        "Food": [("full", "Bored"), (None, "Happy")],
        "Candy": "Playful", "Beer": "Drunk", "Medicine": "Bored",
        "Bed": [("tired", "Asleep"), (None, "Bored")],
    },
    "Hungry": {
        "Kick": "Angry", "Hug": "Sad", "Play": "Angry", "Food": "Happy",
        "Candy": [("starving", "Angry"), (None, "Happy")],      # candy isn't a meal
        "Beer": [("starving", "Sick"), (None, "Drunk")],        # empty stomach
        "Medicine": "Sick",
        "Bed": [("tired", "Asleep"), (None, "Hungry")],
    },
    "Angry": {
        "Kick": "Sad", "Hug": "Happy", "Play": "Playful", "Food": "Happy",
        "Candy": "Happy", "Beer": "Drunk", "Medicine": "Angry", "Bed": "Sleepy",
    },
    "Sad": {
        "Kick": "Sad", "Hug": "Happy",
        "Play": [("tired", "Sad"), (None, "Playful")],
        "Food": "Happy", "Candy": "Happy", "Beer": "Drunk", "Medicine": "Sad",
        "Bed": [("tired", "Asleep"), (None, "Sleepy")],
    },
    "Sleepy": {
        "Kick": "Angry", "Hug": "Sleepy", "Play": "Angry",
        "Food": [("starving", "Happy"), (None, "Sleepy")],
        "Candy": "Playful", "Beer": "Asleep", "Medicine": "Sleepy", "Bed": "Asleep",
    },
    "Asleep": {
        "Kick": "Angry",
        "Hug": [("rested", "Happy"), (None, "Asleep")],         # gentle wake-up
        "Play": [("rested", "Playful"), (None, "Angry")],       # rude wake-up
        "Food": "Asleep", "Candy": "Asleep", "Beer": "Asleep", "Medicine": "Asleep", "Bed": "Asleep",
    },
    "Sick": {
        "Kick": "Sad", "Hug": "Sick", "Play": "Sick", "Food": "Sick", "Candy": "Sick",
        "Beer": "Sick", "Medicine": "Sleepy",                   # recovering, needs rest
        "Bed": [("starving", "Sick"), (None, "Asleep")],
    },
    "Drunk": {
        "Kick": "Angry", "Hug": "Happy", "Play": "Sick", "Food": "Sleepy", "Candy": "Drunk",
        "Beer": [("weak", "Sick"), (None, "Drunk")],
        "Medicine": "Sleepy", "Bed": "Asleep",
    },
}

# What each action does to the stats.
ACTION_EFFECTS = {
    "Kick": {"health": -10, "fun": -10},
    "Hug": {"fun": +10, "health": +3},
    "Play": {"fun": +30, "energy": -20, "food": -10},
    "Food": {"food": +40, "energy": +5},
    "Candy": {"food": +10, "fun": +15, "energy": +10, "health": -5},
    "Beer": {"fun": +15, "health": -12, "energy": -10, "food": +5},
    "Medicine": {"health": +45, "fun": -5},
    "Bed": {"energy": +5},
}

# Stat change per second. STATE_RATES are added on top of BASE_RATES.
BASE_RATES = {"food": -1.2, "energy": -0.9, "fun": -1.3, "health": 0.0}
STATE_RATES = {
    "Happy": {"health": +0.3},
    "Playful": {"energy": -1.5, "fun": +2.5, "food": -0.8},
    "Sad": {"fun": -0.5},
    "Hungry": {"health": -0.4},
    "Asleep": {"food": +0.6, "energy": +9.5, "fun": +1.8, "health": +2.0},
    "Sick": {"health": -0.6, "energy": -0.4},
    "Drunk": {"health": -0.6, "fun": +2.0},
}

# Automatic transitions on stat thresholds, highest priority first:
# (label, condition, target, states it can fire from)
_ALL = set(STATES)
AUTO_RULES = [
    ("health<20", lambda s: s["health"] < 20, "Sick", _ALL - {"Sick", "Asleep"}),
    ("food<15", lambda s: s["food"] < 15, "Hungry", _ALL - {"Hungry", "Sick"}),
    ("energy<12", lambda s: s["energy"] < 12, "Sleepy", _ALL - {"Sleepy", "Asleep", "Sick", "Hungry"}),
    ("rested", lambda s: s["energy"] >= 98, "Happy", {"Asleep"}),
    ("fun<15", lambda s: s["fun"] < 15, "Bored", {"Happy", "Playful"}),
]

# Moods that wear off: state -> (seconds, next_state)
TIMEOUTS = {
    "Playful": (12, "Happy"),
    "Angry": (15, "Sad"),
    "Sad": (25, "Bored"),
    "Sleepy": (15, "Asleep"),
    "Drunk": (10, "Sleepy"),
}


def validate_fsm():
    """Fail fast if the transition table is incomplete or has a typo."""
    errors = []
    for st in STATES:
        row = TRANSITIONS.get(st)
        if row is None:
            errors.append(f"missing row for state '{st}'")
            continue
        for act in ACTIONS:
            entry = row.get(act)
            if entry is None:
                errors.append(f"{st}: missing action '{act}'")
            elif isinstance(entry, str):
                if entry not in STATES:
                    errors.append(f"{st}/{act}: unknown target '{entry}'")
            else:
                if not entry or entry[-1][0] is not None:
                    errors.append(f"{st}/{act}: guarded rules must end with a (None, default) rule")
                for guard, target in entry:
                    if guard is not None and guard not in GUARDS:
                        errors.append(f"{st}/{act}: unknown guard '{guard}'")
                    if target not in STATES:
                        errors.append(f"{st}/{act}: unknown target '{target}'")
        for extra in set(row) - set(ACTIONS):
            errors.append(f"{st}: unknown action '{extra}'")
    for st, (_, target) in TIMEOUTS.items():
        if st not in STATES or target not in STATES:
            errors.append(f"bad timeout {st} -> {target}")
    if errors:
        raise ValueError("Invalid FSM:\n  " + "\n  ".join(errors))


def resolve_transition(state, action, stats):
    """The transition function: delta(state, action, stats) -> (next_state, guard_used)."""
    entry = TRANSITIONS[state][action]
    if isinstance(entry, str):
        return entry, None
    for guard, target in entry:
        if guard is None or GUARDS[guard](stats):
            return target, guard
    raise AssertionError("unreachable: validate_fsm guarantees a default rule")


def possible_targets(state, action):
    entry = TRANSITIONS[state][action]
    return [entry] if isinstance(entry, str) else [t for _, t in entry]


@dataclass
class Transition:
    trigger: str   # an action name, "auto" or "timer"
    old: str
    new: str
    reason: str = ""


class Pet:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = START_STATE
        self.stats = dict(START_STATS)
        self.time_in_state = 0.0

    def _enter(self, new):
        if new != self.state:
            self.time_in_state = 0.0
        self.state = new

    def preview(self, action):
        return resolve_transition(self.state, action, self.stats)

    def apply(self, action):
        old = self.state
        new, guard = resolve_transition(old, action, self.stats)
        ignored = old == "Asleep" and new == "Asleep"   # sleeping pets ignore most things
        if not ignored:
            for k, v in ACTION_EFFECTS[action].items():
                self.stats[k] = max(0.0, min(100.0, self.stats[k] + v))
        self._enter(new)
        reason = f"[{guard}]" if guard else ("(asleep)" if ignored else "")
        return Transition(action, old, new, reason)

    def tick(self, dt):
        rates = dict(BASE_RATES)
        for k, v in STATE_RATES.get(self.state, {}).items():
            rates[k] += v
        for k, r in rates.items():
            self.stats[k] = max(0.0, min(100.0, self.stats[k] + r * dt))
        self.time_in_state += dt

        for label, cond, target, sources in AUTO_RULES:
            if not cond(self.stats):
                continue
            if self.state == target:
                break                                  # already handled: lower rules don't apply
            if self.state in sources:
                old = self.state
                self._enter(target)
                return Transition("auto", old, target, label)

        if self.state in TIMEOUTS:
            secs, target = TIMEOUTS[self.state]
            if self.time_in_state >= secs:
                old = self.state
                self._enter(target)
                return Transition("timer", old, target, f"{secs}s")
        return None

    def timer_info(self):
        if self.state in TIMEOUTS:
            secs, target = TIMEOUTS[self.state]
            return target, max(0.0, secs - self.time_in_state)
        return None


# ===========================================================================
# 2. LOOK & FEEL
# ===========================================================================
W, H = 1000, 640
FPS = 60
PANEL_X = 610

MOOD_COLORS = {
    "Happy": (255, 200, 70), "Playful": (60, 200, 185), "Bored": (150, 150, 165),
    "Hungry": (245, 145, 55), "Angry": (232, 72, 62), "Sad": (80, 135, 225),
    "Sleepy": (150, 120, 215), "Asleep": (75, 85, 165), "Sick": (150, 180, 70),
    "Drunk": (225, 110, 190),
}
FACE_COLORS = {
    "Happy": (190, 255, 245), "Playful": (200, 255, 235), "Bored": (205, 210, 220),
    "Hungry": (255, 225, 165), "Angry": (255, 175, 150), "Sad": (175, 210, 255),
    "Sleepy": (205, 195, 255), "Asleep": (130, 140, 210), "Sick": (215, 240, 170),
    "Drunk": (255, 195, 230),
}
BUTTON_COLORS = {
    "Kick": (235, 95, 80), "Hug": (245, 135, 180), "Play": (70, 205, 190), "Food": (120, 205, 115),
    "Candy": (250, 160, 210), "Beer": (240, 190, 70), "Medicine": (230, 235, 240), "Bed": (135, 135, 230),
}
STAT_ORDER = [("food", "Food"), ("energy", "Energy"), ("fun", "Fun"), ("health", "Health")]
AMBIENT_RATE = {"Happy": 4, "Playful": 7, "Bored": 0.6, "Hungry": 1.6, "Angry": 10, "Sad": 3,
                "Sleepy": 0.7, "Asleep": 0.9, "Sick": 2.5, "Drunk": 2.5}

BG_TOP, BG_BOTTOM = (22, 24, 38), (10, 10, 18)
PANEL_BG = (18, 20, 30)
SCREEN_BG = (25, 30, 45)
TEXT, MUTED = (230, 232, 240), (130, 135, 155)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def draw_arrow(surf, color, p1, p2, node_r, width=3, dashed=False, offset=5):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy) or 1
    ux, uy = dx / dist, dy / dist
    nx, ny = -uy * offset, ux * offset
    start = (p1[0] + ux * node_r + nx, p1[1] + uy * node_r + ny)
    end = (p2[0] - ux * (node_r + 3) + nx, p2[1] - uy * (node_r + 3) + ny)
    if dashed:
        n = max(1, int(math.hypot(end[0] - start[0], end[1] - start[1]) // 9))
        for i in range(0, n, 2):
            a = (lerp(start[0], end[0], i / n), lerp(start[1], end[1], i / n))
            b = (lerp(start[0], end[0], (i + 1) / n), lerp(start[1], end[1], (i + 1) / n))
            pygame.draw.line(surf, color, a, b, width)
    else:
        pygame.draw.line(surf, color, start, end, width)
    hd = 11
    left = (end[0] - ux * hd - uy * hd * 0.55, end[1] - uy * hd + ux * hd * 0.55)
    right = (end[0] - ux * hd + uy * hd * 0.55, end[1] - uy * hd - ux * hd * 0.55)
    pygame.draw.polygon(surf, color, [end, left, right])


# ===========================================================================
# 3. PARTICLES
# ===========================================================================
_font_cache = {}


def cached_font(size):
    size = int(size)
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


class Particle:
    def __init__(self, kind, x, y, vx, vy, life, color, size, gravity=0.0, text=""):
        self.kind, self.x, self.y, self.vx, self.vy = kind, x, y, vx, vy
        self.life, self.color, self.size, self.gravity, self.text = life, color, size, gravity, text
        self.age = 0.0
        self.rot = random.uniform(0, math.pi)
        self.spin = random.uniform(-6, 6)

    def update(self, dt):
        self.age += dt
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.rot += self.spin * dt
        return self.age < self.life

    def draw(self, surf):
        k = 1 - self.age / self.life
        a = int(255 * min(1.0, k * 1.6))
        col = (*self.color, a)
        x, y, s = self.x, self.y, self.size
        kind = self.kind
        if kind == "spark":
            pts = [(x + math.cos(i * math.pi / 4 + self.rot) * (s if i % 2 == 0 else s * 0.35),
                    y + math.sin(i * math.pi / 4 + self.rot) * (s if i % 2 == 0 else s * 0.35)) for i in range(8)]
            pygame.draw.polygon(surf, col, pts)
        elif kind == "steam":
            pygame.draw.circle(surf, (*self.color, int(a * 0.55)), (int(x), int(y)), int(s * (1 + (1 - k) * 1.8)))
        elif kind == "tear":
            pygame.draw.polygon(surf, col, [(x, y - s * 1.1), (x - s * 0.5, y - s * 0.1), (x + s * 0.5, y - s * 0.1)])
            pygame.draw.circle(surf, col, (int(x), int(y)), int(s * 0.55))
        elif kind == "heart":
            pts = []
            for i in range(24):
                t = i / 24 * 2 * math.pi
                hx = 16 * math.sin(t) ** 3
                hy = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
                pts.append((x + hx * s / 16, y - hy * s / 16))
            pygame.draw.polygon(surf, col, pts)
        elif kind == "bubble":
            pygame.draw.circle(surf, col, (int(x), int(y)), int(s), 2)
            pygame.draw.circle(surf, (255, 255, 255, a), (int(x - s * 0.35), int(y - s * 0.35)), max(1, int(s * 0.25)))
        elif kind == "germ":
            pygame.draw.circle(surf, (*self.color, int(a * 0.8)), (int(x), int(y)), int(s))
            for i in range(6):
                ang = self.rot + i * math.pi / 3
                pygame.draw.line(surf, col, (x + math.cos(ang) * s, y + math.sin(ang) * s),
                                 (x + math.cos(ang) * s * 1.6, y + math.sin(ang) * s * 1.6), 2)
        elif kind == "food":
            tip, l, r = (x, y + s), (x - s * 0.75, y - s * 0.6), (x + s * 0.75, y - s * 0.6)
            pygame.draw.polygon(surf, col, [tip, l, r])
            pygame.draw.line(surf, (190, 110, 40, a), l, r, 4)
            pygame.draw.circle(surf, (210, 50, 40, a), (int(x), int(y - s * 0.1)), max(2, int(s * 0.2)))
        elif kind == "confetti":
            c, sn = math.cos(self.rot), math.sin(self.rot)
            hw, hh = s, s * 0.45
            pts = [(x + c * dx - sn * dy, y + sn * dx + c * dy) for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            pygame.draw.polygon(surf, col, pts)
        elif kind == "cross":
            w = max(2, int(s * 0.4))
            pygame.draw.rect(surf, col, (x - s, y - w / 2, 2 * s, w))
            pygame.draw.rect(surf, col, (x - w / 2, y - s, w, 2 * s))
        elif kind == "text":
            img = cached_font(s).render(self.text, True, self.color)
            img.set_alpha(a)
            surf.blit(img, img.get_rect(center=(int(x), int(y))))


# ===========================================================================
# 4. UI
# ===========================================================================
class Button:
    def __init__(self, action, key_label, rect):
        self.action, self.key_label = action, key_label
        self.rect = pygame.Rect(rect)
        self.color = BUTTON_COLORS[action]
        self.flash = 0.0

    def draw(self, surf, fonts, hovered, preview):
        r = self.rect.move(0, -3 if hovered else 0)
        base = lerp_col(self.color, (255, 255, 255), self.flash)
        pygame.draw.rect(surf, (5, 5, 10), self.rect.move(0, 4), border_radius=12)
        pygame.draw.rect(surf, shade(base, 1.12 if hovered else 1.0), r, border_radius=12)
        pygame.draw.rect(surf, shade(base, 0.6), r, 3, border_radius=12)
        dark = (28, 26, 38)
        label = fonts["mid"].render(self.action, True, dark)
        surf.blit(label, label.get_rect(center=(r.centerx, r.centery - 8)))
        if hovered and preview:
            sub = fonts["small"].render(f"-> {preview}", True, shade(MOOD_COLORS[preview], 0.55))
        else:
            sub = fonts["small"].render(f"key {self.key_label}", True, shade(base, 0.45))
        surf.blit(sub, sub.get_rect(center=(r.centerx, r.centery + 13)))


class CyberPetGame:
    def __init__(self):
        validate_fsm()
        pygame.init()
        pygame.display.set_caption("Cyber-Pet Simulator v2 (FSM)")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        self.fonts = {"big": pygame.font.Font(None, 56), "mid": pygame.font.Font(None, 28),
                      "small": pygame.font.Font(None, 21)}

        bw, bh, gap = 134, 50, 10
        x0 = (PANEL_X - (4 * bw + 3 * gap)) // 2
        self.buttons = []
        for i, a in enumerate(ACTIONS):
            row, col = divmod(i, 4)
            self.buttons.append(Button(a, str(i + 1), (x0 + col * (bw + gap), 512 + row * (bh + 12), bw, bh)))

        cx, cy, R = (PANEL_X + W) // 2, 272, 128
        n = len(STATES)
        self.node_pos = {s: (cx + R * math.cos(-math.pi / 2 + i * 2 * math.pi / n),
                             cy + R * math.sin(-math.pi / 2 + i * 2 * math.pi / n)) for i, s in enumerate(STATES)}
        self.diagram_center = (cx, cy)
        self.pet = Pet()
        self.reset()

    # ----- state ----------------------------------------------------------
    def reset(self):
        self.pet.reset()
        self.last_edge = None
        self.history = []
        self.steps = 0
        self.paused = False
        self.disp_color = list(MOOD_COLORS[self.pet.state])
        self.particles, self.rings = [], []
        self.react = None
        self.t = 0.0
        self.blink_timer, self.blinking = random.uniform(2, 4), 0.0
        self.spawn_acc = 0.0
        self.hovered = None
        self.layout = self.compute_layout()
        print("🤖 Welcome to your Cyber-Pet Simulator v2!")
        print(f"Initial State: {self.pet.state}   stats: {self.pet.stats}")
        print("-" * 50)

    @property
    def state(self):
        return self.pet.state

    def record(self, tr):
        self.history.insert(0, tr)
        del self.history[5:]
        self.last_edge = (tr.old, tr.new)
        if tr.new != tr.old:
            self.rings.append([0.0, MOOD_COLORS[tr.new]])
            self.spawn_acc = 0.0
        icon = {"auto": "⚙️ ", "timer": "⏱️ "}.get(tr.trigger, "👉")
        print(f"{icon} {tr.trigger:<8} {tr.old:>7} ➔ {tr.new:<7} {tr.reason}")

    def apply(self, action):
        tr = self.pet.apply(action)
        self.steps += 1
        self.react = [action, 0.0]
        for b in self.buttons:
            if b.action == action:
                b.flash = 1.0
        if tr.reason != "(asleep)":
            self.spawn_burst(action)
        else:
            self.float_text("zzz... (ignored)", (180, 180, 230))
        self.record(tr)

    # ----- pet geometry ---------------------------------------------------
    def compute_layout(self):
        t, s = self.t, self.state
        cx, cy, sx, sy = 300.0, 290.0, 1.0, 1.0
        if s == "Happy":
            cy -= abs(math.sin(t * 4)) * 14
        elif s == "Playful":
            cy -= abs(math.sin(t * 6)) * 22
            cx += math.sin(t * 3) * 22
        elif s == "Bored":
            cx += math.sin(t * 0.8) * 3
            cy += 4
        elif s == "Angry":
            cx += random.uniform(-2.5, 2.5)
            cy += random.uniform(-1.5, 1.5)
        elif s == "Hungry":
            cx += math.sin(t * 2) * 6
        elif s == "Sad":
            cy += 12 + math.sin(t * 1.2) * 3
        elif s == "Sleepy":
            sy, sx = 1 + 0.035 * math.sin(t * 1.6), 1 - 0.015 * math.sin(t * 1.6)
            cy += 4
        elif s == "Asleep":
            sy, sx = 1 + 0.045 * math.sin(t * 1.2), 1 - 0.02 * math.sin(t * 1.2)
            cy += 10
        elif s == "Sick":
            cx += math.sin(t * 28) * 1.6
            cy += 8
        elif s == "Drunk":
            cx += math.sin(t * 1.7) * 26
            cy += math.sin(t * 3.4) * 5

        if self.react:
            action, age = self.react
            p = max(0.0, 1 - age / 0.6)
            if action == "Kick":
                cx += math.sin(age * 70) * 18 * p
            elif action in ("Hug", "Food", "Candy"):
                sq = math.sin(age * 20) * 0.12 * p
                sx, sy = sx + sq, sy - sq
            elif action in ("Beer", "Medicine"):
                cx += math.sin(age * 10) * 12 * p
            elif action == "Play":
                cy -= math.sin(age * 8) * 30 * p
            elif action == "Bed":
                cy += 10 * p

        bw, bh = 240 * sx, 210 * sy
        body = pygame.Rect(0, 0, int(bw), int(bh))
        body.center = (int(cx), int(cy))
        scr = pygame.Rect(0, 0, int(bw * 0.78), int(bh * 0.58))
        scr.center = (int(cx), int(cy - bh * 0.06))
        ey = scr.centery - scr.height * 0.1
        return {"cx": cx, "cy": cy, "body": body, "screen": scr,
                "eyes": [(scr.centerx - scr.width * 0.24, ey), (scr.centerx + scr.width * 0.24, ey)],
                "mouth": (scr.centerx, scr.centery + scr.height * 0.26), "unit": scr.height / 120.0}

    # ----- particles ------------------------------------------------------
    def float_text(self, text, color, size=38):
        b = self.layout["body"]
        self.particles.append(Particle("text", self.layout["cx"], b.top - 55, 0, -60, 1.1, color, size, text=text))

    def spawn_ambient(self):
        L, s, R = self.layout, self.state, random.uniform
        cx, cy, body = L["cx"], L["cy"], L["body"]
        mx, my = L["mouth"]
        P = self.particles.append
        if s == "Happy":
            ang, d = R(0, 2 * math.pi), R(140, 175)
            P(Particle("spark", cx + math.cos(ang) * d, cy + math.sin(ang) * d * 0.8, R(-10, 10), R(-40, -15),
                       R(0.9, 1.4), (255, 240, 150), R(5, 10)))
        elif s == "Playful":
            P(Particle("confetti", cx + R(-160, 160), body.top - R(20, 80), R(-20, 20), R(-30, 10), R(1.2, 1.8),
                       random.choice([(255, 200, 70), (60, 200, 185), (245, 135, 180), (135, 135, 230)]), R(4, 7),
                       gravity=90))
        elif s == "Bored":
            P(Particle("text", cx + R(70, 120), body.top + R(-10, 20), R(5, 15), -18, 2.0, (170, 172, 185), 28,
                       text=random.choice(["...", "meh", "sigh"])))
        elif s == "Angry":
            side = random.choice((-1, 1))
            P(Particle("steam", cx + side * R(30, 75), body.top + R(-5, 10), side * R(20, 60), R(-120, -70),
                       0.8, (255, 130, 100), R(6, 10)))
        elif s == "Hungry":
            if random.random() < 0.4:
                P(Particle("text", cx + R(-90, 90), body.bottom - 10, 0, -30, 1.4, (255, 210, 150), 24, text="grrr..."))
            else:
                P(Particle("tear", mx + R(-6, 6), my + 12, 0, 20, 1.0, (170, 220, 255), 7, gravity=250))
        elif s == "Sad":
            ex, ey = random.choice(L["eyes"])
            P(Particle("tear", ex + R(-4, 4), ey + 14, R(-15, 15), R(10, 40), 1.1, (120, 180, 255), 8, gravity=500))
        elif s == "Sleepy":
            P(Particle("text", cx + 55, body.top - 5, R(20, 35), R(-40, -28), 1.8, (210, 200, 255), R(20, 28),
                       text="z"))
        elif s == "Asleep":
            P(Particle("text", cx + 60, body.top - 10, R(25, 45), R(-50, -35), 2.4, (190, 195, 255), R(30, 48),
                       text=random.choice("zZ")))
        elif s == "Sick":
            if random.random() < 0.6:
                P(Particle("germ", cx + R(-150, 150), cy + R(-80, 80), R(-15, 15), R(-25, -5), R(1.4, 2.2),
                           (150, 200, 80), R(4, 7)))
            else:
                ex, ey = random.choice(L["eyes"])
                P(Particle("tear", ex + R(-25, 25), body.top + 20, R(-10, 10), 10, 0.9, (190, 230, 255), 6, gravity=300))
        elif s == "Drunk":
            if random.random() < 0.25:
                P(Particle("text", cx + R(-80, 80), body.top - 20, R(-10, 10), -45, 1.2, (255, 190, 230), 30,
                           text="hic!"))
            else:
                P(Particle("bubble", cx + R(-100, 100), body.top + R(-20, 40), R(-15, 15), R(-70, -30), R(1.0, 1.6),
                           (255, 190, 230), R(3, 7)))

    def spawn_burst(self, action):
        L, R = self.layout, random.uniform
        cx, cy, body = L["cx"], L["cy"], L["body"]
        mx, my = L["mouth"]
        P = self.particles.append
        if action == "Kick":
            for _ in range(14):
                ang, sp = R(0, 2 * math.pi), R(150, 320)
                P(Particle("spark", body.left + 5, cy, math.cos(ang) * sp, math.sin(ang) * sp, 0.5, (255, 230, 120), R(6, 11)))
        elif action == "Hug":
            for _ in range(10):
                P(Particle("heart", cx + R(-110, 110), cy + R(-40, 60), R(-30, 30), R(-110, -60), 1.4, (255, 120, 170), R(8, 14)))
        elif action == "Play":
            for _ in range(28):
                ang, sp = R(-math.pi, 0), R(120, 300)
                P(Particle("confetti", cx, body.top, math.cos(ang) * sp, math.sin(ang) * sp, R(1.0, 1.6),
                           random.choice([(255, 200, 70), (60, 200, 185), (245, 135, 180), (135, 135, 230)]), R(4, 7),
                           gravity=300))
        elif action in ("Food", "Candy"):
            col = (255, 195, 85) if action == "Food" else (250, 150, 210)
            for _ in range(8):
                x, y = mx + R(-140, 140), body.top - R(60, 110)
                if action == "Food":
                    P(Particle("food", x, y, (mx - x) / 0.6, (my - y) / 0.6, 0.6, col, 12))
                else:
                    P(Particle("spark", x, y, (mx - x) / 0.6, (my - y) / 0.6, 0.6, col, 9))
        elif action == "Beer":
            for _ in range(18):
                P(Particle("bubble", cx + R(-100, 100), body.bottom - R(0, 60), R(-15, 15), R(-140, -60),
                           R(1.0, 1.8), (255, 230, 140), R(4, 9)))
        elif action == "Medicine":
            for _ in range(10):
                P(Particle("cross", cx + R(-110, 110), cy + R(-30, 70), R(-15, 15), R(-90, -50), 1.3,
                           random.choice([(120, 230, 140), (255, 255, 255)]), R(5, 9)))
        elif action == "Bed":
            for _ in range(6):
                P(Particle("text", cx + R(-60, 60), body.top + R(-20, 20), R(-30, 30), R(-80, -40), 1.6,
                           (210, 200, 255), R(28, 46), text="Z"))
        label = {"Kick": "Ouch!", "Hug": "Aww!", "Play": "Wheee!", "Food": "Yum!", "Candy": "Sugar!",
                 "Beer": "Cheers!", "Medicine": "Bleh...", "Bed": "Bedtime..."}[action]
        self.float_text(label, BUTTON_COLORS[action])

    # ----- loop -----------------------------------------------------------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                return False
            if e.key == pygame.K_r:
                print("🔄 Reset")
                self.reset()
            elif e.key == pygame.K_SPACE:
                self.paused = not self.paused
            idx = e.key - pygame.K_1
            if 0 <= idx < len(ACTIONS):
                self.apply(ACTIONS[idx])
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for b in self.buttons:
                if b.rect.collidepoint(e.pos):
                    self.apply(b.action)
        return True

    def update(self, dt):
        self.t += dt
        if not self.paused:
            tr = self.pet.tick(dt)
            if tr:
                self.record(tr)
                self.float_text(f"{tr.new}!", MOOD_COLORS[tr.new], 34)

        k = min(1.0, dt * 5)
        self.disp_color = [lerp(c, tc, k) for c, tc in zip(self.disp_color, MOOD_COLORS[self.state])]
        self.blink_timer -= dt
        if self.blink_timer <= 0:
            self.blinking, self.blink_timer = 0.13, random.uniform(2, 5)
        self.blinking = max(0.0, self.blinking - dt)
        if self.react:
            self.react[1] += dt
            if self.react[1] > 0.8:
                self.react = None

        self.layout = self.compute_layout()
        self.spawn_acc += AMBIENT_RATE[self.state] * dt
        while self.spawn_acc >= 1:
            self.spawn_ambient()
            self.spawn_acc -= 1
        self.particles = [p for p in self.particles if p.update(dt)]
        for r in self.rings:
            r[0] += dt
        self.rings = [r for r in self.rings if r[0] < 0.7]
        for b in self.buttons:
            b.flash = max(0.0, b.flash - dt * 4)
        mpos = pygame.mouse.get_pos()
        self.hovered = next((b.action for b in self.buttons if b.rect.collidepoint(mpos)), None)

    # ----- drawing: pet ---------------------------------------------------
    def draw_background(self):
        mood = tuple(int(c) for c in self.disp_color)
        top, bottom = lerp_col(BG_TOP, mood, 0.30), lerp_col(BG_BOTTOM, mood, 0.10)
        for y in range(H):
            pygame.draw.line(self.screen, lerp_col(top, bottom, y / H), (0, y), (PANEL_X, y))
        pygame.draw.rect(self.screen, lerp_col(bottom, mood, 0.06), (0, 430, PANEL_X, H - 430))
        pygame.draw.line(self.screen, lerp_col(bottom, mood, 0.25), (0, 430), (PANEL_X, 430), 2)

    def _half_lid_eye(self, ex, ey, r, cover, face, lw, look=(0, 0)):
        surf = self.screen
        pygame.draw.circle(surf, face, (ex, ey), r)
        pygame.draw.circle(surf, SCREEN_BG, (ex + look[0], ey + look[1]), max(2, r // 2))
        h = int(2 * r * cover)
        pygame.draw.rect(surf, SCREEN_BG, (ex - r - 2, ey - r - 2, 2 * r + 4, h + 2))
        pygame.draw.line(surf, face, (ex - r - 3, ey - r + h), (ex + r + 3, ey - r + h), max(2, lw - 2))

    def _spiral_eye(self, ex, ey, r, rot, face, width):
        pts = []
        for i in range(40):
            th = i / 39 * 4 * math.pi
            rr = r * i / 39
            pts.append((ex + math.cos(th + rot) * rr, ey + math.sin(th + rot) * rr))
        pygame.draw.lines(self.screen, face, False, pts, width)

    def draw_pet(self):
        L, s, surf = self.layout, self.state, self.screen
        body, scr, u, cx = L["body"], L["screen"], L["unit"], L["cx"]
        mood = tuple(int(c) for c in self.disp_color)
        dark = shade(mood, 0.55)
        face = FACE_COLORS[s]
        lw = max(3, int(6 * u))

        sh = pygame.Rect(0, 0, int(body.width * 0.9), 34)
        sh.center = (int(cx), 440)
        pygame.draw.ellipse(surf, shade(lerp_col(BG_BOTTOM, mood, 0.1), 0.6), sh)
        for side in (-1, 1):
            f = pygame.Rect(0, 0, 64, 28)
            f.center = (int(cx + side * 62), body.bottom + 6)
            pygame.draw.ellipse(surf, dark, f)

        # arms
        for side in (-1, 1):
            wave = 0
            if s == "Happy" and side == 1:
                wave = math.sin(self.t * 8) * 10
            elif s == "Playful":
                wave = math.sin(self.t * 10 + side) * 16 + 10
            elif s == "Drunk":
                wave = math.sin(self.t * 2 + side) * 12
            droop = 18 if s in ("Sad", "Sleepy", "Asleep", "Sick", "Bored") else 0
            a = pygame.Rect(0, 0, 28, 74)
            a.center = (int(body.left - 6) if side < 0 else int(body.right + 6), int(L["cy"] + 10 + droop - wave))
            pygame.draw.rect(surf, dark, a, border_radius=14)

        # antenna
        tip = (int(cx + math.sin(self.t * 3) * 6), body.top - 44)
        pygame.draw.line(surf, dark, (int(cx), body.top), tip, 5)
        if s == "Angry":
            bulb = (255, 60, 50) if int(self.t * 6) % 2 else (120, 30, 30)
        elif s == "Sick":
            bulb = (150, 220, 80) if int(self.t * 2) % 2 else (70, 100, 40)
        elif s in ("Sleepy", "Asleep", "Bored"):
            bulb = (90, 85, 120)
        elif s == "Drunk":
            bulb = lerp_col((255, 120, 200), (120, 200, 255), 0.5 + 0.5 * math.sin(self.t * 5))
        else:
            bulb = lerp_col(face, (255, 255, 255), 0.5 + 0.5 * math.sin(self.t * 4))
        pygame.draw.circle(surf, bulb, tip, 11)
        pygame.draw.circle(surf, dark, tip, 11, 3)

        # body + screen
        pygame.draw.rect(surf, mood, body, border_radius=int(60 * u))
        pygame.draw.rect(surf, dark, body, 5, border_radius=int(60 * u))
        pygame.draw.rect(surf, SCREEN_BG, scr, border_radius=int(30 * u))
        for y in range(scr.top + 10, scr.bottom - 8, 6):
            pygame.draw.line(surf, (33, 39, 57), (scr.left + 14, y), (scr.right - 14, y))
        pygame.draw.rect(surf, shade(mood, 0.75), scr, 4, border_radius=int(30 * u))

        if s in ("Happy", "Playful", "Drunk"):
            strength = 0.75 if s == "Drunk" else 0.55
            for side in (-1, 1):
                pygame.draw.circle(surf, lerp_col(mood, (255, 110, 140), strength),
                                   (int(cx + side * body.width * 0.33), scr.bottom + 14), 13)

        # eyes
        blink = self.blinking > 0
        for i, (ex, ey) in enumerate(L["eyes"]):
            ex, ey = int(ex), int(ey)
            inner = 1 if i == 0 else -1
            if s == "Happy" or (s == "Playful" and i == 0):
                pygame.draw.arc(surf, face, (ex - 20, ey - 12, 40, 30), 0, math.pi, lw)
            elif s == "Playful":
                pygame.draw.circle(surf, face, (ex, ey), 14)
                pygame.draw.circle(surf, SCREEN_BG, (ex - 3, ey - 2), 6)
            elif s == "Asleep":
                pygame.draw.arc(surf, face, (ex - 18, ey - 10, 36, 16), math.pi, 2 * math.pi, max(2, lw - 3))
            elif s == "Bored":
                if blink:
                    pygame.draw.line(surf, face, (ex - 12, ey), (ex + 12, ey), lw)
                else:
                    self._half_lid_eye(ex, ey, 13, 0.5, face, lw, look=(6, 3))
            elif s == "Sleepy":
                if blink:
                    pygame.draw.line(surf, face, (ex - 12, ey), (ex + 12, ey), lw)
                else:
                    self._half_lid_eye(ex, ey, 13, 0.68 + 0.08 * math.sin(self.t * 1.6), face, lw)
            elif s == "Drunk":
                self._spiral_eye(ex, ey, 16, self.t * 4 * inner, face, max(2, lw - 3))
            else:
                r = {"Angry": 12, "Hungry": 16, "Sad": 10, "Sick": 9}[s]
                if blink:
                    pygame.draw.line(surf, face, (ex - r, ey), (ex + r, ey), lw)
                else:
                    pygame.draw.circle(surf, face, (ex, ey), r)
                    if s == "Hungry":
                        pygame.draw.circle(surf, SCREEN_BG, (ex + 4, ey - 5), 6)
                        pygame.draw.circle(surf, (255, 255, 255), (ex + 6, ey - 7), 2)
                if s == "Angry":
                    pygame.draw.line(surf, face, (ex - inner * 22, ey - 28), (ex + inner * 16, ey - 15), lw)
                elif s in ("Sad", "Sick"):
                    pygame.draw.line(surf, face, (ex - inner * 18, ey - 16), (ex + inner * 16, ey - 27), max(2, lw - 2))
                if s == "Sick":  # eye bags
                    pygame.draw.arc(surf, shade(face, 0.6), (ex - 12, ey + 4, 24, 14), math.pi, 2 * math.pi, 2)

        # mouth
        mx, my = int(L["mouth"][0]), int(L["mouth"][1])
        if s == "Happy":
            pygame.draw.arc(surf, face, (mx - 36, my - 32, 72, 50), math.pi, 2 * math.pi, lw)
        elif s == "Playful":
            pygame.draw.ellipse(surf, face, (mx - 30, my - 20, 60, 42))
            pygame.draw.rect(surf, SCREEN_BG, (mx - 32, my - 22, 64, 22))
            pygame.draw.ellipse(surf, (240, 110, 120), (mx - 11, my + 8, 22, 12))
        elif s == "Bored":
            pygame.draw.line(surf, face, (mx - 18, my + 4), (mx + 22, my + 2), max(2, lw - 2))
        elif s == "Angry":
            pts = [(mx - 30 + i * 10, my + (6 if i % 2 else -4)) for i in range(7)]
            pygame.draw.lines(surf, face, False, pts, max(2, lw - 2))
        elif s == "Hungry":
            h = 26 + math.sin(self.t * 6) * 6
            m = pygame.Rect(0, 0, 34, int(h))
            m.center = (mx, my)
            pygame.draw.ellipse(surf, face, m)
            t2 = pygame.Rect(0, 0, 18, int(h * 0.4))
            t2.midbottom = (mx, m.bottom - 3)
            pygame.draw.ellipse(surf, (240, 110, 120), t2)
        elif s == "Sad":
            pygame.draw.arc(surf, face, (mx - 28, my, 56, 36), 0, math.pi, max(2, lw - 1))
        elif s == "Sleepy":
            yawn = max(0.0, math.sin(self.t * 0.9))
            if yawn > 0.8:
                pygame.draw.ellipse(surf, face, (mx - 12, my - 14, 24, int(18 + 22 * (yawn - 0.8) / 0.2)))
            else:
                pygame.draw.circle(surf, face, (mx, my + 4), 6, 3)
        elif s == "Asleep":
            pygame.draw.circle(surf, face, (mx, my + 4), int(5 + 3 * math.sin(self.t * 1.2)), 3)
        elif s == "Sick":
            pts = [(mx - 26 + i * 4, my + 6 + math.sin(i * 1.2 + self.t * 3) * 3) for i in range(14)]
            pygame.draw.lines(surf, face, False, pts, max(2, lw - 3))
            pygame.draw.line(surf, (235, 235, 240), (mx + 20, my + 6), (mx + 78, my - 20), 6)
            pygame.draw.circle(surf, (230, 60, 60), (mx + 20, my + 6), 6)
        elif s == "Drunk":
            off = math.sin(self.t * 2.5) * 6
            pygame.draw.arc(surf, face, (mx - 38 + off, my - 30, 76, 50), math.pi * 1.05, math.pi * 1.95, lw)

    def draw_effects(self):
        self.overlay.fill((0, 0, 0, 0))
        for age, col in self.rings:
            k = age / 0.7
            pygame.draw.circle(self.overlay, (*col, int(200 * (1 - k))),
                               (int(self.layout["cx"]), int(self.layout["cy"])), int(120 + k * 260), max(1, int(8 * (1 - k))))
        for p in self.particles:
            p.draw(self.overlay)
        self.screen.blit(self.overlay, (0, 0))

    # ----- drawing: HUD ---------------------------------------------------
    def draw_hud(self):
        f, surf = self.fonts, self.screen
        hint = "Click or press 1-8    Space = pause    R = reset    Esc = quit"
        surf.blit(f["small"].render(hint, True, MUTED), (16, 12))

        bar_w, gap = 132, 12
        x0 = (PANEL_X - (4 * bar_w + 3 * gap)) // 2
        for i, (key, name) in enumerate(STAT_ORDER):
            v = self.pet.stats[key]
            x = x0 + i * (bar_w + gap)
            col = (110, 210, 120) if v > 50 else (240, 190, 70) if v > 25 else (235, 85, 75)
            surf.blit(f["small"].render(name, True, TEXT), (x, 36))
            num = f["small"].render(f"{int(v)}", True, col)
            surf.blit(num, num.get_rect(topright=(x + bar_w, 36)))
            pygame.draw.rect(surf, (40, 42, 58), (x, 54, bar_w, 10), border_radius=5)
            if v > 0:
                pygame.draw.rect(surf, col, (x, 54, max(6, int(bar_w * v / 100)), 10), border_radius=5)

        if self.paused:
            p = f["big"].render("PAUSED", True, (255, 255, 255))
            surf.blit(p, p.get_rect(center=(PANEL_X // 2, 470)))

        for b in self.buttons:
            preview = self.pet.preview(b.action)[0] if self.hovered == b.action else None
            b.draw(surf, f, self.hovered == b.action, preview)

    def draw_panel(self):
        surf, f = self.screen, self.fonts
        pygame.draw.rect(surf, PANEL_BG, (PANEL_X, 0, W - PANEL_X, H))
        pygame.draw.line(surf, (40, 44, 62), (PANEL_X, 0), (PANEL_X, H), 2)
        x0, xr = PANEL_X + 24, W - 24
        cur = self.state

        surf.blit(f["small"].render("CYBER-PET  //  FINITE STATE MACHINE", True, MUTED), (x0, 16))
        surf.blit(f["small"].render("Current mood", True, MUTED), (x0, 42))
        surf.blit(f["big"].render(cur, True, MOOD_COLORS[cur]), (x0, 60))

        info = [f"Steps: {self.steps}", f"in state: {self.pet.time_in_state:4.1f}s"]
        ti = self.pet.timer_info()
        info.append(f"timer -> {ti[0]} {ti[1]:.0f}s" if ti else "no timer")
        for i, line in enumerate(info):
            col = MOOD_COLORS[ti[0]] if (i == 2 and ti) else MUTED
            img = f["small"].render(line, True, col)
            surf.blit(img, img.get_rect(topright=(xr, 42 + i * 19)))

        # --- diagram ---
        nr = 27
        targets = set()
        for action in ACTIONS:
            targets.update(possible_targets(cur, action))
        for tgt in targets - {cur}:
            draw_arrow(surf, (58, 63, 86), self.node_pos[cur], self.node_pos[tgt], nr, 2)
        if self.last_edge and self.last_edge[0] != self.last_edge[1]:
            a, b = self.last_edge
            draw_arrow(surf, MOOD_COLORS[b], self.node_pos[a], self.node_pos[b], nr, 4)
        preview, guard = self.pet.preview(self.hovered) if self.hovered else (None, None)
        if preview and preview != cur:
            draw_arrow(surf, (255, 255, 255), self.node_pos[cur], self.node_pos[preview], nr, 3, dashed=True)

        for st in STATES:
            px, py = self.node_pos[st]
            p = (int(px), int(py))
            reachable = st in targets or st == cur
            if st == cur:
                pygame.draw.circle(surf, shade(MOOD_COLORS[st], 0.5), p, int(nr + 4 + 3 * math.sin(self.t * 4)), 3)
                pygame.draw.circle(surf, MOOD_COLORS[st], p, nr)
                r_, g_, b_ = MOOD_COLORS[st]
                col = (25, 25, 35) if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) > 140 else (245, 245, 250)
            else:
                pygame.draw.circle(surf, (34, 37, 54), p, nr)
                pygame.draw.circle(surf, MOOD_COLORS[st] if reachable else shade(MOOD_COLORS[st], 0.45), p, nr, 3)
                col = TEXT if reachable else MUTED
            if preview == st == cur:
                ox, oy = px - self.diagram_center[0], py - self.diagram_center[1]
                d = math.hypot(ox, oy) or 1
                pygame.draw.circle(surf, (255, 255, 255), (int(px + ox / d * (nr + 12)), int(py + oy / d * (nr + 12))), 12, 3)
            lbl = f["small"].render(st, True, col)
            surf.blit(lbl, lbl.get_rect(center=p))

        if self.hovered:
            txt = f"{self.hovered}:  {cur} -> {preview}"
            if guard:
                txt += f"  [{guard}]"
            elif preview == cur:
                txt += "  (stays)"
            t_img = f["small"].render(txt, True, MOOD_COLORS[preview])
        else:
            t_img = f["small"].render("Hover an action to preview it", True, MUTED)
        surf.blit(t_img, t_img.get_rect(center=((PANEL_X + W) // 2, 444)))

        # --- history ---
        surf.blit(f["small"].render("HISTORY", True, MUTED), (x0, 466))
        pygame.draw.line(surf, (40, 44, 62), (x0, 484), (xr, 484), 1)
        if not self.history:
            surf.blit(f["small"].render("No transitions yet", True, (80, 84, 105)), (x0, 494))
        for i, tr in enumerate(self.history):
            y = 494 + i * 27
            alpha = 255 - i * 35
            trig_col = BUTTON_COLORS.get(tr.trigger, MUTED)
            parts = [(tr.trigger, trig_col, x0, f["mid"]),
                     (tr.old, MOOD_COLORS[tr.old], x0 + 92, f["mid"])]
            x = x0 + 92
            for text, col, px, font in parts:
                img = font.render(text, True, col)
                img.set_alpha(alpha)
                surf.blit(img, (px, y))
                x = px + img.get_width()
            for text, col, font, dy in (("->", MUTED, f["mid"], 0), (tr.new, MOOD_COLORS[tr.new], f["mid"], 0),
                                        (tr.reason, MUTED, f["small"], 4)):
                if not text:
                    continue
                x += 8
                img = font.render(text, True, col)
                img.set_alpha(alpha)
                surf.blit(img, (x, y + dy))
                x += img.get_width()

    def draw(self):
        self.draw_background()
        self.draw_pet()
        self.draw_effects()
        self.draw_hud()
        self.draw_panel()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for e in pygame.event.get():
                if not self.handle_event(e):
                    running = False
            self.update(dt)
            self.draw()
        print("👋 Shutting down Cyber-Pet. Goodbye!")
        pygame.quit()


def run_cyber_pet():
    CyberPetGame().run()


if __name__ == "__main__":
    run_cyber_pet()
