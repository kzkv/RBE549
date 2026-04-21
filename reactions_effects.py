# Tom Kazakov
# RBE 549 Week 13 Assignment: Emoji burst effects

import math
import random
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_STRIKE = 160
EMOJI_DISPLAY_SIZE = 72

EMOJI = {
    "thumbs_up": "👍",
    "thumbs_down": "👎",
    "balloons": "🎈",
    "rain": "💧",
    "confetti": "🎊",
    "hearts": "💚",
}

BURST_COUNT = 15
BURST_SPEED_MIN = 6
BURST_SPEED_MAX = 14
LIFETIME_FRAMES = 60
ORIGIN_JITTER_PX = 10

BIAS = {
    "thumbs_up": (0, -3),
    "thumbs_down": (0, 3),
    "balloons": (0, -6),
    "rain": (0, 4),
    "confetti": (0, -2),
    "hearts": (0, -4),
}

PALM_LANDMARKS = (0, 5, 9, 13, 17)

THUMBS_UP_SAT_GAIN = 0.8
THUMBS_UP_VAL_GAIN = 0.2
BALLOONS_SCROLL_PX = 15
RAIN_MAX_SIGMA = 7.0
CONFETTI_ROI = 600
CONFETTI_COPY_SIZE = 90
CONFETTI_COPIES = 8
HEARTS_PINK_BGR = (180, 105, 255)
HEARTS_TINT_STRENGTH = 0.35
HEARTS_VIGNETTE_STRENGTH = 0.6


@lru_cache(maxsize=64)
def emoji_image(char, size):
    font = ImageFont.truetype(EMOJI_FONT_PATH, EMOJI_STRIKE)
    img = Image.new("RGBA", (EMOJI_STRIKE, EMOJI_STRIKE), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((0, 0), char, font=font, embedded_color=True)
    rgba = np.array(img)
    bgra = rgba[:, :, [2, 1, 0, 3]]
    if size != EMOJI_STRIKE:
        bgra = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)
    return bgra


def paste_rgba(frame, rgba, center, alpha_mul=1.0):
    fh, fw = frame.shape[:2]
    eh, ew = rgba.shape[:2]
    cx, cy = center
    x0, y0 = cx - ew // 2, cy - eh // 2
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(fw, x0 + ew), min(fh, y0 + eh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    ex0, ey0 = fx0 - x0, fy0 - y0
    ex1, ey1 = ex0 + (fx1 - fx0), ey0 + (fy1 - fy0)
    bgr = rgba[ey0:ey1, ex0:ex1, :3].astype(np.float32)
    alpha = rgba[ey0:ey1, ex0:ex1, 3:].astype(np.float32) / 255.0 * alpha_mul
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)


def pick_origin(result, fired_class, frame_shape):
    indices = [
        i
        for i, g in enumerate(result.gestures)
        if g and g[0].category_name == fired_class
    ]
    landmarks = result.hand_landmarks[random.choice(indices)]
    nx = sum(landmarks[i].x for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
    ny = sum(landmarks[i].y for i in PALM_LANDMARKS) / len(PALM_LANDMARKS)
    h, w = frame_shape[:2]
    return int(nx * w), int(ny * h)


def _fx_thumbs_up(frame, age, lifetime, origin):
    env = math.sin(math.pi * age / lifetime)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.minimum(hsv[..., 1] * (1 + env * THUMBS_UP_SAT_GAIN), 255)
    hsv[..., 2] = np.minimum(hsv[..., 2] * (1 + env * THUMBS_UP_VAL_GAIN), 255)
    frame[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _fx_thumbs_down(frame, age, lifetime, origin):
    env = math.sin(math.pi * age / lifetime)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame[:] = cv2.addWeighted(frame, 1 - env, gray_bgr, env, 0)


def _fx_balloons(frame, age, lifetime, origin):
    shift = int(age * BALLOONS_SCROLL_PX)
    if shift > 0:
        frame[:] = np.roll(frame, -shift, axis=0)


def _fx_rain(frame, age, lifetime, origin):
    env = math.sin(math.pi * age / lifetime)
    sigma = env * RAIN_MAX_SIGMA
    if sigma > 0.1:
        frame[:] = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma)


def _fx_confetti(frame, age, lifetime, origin):
    env = math.sin(math.pi * age / lifetime)
    if env <= 0.0:
        return
    h, w = frame.shape[:2]
    ox, oy = origin
    x0 = max(0, min(w - CONFETTI_ROI, ox - CONFETTI_ROI // 2))
    y0 = max(0, min(h - CONFETTI_ROI, oy - CONFETTI_ROI // 2))
    patch = frame[y0 : y0 + CONFETTI_ROI, x0 : x0 + CONFETTI_ROI].copy()
    small = cv2.resize(patch, (CONFETTI_COPY_SIZE, CONFETTI_COPY_SIZE))
    rng = random.Random(hash(origin))
    for _ in range(CONFETTI_COPIES):
        px = rng.randint(0, w - CONFETTI_COPY_SIZE)
        py = rng.randint(0, h - CONFETTI_COPY_SIZE)
        dst = frame[py : py + CONFETTI_COPY_SIZE, px : px + CONFETTI_COPY_SIZE]
        cv2.addWeighted(dst, 1 - env, small, env, 0, dst)


@lru_cache(maxsize=4)
def _vignette_mask(h, w):
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    return ((np.cos(np.pi * dist / max_dist) + 1) / 2).astype(np.float32)


def _fx_hearts(frame, age, lifetime, origin):
    env = math.sin(math.pi * age / lifetime)
    h, w = frame.shape[:2]
    pink = np.full_like(frame, HEARTS_PINK_BGR)
    tinted = cv2.addWeighted(
        frame, 1 - HEARTS_TINT_STRENGTH * env, pink, HEARTS_TINT_STRENGTH * env, 0
    )
    mask = 1 - (1 - _vignette_mask(h, w)) * HEARTS_VIGNETTE_STRENGTH * env
    frame[:] = (tinted.astype(np.float32) * mask[:, :, None]).astype(np.uint8)


FRAME_EFFECTS = {
    "thumbs_up": _fx_thumbs_up,
    "thumbs_down": _fx_thumbs_down,
    "balloons": _fx_balloons,
    "rain": _fx_rain,
    "confetti": _fx_confetti,
    "hearts": _fx_hearts,
}


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "emoji", "age", "lifetime")

    def __init__(self, x, y, vx, vy, emoji, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.emoji = emoji
        self.age = 0
        self.lifetime = lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    def alive(self):
        return self.age < self.lifetime

    def draw(self, frame):
        alpha = 1.0 - self.age / self.lifetime
        if alpha <= 0:
            return
        paste_rgba(frame, self.emoji, (int(self.x), int(self.y)), alpha_mul=alpha)


class EffectManager:
    def __init__(self):
        self.particles = []
        self.frame_effect = None
        self.frame_effect_age = 0
        self.frame_effect_origin = None

    def spawn(self, reaction_id, origin):
        emoji = emoji_image(EMOJI[reaction_id], EMOJI_DISPLAY_SIZE)
        bx, by = BIAS[reaction_id]
        cx, cy = origin
        for _ in range(BURST_COUNT):
            theta = random.uniform(0, 2 * math.pi)
            speed = random.uniform(BURST_SPEED_MIN, BURST_SPEED_MAX)
            vx = speed * math.cos(theta) + bx
            vy = speed * math.sin(theta) + by
            jx = random.randint(-ORIGIN_JITTER_PX, ORIGIN_JITTER_PX)
            jy = random.randint(-ORIGIN_JITTER_PX, ORIGIN_JITTER_PX)
            self.particles.append(
                Particle(cx + jx, cy + jy, vx, vy, emoji, LIFETIME_FRAMES)
            )
        self.frame_effect = FRAME_EFFECTS[reaction_id]
        self.frame_effect_age = 0
        self.frame_effect_origin = origin

    def update_and_draw(self, frame):
        if self.frame_effect is not None:
            self.frame_effect(
                frame,
                self.frame_effect_age,
                LIFETIME_FRAMES,
                self.frame_effect_origin,
            )
            self.frame_effect_age += 1
            if self.frame_effect_age >= LIFETIME_FRAMES:
                self.frame_effect = None
                self.frame_effect_origin = None
        alive = []
        for p in self.particles:
            p.update()
            if p.alive():
                p.draw(frame)
                alive.append(p)
        self.particles = alive
