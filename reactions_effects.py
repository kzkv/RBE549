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


def _palm_center_normalized(landmarks):
    xs = [landmarks[i].x for i in PALM_LANDMARKS]
    ys = [landmarks[i].y for i in PALM_LANDMARKS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def pick_origin(result, fired_class, frame_shape):
    indices = [
        i
        for i, g in enumerate(result.gestures)
        if g and g[0].category_name == fired_class
    ]
    i = random.choice(indices)
    nx, ny = _palm_center_normalized(result.hand_landmarks[i])
    h, w = frame_shape[:2]
    return int(nx * w), int(ny * h)


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

    def update_and_draw(self, frame):
        alive = []
        for p in self.particles:
            p.update()
            if p.alive():
                p.draw(frame)
                alive.append(p)
        self.particles = alive


if __name__ == "__main__":
    canvas = np.zeros((600, 1200, 3), dtype=np.uint8)
    cell_w, cell_h = 400, 300
    positions = [
        (cell_w // 2 + c * cell_w, cell_h // 2 + r * cell_h)
        for r in range(2)
        for c in range(3)
    ]
    for pos, rid in zip(positions, EMOJI.keys()):
        paste_rgba(canvas, emoji_image(EMOJI[rid], 120), pos)
        cv2.putText(
            canvas,
            rid,
            (pos[0] - 60, pos[1] + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("effects.py", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
