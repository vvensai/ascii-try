import math
import random
import time

import numpy as np
import pygame

STYLE = "clear"
SMOOTH = STYLE == "smooth"

FONT_SIZE = 7
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$" if SMOOTH else " .:-=+*xo#%@"
OUTLINE_CHARS = "|-/\\"
OUTLINE_DARKNESS = 0.45
NUM_FLOWERS = 11 if SMOOTH else 16
SIZE_BOOST = 1.3 if SMOOTH else 1.0
GLOW = 0.45
BACKGROUND = (10, 10, 14)

PEONY_COLORS = [
    (1.00, 0.52, 0.70),
    (1.00, 0.70, 0.78),
    (1.00, 0.45, 0.40),
    (0.92, 0.22, 0.55),
    (0.85, 0.10, 0.25),
]
HYDRANGEA_COLORS = [
    ((0.35, 0.50, 0.98), (0.62, 0.45, 0.92)),
    ((0.92, 0.52, 0.78), (0.76, 0.45, 0.88)),
    ((0.40, 0.75, 1.00), (0.70, 0.80, 1.00)),
]
PAPER_COLORS = [(0.85, 0.70, 0.52), (0.96, 0.82, 0.84), (0.95, 0.92, 0.85), (0.70, 0.80, 0.72)]
RIBBON_COLORS = [(0.88, 0.15, 0.28), (0.96, 0.80, 0.30), (0.98, 0.97, 0.95), (0.55, 0.32, 0.75)]
LEAF_COLORS = [(0.22, 0.58, 0.25), (0.30, 0.66, 0.26), (0.18, 0.48, 0.22), (0.38, 0.70, 0.32)]
BUD_GREEN = np.array([0.45, 0.72, 0.35])

LIGHT = np.array([-0.45, 0.6, 0.65], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)
SCREEN_LIGHT = np.array([-0.6, -0.8])
UP = np.array([0.0, 1.0, 0.0])

DOME_X, DOME_Y, DOME_W, DOME_H = 0.0, 8.3, 3.3, 2.3
TIE_Y = 3.3
WRAP_TOP = 6.0
STEM_END = 0.4

EDGE_V, EDGE_H, EDGE_F, EDGE_B = range(len(RAMP), len(RAMP) + 4)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def ellipse(x, y, cx, cy, rx, ry, angle=0.0):
    ca, sa = math.cos(angle), math.sin(angle)
    u, v = x - cx, y - cy
    return np.hypot((u * ca + v * sa) / rx, (-u * sa + v * ca) / ry)


class Flower:

    def __init__(self, head, size):
        self.head = np.array(head, dtype=float)
        self.size = size
        self.phase = random.uniform(0, 2 * math.pi)
        self.root = np.array([random.uniform(-0.3, 0.3), STEM_END + random.uniform(0, 0.15), head[2] * 0.1])
        self.knot = np.array([random.uniform(-0.12, 0.12), TIE_Y, head[2] * 0.1])
        self.ctrl = np.array([head[0] * 0.3, TIE_Y + (head[1] - TIE_Y) * 0.45, head[2] * 0.5])
        out_x, out_y = (head[0] - DOME_X) / DOME_W, (head[1] - DOME_Y) / DOME_H
        edge = min(math.hypot(out_x, out_y), 1.0)
        self.tilt_angle = math.atan2(-out_y, out_x)
        self.tilt = 1.0 - 0.38 * edge ** 1.5
        self.grow_start = random.uniform(0, 0.4)
        self.bloom_start = self.grow_start + 2.0 + random.uniform(0, 1.2)

    def tip(self, sway):
        return self.head + np.array([sway, -abs(sway) * 0.15, 0.0])

    def stem(self, s, sway):
        s = s[:, None]
        low = self.root + (self.knot - self.root) * (s / 0.3)
        q = np.clip((s - 0.3) / 0.7, 0, 1)
        high = (1 - q) ** 2 * self.knot + 2 * (1 - q) * q * self.ctrl + q * q * self.tip(sway)
        return np.where(s < 0.3, low, high)


class Peony(Flower):
    LAYERS = [(5, 1.00), (6, 0.80), (7, 0.62), (8, 0.46), (9, 0.31)]

    def __init__(self, head, size):
        super().__init__(head, size)
        self.color = np.array(random.choice(PEONY_COLORS))
        self.offsets = np.random.uniform(0, 2 * math.pi, len(self.LAYERS))
        self.stamen_noise = np.random.random(4096)

    def paint(self, x, y, opened):
        n_cells = len(x)
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        layer = np.full(n_cells, -1)
        bright = np.zeros(n_cells)
        part = np.zeros(n_cells, dtype=np.int32)
        rim = np.zeros(n_cells)

        for j, (n, radius) in enumerate(self.LAYERS):
            o = smooth((opened - 0.1 * j) / 0.6)
            R = lerp(0.3 - 0.015 * j, radius, o)
            pos = (th - self.offsets[j]) * n / (2 * math.pi)
            a = pos % 1.0
            e = np.abs(a - 0.5) * 2
            edge = R * (1 - 0.22 * e ** 2)
            inside = r < edge
            rr = r / edge
            if SMOOTH:
                b = (0.3 + 0.65 * rr ** 1.5) * (1 - 0.3 * e ** 6)
            else:
                b = (0.5 + 0.55 * rr ** 1.2) * (1 - 0.5 * e ** 8)
            layer[inside] = j
            bright[inside] = b[inside]
            part[inside] = (j * 16 + np.floor(pos) % n)[inside].astype(np.int32)
            rim[inside] = rr[inside]

        center = smooth((opened - 0.6) / 0.4) * 0.13
        mid = r < center
        layer[mid] = len(self.LAYERS)
        noise = 0.2 if SMOOTH else 0.5
        bright[mid] = 1 - noise + noise * self.stamen_noise[np.arange(n_cells)[mid] % 4096]
        part[mid] = 99

        hit = layer >= 0
        light = 1 + 0.3 * rim * (x * SCREEN_LIGHT[0] + y * SCREEN_LIGHT[1]) / np.maximum(r, 1e-6)
        bright = bright * light
        col = self.color * (1 - 0.07 * layer)[:, None]
        col = col * (1 - 0.3 * rim[:, None] ** 4) + 0.3 * rim[:, None] ** 4
        col = col * (1 - (1 - opened) * 0.6) + BUD_GREEN * (1 - opened) * 0.6
        col[mid] = (0.98, 0.78, 0.25)
        return hit, bright, col, part, layer.astype(float)


class Hydrangea(Flower):
    def __init__(self, head, size):
        super().__init__(head, size)
        n = 34
        i = np.arange(n) + 0.5
        z = 1 - 2 * i / n
        rad = np.sqrt(1 - z * z)
        a = i * math.pi * (3 - math.sqrt(5))
        dirs = np.stack([rad * np.cos(a), rad * np.sin(a), z], axis=1)
        dirs = dirs[dirs[:, 2] > -0.2]
        self.pos = dirs[:, :2] * 0.68
        self.front = dirs[:, 2]
        self.out = normalize(self.pos + 1e-6)
        self.spin = np.random.uniform(0, 2 * math.pi, len(dirs))
        self.appear = np.random.uniform(0, 0.6, len(dirs))
        c1, c2 = (np.array(c) for c in random.choice(HYDRANGEA_COLORS))
        mix = np.clip(0.5 + 0.45 * self.pos[:, 0] + np.random.uniform(-0.3, 0.3, len(dirs)), 0, 1)
        self.colors = c1 * (1 - mix[:, None]) + c2 * mix[:, None]

    def paint(self, x, y, opened):
        s = smooth((opened - self.appear) / 0.4)
        size = 0.44 * (0.5 + 0.5 * s) * (0.6 + 0.4 * np.clip(self.front, 0, 1))
        squash = np.clip(self.front, 0.35, 1.0)

        qx = x[:, None] - self.pos[None, :, 0]
        qy = y[:, None] - self.pos[None, :, 1]
        a = (qx * self.out[:, 0] + qy * self.out[:, 1]) / squash
        b = -qx * self.out[:, 1] + qy * self.out[:, 0]
        rho = np.hypot(a, b) / size
        phi = np.arctan2(b, a) - self.spin
        reach = 0.45 + 0.55 * np.abs(np.cos(2 * phi)) ** 0.7
        covered = rho < reach

        score = np.where(covered, self.front[None, :], -np.inf)
        f = np.argmax(score, axis=1)
        rows = np.arange(len(x))
        hit = np.isfinite(score[rows, f])
        rr = rho[rows, f] / reach[rows, f]
        if SMOOTH:
            bright = 0.35 + 0.55 * rr
        else:
            bright = 0.4 + 0.65 * rr
            bright = np.where(rho[rows, f] < 0.15, 0.35, bright)
        bright *= (0.75 + 0.3 * np.clip(self.front[f], 0, 1))
        bright *= 1 + 0.25 * (self.pos[f] @ SCREEN_LIGHT)
        col = BUD_GREEN * (1 - s[f, None]) + self.colors[f] * s[f, None]
        return hit, bright, col, f.astype(np.int32), 3 * self.front[f]


class Leaf:

    def __init__(self, start, out, delay):
        self.start = np.array(start)
        self.dir = normalize(np.array(out))
        self.out = normalize(np.array([self.dir[0], 0.0, self.dir[2] + 1e-6]))
        self.length = random.uniform(1.4, 2.3)
        self.width = random.uniform(0.3, 0.45)
        self.droop = random.uniform(0.1, 0.35)
        self.color = np.array(random.choice(LEAF_COLORS))
        self.delay = delay
        self._done = None

    def points(self, t):
        g = float(smooth((t - self.delay) / 1.5))
        if g <= 0:
            return None
        if g >= 1 and self._done is not None:
            return self._done
        s = np.linspace(0, g, 36)
        sag = (self.out * 0.3 - UP) * self.droop * self.length * 0.5
        spine = self.start + np.outer(s, self.dir * self.length) + np.outer(s ** 2, sag)
        tangent = normalize(np.gradient(spine, axis=0))
        side = normalize(np.cross(tangent, np.array([0.0, 0.0, 1.0])))
        normal = np.cross(tangent, side)
        S, V = np.meshgrid(s, np.linspace(-1, 1, 11), indexing="ij")
        w = self.width * np.sin(np.pi * s ** 0.6) ** 0.7
        P = spine[:, None, :] + (V * w[:, None])[..., None] * side[:, None, :]
        pattern = (0.75 + 0.25 * S) * (1 - 0.35 * V ** 2) * (1 + 0.35 * np.exp(-(V / 0.15) ** 2))
        n = P.shape[0] * P.shape[1]
        result = (P.reshape(-1, 3).astype(np.float32),
                  np.broadcast_to(normal[:, None, :], P.shape).reshape(-1, 3).astype(np.float32),
                  pattern.reshape(-1).astype(np.float32),
                  np.tile(self.color, (n, 1)).astype(np.float32))
        if g >= 1:
            self._done = result
        return result


def neighbor(a, dy, dx, fill):
    p = np.pad(a, 1, constant_values=fill)
    r, c = a.shape
    return p[1 + dy:1 + dy + r, 1 + dx:1 + dx + c]


class Scene:
    def __init__(self, cols, rows, cw, ch):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        height_px = rows * ch
        self.k = height_px * 0.82 / 11.0
        self.cx = cols * cw / 2
        self.base = height_px * 0.99

        self.flowers = []
        for _ in range(6000):
            if len(self.flowers) >= NUM_FLOWERS:
                break
            kind = Peony if random.random() < 0.55 else Hydrangea
            size = (random.uniform(1.15, 1.45) if kind is Peony else random.uniform(1.0, 1.25)) * SIZE_BOOST
            x = random.uniform(-DOME_W, DOME_W) + DOME_X
            y = random.uniform(-DOME_H, DOME_H) + DOME_Y
            d = math.hypot((x - DOME_X) / DOME_W, (y - DOME_Y) / DOME_H)
            if d > 1:
                continue
            if any(math.hypot(x - f.head[0], y - f.head[1]) < (size + f.size) * 0.5 for f in self.flowers):
                continue
            z = 1.5 * (1 - d * d) + random.uniform(-0.3, 0.3)
            self.flowers.append(kind((x, y, z), size))

        gx, gy = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 31))
        inside = np.hypot(gx, gy) <= 1.0
        spots = np.stack([gx[inside] * DOME_W + DOME_X, gy[inside] * DOME_H + DOME_Y], axis=1)
        for _ in range(60):
            heads = np.array([f.head[:2] for f in self.flowers])
            reach = np.array([f.size * 0.62 for f in self.flowers])
            dist = np.hypot(spots[:, None, 0] - heads[None, :, 0], spots[:, None, 1] - heads[None, :, 1])
            gaps = spots[(dist > reach[None, :]).all(axis=1)]
            if len(gaps) == 0:
                break
            x, y = gaps[random.randrange(len(gaps))]
            d = math.hypot((x - DOME_X) / DOME_W, (y - DOME_Y) / DOME_H)
            kind = Peony if random.random() < 0.5 else Hydrangea
            size = (random.uniform(0.85, 1.05) if kind is Peony else random.uniform(0.8, 0.95)) * SIZE_BOOST
            z = 1.5 * (1 - d * d) - 1.0 + random.uniform(-0.3, 0.3)
            self.flowers.append(kind((x, y, z), size))
            self.flowers[-1].filler = True
        self.flowers.sort(key=lambda f: f.head[2])

        self.leaves = []
        for a in np.linspace(-1.15 * math.pi, 0.15 * math.pi, 16):
            a += random.uniform(-0.12, 0.12)
            ca, sa = math.cos(a), math.sin(a)
            start = (DOME_X + ca * DOME_W * 0.75, DOME_Y + sa * DOME_H * 0.75, -1.8)
            self.leaves.append(Leaf(start, (ca, sa * 0.9 + 0.1, 0.0), random.uniform(1.2, 2.2)))

        clusters = []
        for _ in range(14):
            a = random.uniform(-1.2 * math.pi, 0.2 * math.pi)
            rad = random.uniform(1.0, 1.3)
            c = np.array([DOME_X + math.cos(a) * DOME_W * rad, DOME_Y + math.sin(a) * DOME_H * rad, 0.5])
            clusters.append(c + np.random.normal(0, 0.22, (14, 3)) * [1, 1, 0])
        self.babys_breath = np.concatenate(clusters).astype(np.float32)
        self.bb_twinkle = np.random.uniform(0, 6.3, len(self.babys_breath))

        self.paper = np.array(random.choice(PAPER_COLORS))
        self.ribbon = np.array(random.choice(RIBBON_COLORS))

        n = cols * rows // 120
        self.stars = (np.random.randint(0, rows, n), np.random.randint(0, cols, n),
                      np.random.uniform(0, 6.3, n), np.random.uniform(1.5, 4, n))

    def to_screen(self, p):
        return self.cx + p[0] * self.k, self.base - p[1] * self.k

    def cells_in(self, x0, x1, y0, y1):
        c0 = max(int((self.cx + x0 * self.k) // self.cw), 0)
        c1 = min(int((self.cx + x1 * self.k) // self.cw) + 1, self.cols)
        r0 = max(int((self.base - y1 * self.k) // self.ch), 0)
        r1 = min(int((self.base - y0 * self.k) // self.ch) + 1, self.rows)
        if c0 >= c1 or r0 >= r1:
            return None
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        cc, rr = cc.ravel(), rr.ravel()
        wx = ((cc + 0.5) * self.cw - self.cx) / self.k
        wy = (self.base - (rr + 0.5) * self.ch) / self.k
        return cc, rr, wx, wy


    def wrapper(self, t, back):
        reveal = smooth((t - 0.2) / 1.6)
        if reveal <= 0:
            return None
        top = TIE_Y + (DOME_Y + 3.4 - TIE_Y) * reveal if back else TIE_Y + (WRAP_TOP + 0.4 - TIE_Y) * reveal
        cells = self.cells_in(-4.6, 4.6, TIE_Y - 0.2, top + 0.6)
        if cells is None:
            return None
        cc, rr, x, y = cells
        s = np.clip((y - TIE_Y) / (WRAP_TOP - TIE_Y), 0, None)
        half = 0.42 + (3.0 - 0.42) * s ** 0.85 * (1.15 if back else 1.0)
        u = x / np.maximum(half, 1e-6)
        if back:
            edge_top = WRAP_TOP + 0.8 + 4.2 * np.abs(u) ** 1.6 + 0.4 * np.abs(np.sin(u * 5))
            hit = (np.abs(u) <= 1) & (y >= TIE_Y) & (y <= np.minimum(edge_top, top))
        else:
            edge_top = WRAP_TOP + 0.3 * np.abs(np.sin(u * 7))
            hit = (np.abs(u) <= 1) & (y >= TIE_Y) & (y <= np.minimum(edge_top, top))
        folds = 5
        panel = np.floor((u + 1) / 2 * folds).astype(np.int32)
        f = ((u + 1) / 2 * folds) % 1.0
        bright = 0.55 + 0.3 * np.sin(f * math.pi) - 0.18 * u
        bright *= 0.8 if back else 1.0
        col = np.tile(self.paper * (0.85 if back else 1.0), (len(x), 1))
        return cc, rr, hit, bright, col, panel

    def bow(self, t):
        pop = smooth((t - 1.6) / 0.6)
        if pop <= 0:
            return None
        big = 1.45 * pop
        cells = self.cells_in(-1.0 * big - 0.1, 1.0 * big + 0.1, TIE_Y - 1.2 * big - 0.1, TIE_Y + 0.6 * big + 0.1)
        if cells is None:
            return None
        cc, rr, x, y = cells
        x, y = x / big, (y - TIE_Y) / big
        n = len(x)
        part = np.full(n, -1, np.int32)
        bright = np.zeros(n)
        for side, pid in ((-1, 0), (1, 1)):
            along = -y / 1.1
            centre = side * (0.12 + 0.35 * along)
            inside = (along > 0) & (along < 1 - 0.15 * np.abs(np.sin((x - centre) * 12))) & (np.abs(x - centre) < 0.13)
            part[inside] = pid
            bright[inside] = 0.6 + 0.3 * (1 - along[inside])
        for side, pid in ((-1, 2), (1, 3)):
            d = ellipse(x, y, side * 0.48, 0.14, 0.52, 0.36, side * 0.4)
            inside = d < 1
            part[inside] = pid
            bright[inside] = np.where(d[inside] < 0.45, 0.35, 0.55 + 0.45 * d[inside])
        d = ellipse(x, y, 0.0, 0.05, 0.18, 0.16)
        part[d < 1] = 4
        bright[d < 1] = 0.9 - 0.3 * d[d < 1]
        hit = part >= 0
        col = np.tile(self.ribbon, (n, 1))
        return cc, rr, hit, bright, col, part


    def draw(self, t):
        rows, cols = self.rows, self.cols
        cells = rows * cols
        bright = np.full(cells, -1.0, np.float32)
        color = np.zeros((cells, 3), np.float32)
        ids = np.full(cells, -1, np.int32)
        depth = np.full(cells, -1e9, np.float32)
        outline = np.zeros(cells, bool)
        next_id = [0]

        def paint(cc, rr, hit, b, col, part, dep):
            idx = (rr * cols + cc)[hit]
            dep = np.broadcast_to(dep, hit.shape)[hit]
            front = dep > depth[idx]
            idx = idx[front]
            bright[idx] = b[hit][front]
            color[idx] = col[hit][front]
            ids[idx] = part[hit][front] + next_id[0]
            depth[idx] = dep[front]
            outline[idx] = True
            next_id[0] += 1000

        r, c, phase, speed = self.stars
        tw = (0.5 + 0.5 * np.sin(t * speed + phase)) * smooth((t - 1.5) / 2)
        bright[r * cols + c] = np.where(tw > 0.15, tw * 0.12, -1.0)
        color[r * cols + c] = 0.75

        w = self.wrapper(t, back=True)
        if w is not None:
            paint(*w, -10.0)

        chunks, heads = [], []
        for leaf in self.leaves:
            p = leaf.points(t)
            if p is not None:
                chunks.append(p)
        for f in self.flowers:
            g = float(smooth((t - f.grow_start) / 2.2))
            if g <= 0:
                continue
            bloom = min(max((t - f.bloom_start) / 2.5, 0.0), 1.0)
            sway = 0.0 if getattr(f, "filler", False) else math.sin(t * 1.1 + f.phase) * 0.08 * bloom
            s = np.linspace(0, g, 160)
            stem = f.stem(s, sway)
            stem = np.concatenate([stem, stem + [0.035, 0, 0]]).astype(np.float32)
            n = len(stem)
            chunks.append((stem, np.tile(np.float32([0.6, 0.0, 0.8]), (n, 1)),
                           np.tile(0.6 + 0.3 * s, 2).astype(np.float32),
                           np.tile(np.float32([0.3, 0.65, 0.3]), (n, 1))))
            if g > 0.75:
                budscale = float(smooth((g - 0.75) / 0.25))
                opened = float(smooth(bloom))
                heads.append((f, f.stem(np.array([g]), sway)[0], f.size * (0.35 * budscale + 0.65 * opened), opened))
        show = smooth((t - 3.0) / 1.5)
        if show > 0:
            n = len(self.babys_breath)
            chunks.append((self.babys_breath, np.tile(np.float32([0, 0, 1]), (n, 1)),
                           ((0.75 + 0.25 * np.sin(t * 2 + self.bb_twinkle)) * show).astype(np.float32),
                           np.tile(np.float32([0.97, 0.97, 0.93]), (n, 1))))
        if chunks:
            P = np.concatenate([ch[0] for ch in chunks])
            N = np.concatenate([ch[1] for ch in chunks])
            N = np.where(N[:, 2:3] < 0, -N, N)
            B = np.concatenate([ch[2] for ch in chunks]) * (0.3 + 0.9 * np.clip(N @ LIGHT, 0, 1))
            C = np.concatenate([ch[3] for ch in chunks])
            ci = ((self.cx + P[:, 0] * self.k) // self.cw).astype(np.int32)
            ri = ((self.base - P[:, 1] * self.k) // self.ch).astype(np.int32)
            keep = np.nonzero((ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows))[0]
            keep = keep[P[keep, 2] > depth[ri[keep] * cols + ci[keep]]]
            order = keep[np.argsort(P[keep, 2])]
            idx = ri[order] * cols + ci[order]
            bright[idx] = B[order]
            color[idx] = C[order]
            depth[idx] = P[order, 2]
            ids[idx] = -1
            outline[idx] = False

        w = self.wrapper(t, back=False)
        if w is not None:
            paint(*w, 50.0)
        b = self.bow(t)
        if b is not None:
            paint(*b, 60.0)

        for order_no, (f, tip, scale, opened) in enumerate(heads):
            hx, hy = self.to_screen(tip)
            R = scale * self.k * 1.05
            c0, c1 = max(int((hx - R) // self.cw), 0), min(int((hx + R) // self.cw) + 1, cols)
            r0, r1 = max(int((hy - R) // self.ch), 0), min(int((hy + R) // self.ch) + 1, rows)
            if c0 >= c1 or r0 >= r1:
                continue
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            cc, rr = cc.ravel(), rr.ravel()
            dx = ((cc + 0.5) * self.cw - hx) / (scale * self.k)
            dy = ((rr + 0.5) * self.ch - hy) / (scale * self.k)
            ca, sa = math.cos(f.tilt_angle), math.sin(f.tilt_angle)
            u, v = (dx * ca + dy * sa) / f.tilt, -dx * sa + dy * ca
            x, y = u * ca - v * sa, u * sa + v * ca
            key = (opened, scale, hx, hy)
            if getattr(f, "_key", None) != key:
                f._key, f._painted = key, f.paint(x, y, opened)
            hit, b, col, part, front = f._painted
            paint(cc, rr, hit, b, col, part, 100 + order_no * 10 + front)

        level = bright if SMOOTH else np.where(bright < 0.15, bright, 0.25 + 0.75 * bright)
        char = np.where(bright < 0.02, 0, 1 + np.round(np.clip(level, 0, 1) * (len(RAMP) - 2))).astype(np.int32)
        tint = 0.75 + 0.45 * bright if SMOOTH else 0.45 + 0.8 * bright
        rgb = np.clip(color * np.clip(tint, 0, 1.25)[:, None], 0, 1)

        if SMOOTH:
            D = depth.reshape(rows, cols)
            shade = np.zeros((rows, cols))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                nD = neighbor(D, dy, dx, -1e9)
                shade += (nD > D + 5) & (D > 50)
            rgb *= (1 - 0.09 * shade.reshape(-1))[:, None]
        else:
            I, D = ids.reshape(rows, cols), depth.reshape(rows, cols)
            def differs(dy, dx):
                nI, nD = neighbor(I, dy, dx, -1), neighbor(D, dy, dx, -1e9)
                other = (nI != I) & ((nI >= 0) | (nD > -1e8))
                return other & ((nD < D - 0.12) | ((np.abs(nD - D) < 0.01) & (nI > I)))
            up, down, left, right = differs(-1, 0), differs(1, 0), differs(0, -1), differs(0, 1)
            edge = outline.reshape(rows, cols) & (I >= 0) & (up | down | left | right)
            shape = np.full((rows, cols), EDGE_H, np.int32)
            shape[(left | right) & ~(up | down)] = EDGE_V
            shape[(up & left & ~right) | (down & right & ~left)] = EDGE_F
            shape[(up & right & ~left) | (down & left & ~right)] = EDGE_B
            edge = edge.reshape(-1)
            char[edge] = shape.reshape(-1)[edge]
            rgb[edge] = color[edge] * OUTLINE_DARKNESS

        rgb_cells = (rgb.reshape(rows, cols, 3).transpose(1, 0, 2) * 255).astype(np.uint8)
        return char.reshape(rows, cols), rgb_cells


def build_atlas(font, cw, ch):
    chars = RAMP + OUTLINE_CHARS
    atlas = np.zeros((len(chars), cw, ch), dtype=np.float32)
    for i, c in enumerate(chars):
        surf = font.render(c, True, (255, 255, 255), (0, 0, 0))
        a = pygame.surfarray.array3d(surf)[:, :, 0] / 255.0
        w, h = min(cw, a.shape[0]), min(ch, a.shape[1])
        atlas[i, :w, :h] = a[:w, :h] ** 0.7
    return np.repeat((atlas * 255).astype(np.uint8)[..., None], 3, axis=3)


class App:
    def __init__(self, width, height):
        self.font = pygame.font.SysFont("consolas", FONT_SIZE, bold=True)
        self.cw, self.ch = self.font.size("@")[0], self.font.get_ascent() + 1
        self.cols, self.rows = width // self.cw, height // self.ch
        self.size = (self.cols * self.cw, self.rows * self.ch)
        self.atlas = build_atlas(self.font, self.cw, self.ch)
        self.new_bouquet()

    def new_bouquet(self):
        self.scene = Scene(self.cols, self.rows, self.cw, self.ch)

    def draw(self, surface, t):
        char, rgb = self.scene.draw(t)
        glyphs = self.atlas[char.T].transpose(0, 2, 1, 3, 4).reshape(self.size[0], self.size[1], 3)
        pygame.surfarray.blit_array(surface, glyphs)
        scale = pygame.transform.smoothscale if SMOOTH else pygame.transform.scale
        colors = scale(pygame.surfarray.make_surface(rgb), self.size)
        surface.blit(colors, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        if SMOOTH:
            small = pygame.transform.smoothscale(surface, (self.size[0] // 6, self.size[1] // 6))
            glow = pygame.transform.smoothscale(small, self.size)
            level = int(255 * GLOW)
            glow.fill((level, level, level), special_flags=pygame.BLEND_RGB_MULT)
            surface.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        surface.fill(BACKGROUND, special_flags=pygame.BLEND_RGB_ADD)


def main():
    pygame.init()
    info = pygame.display.Info()
    height = min(950, info.current_h - 120)
    width = min(int(height * 1.0), info.current_w - 80)
    app = App(width, height)
    screen = pygame.display.set_mode(app.size)
    pygame.display.set_caption("ASCII Peony & Hydrangea Bouquet")
    clock = pygame.time.Clock()
    start = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_r):
                    app.new_bouquet()
                    start = time.time()
        app.draw(screen, time.time() - start)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
