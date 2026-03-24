"""
AQUA-DETECT AI — 3D Drone Pool Detection Simulation
AI-Powered Swimming Pool Detection for Home Insurance Underwriting

Requirements: pip install pygame numpy
Run: python drone_pool_detection.py
"""

import pygame
import numpy as np
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# ─── CONFIG ───────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1400, 820
PANEL_W = 320
SIM_W = WIDTH - PANEL_W
FPS = 60
FOV = 700

# Colors
BG          = (5,  14, 26)
GRID_C      = (0,  40, 70)
CYAN        = (0,  220, 255)
GREEN       = (74, 222, 128)
RED         = (255, 80, 80)
ORANGE      = (251, 191, 36)
BLUE_POOL   = (26, 143, 227)
CYAN_POOL   = (34, 211, 238)
PURPLE_POOL = (124, 58, 237)
TEAL_POOL   = (6,  182, 212)
DRONE_CLR   = (0,  180, 255)
SCAN_CLR    = (0,  255, 200)
HOUSE_WALL  = (80, 80, 110)
HOUSE_ROOF  = (120, 70, 40)
PANEL_BG    = (0,  10, 25)
PANEL_BORD  = (0,  60, 100)
TEXT_DIM    = (71, 85, 105)
TEXT_BRIGHT = (148, 163, 184)
WHITE       = (255, 255, 255)

# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────
class DroneMode(Enum):
    PATROL  = "PATROL"
    SCANNING = "SCANNING"
    COMPLETE = "COMPLETE"

@dataclass
class Property:
    id: int
    x: float
    z: float
    has_pool: bool
    pool_type: Optional[str]   # in-ground | above-ground | covered | freeform
    pool_shape: Optional[str]  # oval | rect | circle | freeform
    disclosed: bool
    scanned: bool = False
    scan_time: str = ""

    @property
    def risk_color(self):
        if not self.scanned:          return TEXT_DIM
        if not self.has_pool:         return GREEN
        if self.disclosed:            return GREEN
        return RED if self.pool_type == "in-ground" else ORANGE

    @property
    def risk_label(self):
        if not self.scanned:          return "PENDING"
        if not self.has_pool:         return "CLEAR"
        if self.disclosed:            return "COMPLIANT"
        return "FLAGGED"

@dataclass
class Alert:
    message: str
    timestamp: str
    prop_id: int

# ─── PROPERTIES ───────────────────────────────────────────────────────────────
def make_properties():
    data = [
        (1, -3.5, -3.5, True,  "in-ground",    "oval",    False),
        (2,  0.0, -3.5, True,  "in-ground",    "rect",    True),
        (3,  3.5, -3.5, False, None,            None,      True),
        (4, -3.5,  0.0, True,  "above-ground", "circle",  False),
        (5,  0.0,  0.0, False, None,            None,      True),
        (6,  3.5,  0.0, True,  "freeform",     "freeform",True),
        (7, -3.5,  3.5, False, None,            None,      True),
        (8,  0.0,  3.5, True,  "covered",      "rect",    False),
        (9,  3.5,  3.5, True,  "in-ground",    "rect",    True),
    ]
    return [Property(*d) for d in data]

POOL_COLORS = {
    "in-ground":    BLUE_POOL,
    "above-ground": CYAN_POOL,
    "covered":      PURPLE_POOL,
    "freeform":     TEAL_POOL,
}

# ─── 3D MATH ──────────────────────────────────────────────────────────────────
def project(x, y, z, cam_angle, cam_elev, cam_dist):
    """Simple perspective projection with camera rotation."""
    cos_a, sin_a = math.cos(cam_angle), math.sin(cam_angle)
    cos_e, sin_e = math.cos(cam_elev),  math.sin(cam_elev)

    rx = x * cos_a - z * sin_a
    rz = x * sin_a + z * cos_a

    ez = y * sin_e + rz * cos_e + cam_dist
    if ez < 0.1:
        return None

    sx = SIM_W // 2 + (rx / ez) * FOV
    sy = HEIGHT // 2 - ((y * cos_e - rz * sin_e) / ez) * FOV
    return (int(sx), int(sy), ez)


def project_pt(pt, ca, ce, cd):
    return project(pt[0], pt[1], pt[2], ca, ce, cd)


# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────
def draw_polygon(surf, pts_3d, fill, stroke, stroke_w, ca, ce, cd):
    projected = [project_pt(p, ca, ce, cd) for p in pts_3d]
    if any(p is None for p in projected):
        return
    pts_2d = [(p[0], p[1]) for p in projected]
    if fill:
        pygame.draw.polygon(surf, fill, pts_2d)
    if stroke:
        pygame.draw.polygon(surf, stroke, pts_2d, stroke_w)


def draw_house(surf, px, pz, ca, ce, cd, scanned):
    """Draw a simple 3D box house."""
    corners = [
        (-0.7, 0,   -0.7), ( 0.7, 0,   -0.7),
        ( 0.7, 0,    0.5), (-0.7, 0,    0.5),
        (-0.7, 0.8, -0.7), ( 0.7, 0.8, -0.7),
        ( 0.7, 0.8,  0.5), (-0.7, 0.8,  0.5),
    ]
    pts = [(px + cx, cy, pz + cz) for cx, cy, cz in corners]
    faces = [
        ([0,1,5,4], HOUSE_WALL),
        ([1,2,6,5], (70,70,100)),
        ([2,3,7,6], HOUSE_WALL),
        ([3,0,4,7], (60,60,95)),
        ([4,5,6,7], HOUSE_ROOF),
    ]
    for indices, color in faces:
        face_pts = [pts[i] for i in indices]
        projected = [project_pt(p, ca, ce, cd) for p in face_pts]
        if any(p is None for p in projected):
            continue
        pts_2d = [(p[0], p[1]) for p in projected]
        pygame.draw.polygon(surf, color, pts_2d)
        pygame.draw.polygon(surf, (0, 50, 80), pts_2d, 1)


def draw_lot(surf, prop, ca, ce, cd, is_current, tick):
    """Draw property lot boundary."""
    corners_3d = [
        (prop.x - 1.4, 0, prop.z - 1.4),
        (prop.x + 1.4, 0, prop.z - 1.4),
        (prop.x + 1.4, 0, prop.z + 1.4),
        (prop.x - 1.4, 0, prop.z + 1.4),
    ]
    projected = [project_pt(p, ca, ce, cd) for p in corners_3d]
    if any(p is None for p in projected):
        return
    pts_2d = [(p[0], p[1]) for p in projected]

    # Fill
    if prop.scanned:
        if prop.has_pool and not prop.disclosed:
            fill = (80, 15, 15)
        else:
            fill = (10, 40, 20)
    else:
        fill = (10, 25, 40)
    pygame.draw.polygon(surf, fill, pts_2d)

    # Border
    if is_current:
        alpha = int(155 + 100 * math.sin(tick * 0.15))
        color = (0, min(255, alpha), 255)
        width = 2
    elif prop.scanned:
        color = (255, 60, 60) if (prop.has_pool and not prop.disclosed) else (50, 180, 80)
        width = 1
    else:
        color = (0, 80, 120)
        width = 1
    pygame.draw.polygon(surf, color, pts_2d, width)


def draw_pool(surf, prop, ca, ce, cd, tick):
    """Draw pool as a colored circle with pulse if flagged."""
    pc = project(prop.x + 0.65, 0.05, prop.z + 0.65, ca, ce, cd)
    if not pc:
        return
    sx, sy, depth = pc
    r = max(4, int(14 / depth * 3))

    color = POOL_COLORS.get(prop.pool_type, BLUE_POOL) if prop.scanned else (20, 60, 100)
    pygame.draw.circle(surf, color, (sx, sy), r)

    if prop.scanned and not prop.disclosed:
        pygame.draw.circle(surf, RED, (sx, sy), r, 2)
        pulse = abs(math.sin(tick * 0.1))
        pulse_r = r + int(pulse * 10)
        pulse_alpha = int(100 + pulse * 155)
        pygame.draw.circle(surf, (255, 60, 60), (sx, sy), pulse_r, 1)


def draw_scan_beam(surf, prop, ca, ce, cd, scan_line):
    """Draw horizontal scan line sweeping over current property."""
    sc = project(prop.x, 0.05, prop.z, ca, ce, cd)
    if not sc:
        return
    sx, sy, _ = sc
    beam_y = sy - 55 + scan_line
    # Glow fill
    for dy in range(-3, 4):
        alpha = max(0, 20 - abs(dy) * 7)
        pygame.draw.line(surf, (0, min(255, alpha * 5), min(255, alpha * 3)), (sx - 55, beam_y + dy), (sx + 55, beam_y + dy))
    pygame.draw.line(surf, SCAN_CLR, (sx - 55, beam_y), (sx + 55, beam_y), 1)


def draw_targeting_reticle(surf, prop, ca, ce, cd, tick):
    """Draw animated targeting crosshair on current property."""
    tg = project(prop.x, 0.5, prop.z, ca, ce, cd)
    if not tg:
        return
    tx, ty, _ = tg
    sz = int(20 + math.sin(tick * 0.08) * 3)
    gap = int(sz * 0.4)
    b = int(sz * 0.55)

    color = CYAN
    # Cross lines
    pygame.draw.line(surf, color, (tx - sz, ty), (tx - gap, ty), 1)
    pygame.draw.line(surf, color, (tx + gap, ty), (tx + sz, ty), 1)
    pygame.draw.line(surf, color, (tx, ty - sz), (tx, ty - gap), 1)
    pygame.draw.line(surf, color, (tx, ty + gap), (tx, ty + sz), 1)
    # Corner brackets
    for cx, cy in [(tx-sz, ty-sz),(tx+sz, ty-sz),(tx-sz, ty+sz),(tx+sz, ty+sz)]:
        dx = b if cx < tx else -b
        dy = b if cy < ty else -b
        pygame.draw.line(surf, color, (cx, cy), (cx + dx, cy), 1)
        pygame.draw.line(surf, color, (cx, cy), (cx, cy + dy), 1)


def draw_drone(surf, drone_x, drone_y, drone_z, ca, ce, cd, mode, scan_prog, prop_angle, tick):
    """Draw the drone with animated propellers and scan cone."""
    dp = project(drone_x, drone_y, drone_z, ca, ce, cd)
    if not dp:
        return
    dx, dy, depth = dp
    ds = max(6, int(18 / depth * 4))

    # Beam to ground
    dg = project(drone_x, 0, drone_z, ca, ce, cd)
    if dg:
        gx, gy, _ = dg
        for i in range(8):
            alpha = int(100 * (1 - i / 8))
            y_step = dy + (gy - dy) * i // 8
            y_next = dy + (gy - dy) * (i + 1) // 8
            pygame.draw.line(surf, (0, min(255, alpha * 2), 255), (dx, y_step), (dx, y_next), 1)

        if mode == DroneMode.SCANNING:
            cone_pts = [(dx, dy), (gx - 28, gy), (gx + 28, gy)]
            cone_surf = pygame.Surface((SIM_W, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(cone_surf, (0, 220, 255, 10), cone_pts)
            surf.blit(cone_surf, (0, 0))

    # Arms
    arm_angles = [0.785, 2.356, 3.927, 5.498]
    for a in arm_angles:
        ax = dx + int(math.cos(a) * ds * 1.2)
        ay = dy + int(math.sin(a) * ds * 0.6)
        pygame.draw.line(surf, (180, 200, 240), (dx, dy), (ax, ay), 2)

        # Propeller
        prop_s = pygame.Surface((ds * 4, ds * 2), pygame.SRCALPHA)
        prop_r = max(2, int(ds * 0.8))
        prop_h = max(1, int(ds * 0.15))
        pygame.draw.ellipse(prop_s, (100, 200, 255, 120), (0, ds//2 - prop_h, ds*4, prop_h*2))
        rotated = pygame.transform.rotate(prop_s, math.degrees(prop_angle + a))
        surf.blit(rotated, (ax - rotated.get_width()//2, ay - rotated.get_height()//2))

    # Body
    body_color = SCAN_CLR if mode == DroneMode.SCANNING else DRONE_CLR
    body_r = max(4, int(ds * 0.45))
    pygame.draw.circle(surf, body_color, (dx, dy), body_r)
    pygame.draw.circle(surf, WHITE, (dx, dy), body_r, 1)

    # Blink LED
    if tick % 30 < 15:
        pygame.draw.circle(surf, RED, (dx, dy - body_r // 2), max(2, body_r // 4))

    # Label
    font_sm = pygame.font.SysFont("Courier New", 10, bold=True)
    label = font_sm.render("DRONE-1", True, CYAN)
    surf.blit(label, (dx - label.get_width()//2, dy - ds - 18))
    if mode == DroneMode.SCANNING:
        slabel = font_sm.render(f"SCANNING... {int(scan_prog * 100)}%", True, SCAN_CLR)
        surf.blit(slabel, (dx - slabel.get_width()//2, dy - ds - 30))


# ─── PANEL DRAWING ────────────────────────────────────────────────────────────
def draw_panel(surf, properties, alerts, mode, tick, font_title, font_body, font_sm):
    ox = SIM_W  # panel x offset
    pygame.draw.rect(surf, PANEL_BG, (ox, 0, PANEL_W, HEIGHT))
    pygame.draw.line(surf, PANEL_BORD, (ox, 0), (ox, HEIGHT), 1)

    y = 15

    # Title
    t = font_title.render("RISK INTELLIGENCE", True, CYAN)
    surf.blit(t, (ox + 15, y)); y += 28

    # Stats
    total     = len(properties)
    scanned   = sum(1 for p in properties if p.scanned)
    pools     = sum(1 for p in properties if p.scanned and p.has_pool)
    flagged   = sum(1 for p in properties if p.scanned and p.has_pool and not p.disclosed)
    coverage  = int(scanned / total * 100)

    stats = [
        ("TOTAL PROPERTIES",  str(total),    TEXT_DIM),
        ("POOLS DETECTED",    str(pools),    (56, 189, 248)),
        ("NON-DISCLOSED",     str(flagged),  RED),
        ("SCAN COVERAGE",     f"{coverage}%", GREEN),
    ]
    for label, val, color in stats:
        lbl = font_sm.render(label, True, TEXT_DIM)
        val_r = font_body.render(val, True, color)
        surf.blit(lbl, (ox + 15, y))
        surf.blit(val_r, (ox + PANEL_W - val_r.get_width() - 15, y))
        y += 22

    # Divider
    y += 5
    pygame.draw.line(surf, PANEL_BORD, (ox + 10, y), (ox + PANEL_W - 10, y), 1)
    y += 10

    # Scan status
    mode_color = SCAN_CLR if mode == DroneMode.SCANNING else TEXT_DIM
    ms = font_title.render(f"● {mode.value}", True, mode_color)
    surf.blit(ms, (ox + 15, y)); y += 26

    # Divider
    pygame.draw.line(surf, PANEL_BORD, (ox + 10, y), (ox + PANEL_W - 10, y), 1)
    y += 10

    # Property list
    prop_title = font_title.render("PROPERTY SCAN RESULTS", True, CYAN)
    surf.blit(prop_title, (ox + 15, y)); y += 22

    for prop in properties:
        # Row background
        if prop.scanned:
            if prop.has_pool and not prop.disclosed:
                bg = (60, 10, 10)
            else:
                bg = (10, 35, 20)
        else:
            bg = (8, 20, 35)

        row_h = 38
        pygame.draw.rect(surf, bg, (ox + 8, y, PANEL_W - 16, row_h), border_radius=3)
        pygame.draw.rect(surf, prop.risk_color if prop.scanned else PANEL_BORD,
                         (ox + 8, y, PANEL_W - 16, row_h), 1, border_radius=3)

        # Prop ID
        id_color = CYAN if not prop.scanned else TEXT_BRIGHT
        id_txt = font_body.render(f"PROP #{prop.id}", True, id_color)
        surf.blit(id_txt, (ox + 14, y + 5))

        # Status badge
        badge_txt = font_sm.render(prop.risk_label, True, prop.risk_color)
        surf.blit(badge_txt, (ox + PANEL_W - badge_txt.get_width() - 14, y + 6))

        # Pool type sub-label
        if prop.scanned and prop.has_pool:
            sub = font_sm.render(f"{prop.pool_type.upper()} · {prop.pool_shape}", True, (56, 189, 248))
            surf.blit(sub, (ox + 14, y + 22))

        y += row_h + 4
        if y > HEIGHT - 150:
            break

    # Divider
    y += 4
    pygame.draw.line(surf, PANEL_BORD, (ox + 10, y), (ox + PANEL_W - 10, y), 1)
    y += 10

    # Alerts section
    alert_title = font_title.render("⚠  ALERTS", True, RED)
    surf.blit(alert_title, (ox + 15, y)); y += 22

    if not alerts:
        no_alert = font_sm.render("No alerts yet...", True, TEXT_DIM)
        surf.blit(no_alert, (ox + 15, y))
    else:
        for alert in alerts[:4]:
            pygame.draw.rect(surf, (50, 8, 8), (ox + 8, y, PANEL_W - 16, 38), border_radius=3)
            pygame.draw.rect(surf, (150, 40, 40), (ox + 8, y, PANEL_W - 16, 38), 1, border_radius=3)
            msg = font_sm.render(alert.message[:42], True, RED)
            ts  = font_sm.render(alert.timestamp, True, TEXT_DIM)
            surf.blit(msg, (ox + 12, y + 4))
            surf.blit(ts,  (ox + 12, y + 20))
            y += 44
            if y > HEIGHT - 20:
                break


# ─── HUD OVERLAY ──────────────────────────────────────────────────────────────
def draw_hud(surf, font_sm, font_title, tick, mode, scan_prog, current_prop_id):
    # Top-left header
    pygame.draw.rect(surf, (0, 10, 25, 200), (0, 0, SIM_W, 42))
    pygame.draw.line(surf, PANEL_BORD, (0, 42), (SIM_W, 42), 1)
    blink = (0, 255, 136) if tick % 60 < 30 else (0, 100, 60)
    pygame.draw.circle(surf, blink, (15, 21), 6)
    title = font_title.render("AquaIntelligence  —  DRONE SURVEILLANCE  —  POOL DETECTION ACTIVE", True, CYAN)
    surf.blit(title, (30, 10))

    # Bottom hint
    hint = font_sm.render("DRAG: rotate camera   SCROLL: zoom   R: reset camera", True, (50, 80, 110))
    surf.blit(hint, (10, HEIGHT - 20))

    # Scan progress bar
    if mode == DroneMode.SCANNING:
        bw = 260
        bx = SIM_W // 2 - bw // 2
        by = HEIGHT - 55
        pygame.draw.rect(surf, (0, 20, 40), (bx - 10, by - 30, bw + 20, 50), border_radius=4)
        pygame.draw.rect(surf, SCAN_CLR, (bx - 10, by - 30, bw + 20, 50), 1, border_radius=4)
        label = font_sm.render(f"SCANNING PROPERTY #{current_prop_id}", True, SCAN_CLR)
        surf.blit(label, (bx, by - 24))
        pygame.draw.rect(surf, (0, 40, 30), (bx, by - 6, bw, 10), border_radius=3)
        fill_w = int(bw * scan_prog)
        if fill_w > 0:
            pygame.draw.rect(surf, SCAN_CLR, (bx, by - 6, fill_w, 10), border_radius=3)
        pct = font_sm.render(f"{int(scan_prog*100)}%", True, CYAN)
        surf.blit(pct, (bx + bw + 5, by - 8))


# ─── MAIN SIMULATION ──────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AQUA-DETECT AI — 3D Drone Pool Detection Simulation")
    clock = pygame.time.Clock()

    # Fonts
    font_title = pygame.font.SysFont("Courier New", 11, bold=True)
    font_body  = pygame.font.SysFont("Courier New", 12, bold=True)
    font_sm    = pygame.font.SysFont("Courier New", 10)

    # State
    properties   = make_properties()
    alerts: List[Alert] = []
    prop_idx     = 0
    mode         = DroneMode.PATROL
    drone_x      = -6.0
    drone_y      = 5.0
    drone_z      = -6.0
    scan_prog    = 0.0
    scan_line    = 0
    prop_angle   = 0.0
    tick         = 0

    # Camera
    cam_angle = 0.4
    cam_elev  = 0.6
    cam_dist  = 18.0
    dragging  = False
    last_mx   = 0
    last_my   = 0
    auto_rotate = True

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        tick += 1
        prop_angle += 0.3

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    cam_angle, cam_elev, cam_dist = 0.4, 0.6, 18.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[0] < SIM_W:
                    dragging = True
                    auto_rotate = False
                    last_mx, last_my = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    cam_angle += (mx - last_mx) * 0.005
                    cam_elev   = max(0.1, min(1.4, cam_elev - (my - last_my) * 0.005))
                    last_mx, last_my = mx, my
            elif event.type == pygame.MOUSEWHEEL:
                cam_dist = max(8.0, min(30.0, cam_dist - event.y * 0.6))

        if auto_rotate:
            cam_angle += 0.001

        # ── Drone logic ───────────────────────────────────────────────────────
        target = properties[prop_idx]
        ddx    = target.x - drone_x
        ddz    = target.z - drone_z
        dist   = math.sqrt(ddx ** 2 + ddz ** 2)

        if dist > 0.15:
            speed = 0.04
            drone_x += (ddx / dist) * speed
            drone_z += (ddz / dist) * speed
            drone_y  = 5.0 + math.sin(tick * 0.05) * 0.05
            mode     = DroneMode.PATROL
        else:
            mode       = DroneMode.SCANNING
            scan_prog += 0.007
            scan_line  = (scan_line + 3) % 140

            if scan_prog >= 1.0:
                if not target.scanned:
                    target.scanned   = True
                    target.scan_time = time.strftime("%H:%M:%S")
                    if target.has_pool and not target.disclosed:
                        alerts.insert(0, Alert(
                            message=f"PROP #{target.id}: UNDISCLOSED {target.pool_type} pool!",
                            timestamp=target.scan_time,
                            prop_id=target.id,
                        ))
                        if len(alerts) > 6:
                            alerts.pop()
                scan_prog = 0.0
                prop_idx  = (prop_idx + 1) % len(properties)
                mode      = DroneMode.PATROL

        # ── Drawing ───────────────────────────────────────────────────────────
        screen.fill(BG)

        # Clip drawing to simulation area
        sim_surf = screen.subsurface((0, 0, SIM_W, HEIGHT))

        # Grid
        for g in range(-8, 9):
            a = project(g, 0, -8, cam_angle, cam_elev, cam_dist)
            b = project(g, 0,  8, cam_angle, cam_elev, cam_dist)
            if a and b:
                pygame.draw.line(sim_surf, GRID_C, (a[0], a[1]), (b[0], b[1]), 1)
            c = project(-8, 0, g, cam_angle, cam_elev, cam_dist)
            d = project( 8, 0, g, cam_angle, cam_elev, cam_dist)
            if c and d:
                pygame.draw.line(sim_surf, GRID_C, (c[0], c[1]), (d[0], d[1]), 1)

        # Properties
        for i, prop in enumerate(properties):
            is_current = (i == prop_idx)
            draw_lot(sim_surf, prop, cam_angle, cam_elev, cam_dist, is_current, tick)
            draw_house(sim_surf, prop.x, prop.z, cam_angle, cam_elev, cam_dist, prop.scanned)
            if prop.has_pool:
                draw_pool(sim_surf, prop, cam_angle, cam_elev, cam_dist, tick)
            if is_current and mode == DroneMode.SCANNING:
                draw_scan_beam(sim_surf, prop, cam_angle, cam_elev, cam_dist, scan_line)

            # Property ID label
            lp = project(prop.x, 1.4, prop.z, cam_angle, cam_elev, cam_dist)
            if lp:
                color = prop.risk_color if prop.scanned else (50, 80, 110)
                lbl = font_sm.render(f"#{prop.id}", True, color)
                sim_surf.blit(lbl, (lp[0] - lbl.get_width()//2, lp[1]))

        # Targeting reticle on current property
        draw_targeting_reticle(sim_surf, properties[prop_idx], cam_angle, cam_elev, cam_dist, tick)

        # Drone
        draw_drone(sim_surf, drone_x, drone_y, drone_z,
                   cam_angle, cam_elev, cam_dist,
                   mode, scan_prog, prop_angle, tick)

        # HUD
        draw_hud(sim_surf, font_sm, font_title, tick, mode, scan_prog,
                 properties[prop_idx].id)

        # Right panel
        draw_panel(screen, properties, alerts, mode, tick,
                   font_title, font_body, font_sm)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()