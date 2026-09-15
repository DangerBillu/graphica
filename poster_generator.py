#!/usr/bin/env python3
"""
poster_generator.py — Graphic design poster synthesis engine for REACTIONARY.

Translates facial reaction, detected emotion, and design style into high-resolution,
professionally composed graphic design posters adhering to authentic design principles:
typographic hierarchy, mathematical grids, texture overlays, bespoke face treatments,
and intentional editorial metadata.
"""

import os
import random
import time
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

# Standard poster dimensions: 1200 x 1600 (3:4 ratio)
POSTER_WIDTH = 1200
POSTER_HEIGHT = 1600

# Font resolution helper
FONTS_DIR = "C:/Windows/Fonts"

def get_font(name_candidates, size):
    """Attempt to load the first existing font candidate, fallback to default."""
    for name in name_candidates:
        p = os.path.join(FONTS_DIR, name)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    # Fallback
    try:
        return ImageFont.truetype(os.path.join(FONTS_DIR, "arial.ttf"), size)
    except Exception:
        return ImageFont.load_default()

# Typography presets
FONTS = {
    "display_condensed": lambda s: get_font(["ARIALNB.TTF", "impact.ttf", "arialbd.ttf"], s),
    "display_sans": lambda s: get_font(["arialbd.ttf", "segoeuib.ttf", "trebucbd.ttf"], s),
    "display_serif": lambda s: get_font(["georgiab.ttf", "timesbd.ttf", "georgia.ttf"], s),
    "body_sans": lambda s: get_font(["arial.ttf", "segoeui.ttf"], s),
    "body_serif": lambda s: get_font(["georgia.ttf", "times.ttf"], s),
    "mono": lambda s: get_font(["consolab.ttf", "consola.ttf", "cour.ttf"], s),
    "display_rounded": lambda s: get_font(["comicbd.ttf", "arialbd.ttf"], s),
    "display_heavy": lambda s: get_font(["impact.ttf", "arialbd.ttf"], s),
}


# --- Image and Texture Processing Helpers ---

def add_film_grain(img, intensity=0.12):
    """Add subtle photographic film grain texture to a PIL image."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_halftone_effect(img, sample_rate=4):
    """Apply print halftone dot pattern effect."""
    gray = img.convert("L")
    w, h = gray.size
    small = gray.resize((w // sample_rate, h // sample_rate), Image.Resampling.BILINEAR)
    small_arr = np.array(small)
    
    out = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(out)
    
    for y in range(small_arr.shape[0]):
        for x in range(small_arr.shape[1]):
            val = small_arr[y, x]
            # Darker pixel = larger dot
            radius = (1.0 - (val / 255.0)) * (sample_rate * 0.7)
            if radius > 0.5:
                cx = x * sample_rate + sample_rate // 2
                cy = y * sample_rate + sample_rate // 2
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=0)
    return out


def chromatic_aberration(img, offset=6):
    """Apply cyber chromatic RGB channel offset."""
    arr = np.array(img.convert("RGB"))
    h, w, c = arr.shape
    out = np.zeros_like(arr)
    
    # Red shifted right
    out[:, offset:w, 0] = arr[:, 0:w - offset, 0]
    # Green untouched
    out[:, :, 1] = arr[:, :, 1]
    # Blue shifted left
    out[:, 0:w - offset, 2] = arr[:, offset:w, 2]
    return Image.fromarray(out)


def duotone_filter(img, dark_rgb, light_rgb):
    """Convert grayscale image into a 2-color duotone."""
    gray = img.convert("L")
    arr = np.array(gray).astype(np.float32) / 255.0
    
    r = (dark_rgb[0] * (1.0 - arr) + light_rgb[0] * arr).astype(np.uint8)
    g = (dark_rgb[1] * (1.0 - arr) + light_rgb[1] * arr).astype(np.uint8)
    b = (dark_rgb[2] * (1.0 - arr) + light_rgb[2] * arr).astype(np.uint8)
    
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb)


def crop_face_area(frame, box=None):
    """Crop the head and shoulders region from camera frame (BGR numpy array)."""
    h, w = frame.shape[:2]
    if box:
        x0, y0, x1, y1 = box
        fw, fh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        
        # Generous portrait crop (head and upper shoulders)
        crop_w = int(fw * 1.7)
        crop_h = int(fh * 2.2)
        
        nx0 = max(0, cx - crop_w // 2)
        ny0 = max(0, cy - int(crop_h * 0.45))
        nx1 = min(w, nx0 + crop_w)
        ny1 = min(h, ny0 + crop_h)
        face_roi = frame[ny0:ny1, nx0:nx1]
    else:
        # Center portrait fallback
        ch, cw = int(h * 0.8), int(h * 0.8 * 0.75)
        nx0 = max(0, (w - cw) // 2)
        ny0 = max(0, (h - ch) // 2)
        face_roi = frame[ny0:ny0 + ch, nx0:nx0 + cw]
        
    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_barcode(draw, x, y, width=180, height=36, fill=(0, 0, 0)):
    """Draw a realistic vector barcode graphic."""
    np.random.seed(42)
    cur_x = x
    while cur_x < x + width:
        bar_w = np.random.choice([2, 3, 4, 6])
        gap = np.random.choice([2, 3, 5])
        draw.rectangle([cur_x, y, cur_x + bar_w, y + height], fill=fill)
        cur_x += bar_w + gap


def draw_crosshair(draw, cx, cy, size=14, fill=(255, 255, 255, 180)):
    """Draw an editorial alignment crosshair."""
    draw.line([cx - size, cy, cx + size, cy], fill=fill, width=1)
    draw.line([cx, cy - size, cx, cy + size], fill=fill, width=1)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=fill, width=1)


def draw_chrome_star(draw, cx, cy, radius=24, fill=(255, 255, 255)):
    """Draw a Y2K 4-pointed sparkle star."""
    points = [
        (cx, cy - radius),
        (cx + radius * 0.22, cy - radius * 0.22),
        (cx + radius, cy),
        (cx + radius * 0.22, cy + radius * 0.22),
        (cx, cy + radius),
        (cx - radius * 0.22, cy + radius * 0.22),
        (cx - radius, cy),
        (cx - radius * 0.22, cy - radius * 0.22),
    ]
    draw.polygon(points, fill=fill)


# --- 10 DESIGN STYLE POSTER RENDERERS ---

class PosterRenderer:
    """Base class for design styles."""
    def __init__(self, data):
        self.emotion = data.get("emotion", "NEUTRAL")
        self.label = data.get("label", "Unimpressed").upper()
        self.style = data.get("style", "Swiss")
        self.score = data.get("confidence", 85)
        self.design_id = data.get("design_id", f"#RX-{random.randint(100, 999)}")
        self.timestamp = time.strftime("%Y.%m.%d // %H:%M:%S")
        self.face_img = data.get("face_img")
        self.params = data.get("params", {})


class SwissRenderer(PosterRenderer):
    """
    Swiss / International Typographic Style:
    - Strict mathematical modular grid
    - Stark monochrome with striking vermillion red (#FF3B00)
    - Grotesque uppercase display typography, asymmetrical balance
    - Section index numbers, alignment crosshairs, hairline borders
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (245, 245, 243))
        draw = ImageDraw.Draw(poster)
        
        # Grid lines (light gray)
        margin = 60
        cols = 4
        col_w = (W - margin * 2) // cols
        
        for i in range(cols + 1):
            gx = margin + i * col_w
            draw.line([gx, margin, gx, H - margin], fill=(225, 225, 222), width=1)
        
        for gy in range(margin, H - margin, 120):
            draw.line([margin, gy, W - margin, gy], fill=(235, 235, 233), width=1)

        # Header section
        f_label = FONTS["body_sans"](16)
        f_num = FONTS["display_condensed"](42)
        f_title = FONTS["display_sans"](118)
        f_mono = FONTS["mono"](14)
        
        draw.text((margin, margin), "REACTIONARY / SWISS ARCHIVE", font=f_label, fill=(20, 20, 20))
        draw.text((W - margin - 140, margin), "ISO 216 / A1", font=f_mono, fill=(120, 120, 120))
        draw.text((W - margin - 60, margin + 25), "01", font=f_num, fill=(255, 59, 0))
        
        # Solid vermillion accent block
        draw.rectangle([margin, margin + 60, margin + col_w * 2, margin + 68], fill=(255, 59, 0))
        
        # Main Display Headline
        draw.text((margin - 6, margin + 90), self.label, font=f_title, fill=(12, 12, 12))
        
        # Secondary subhead
        f_sub = FONTS["display_sans"](28)
        draw.text((margin, margin + 225), "INTERNATIONAL TYPOGRAPHIC SYSTEM", font=f_sub, fill=(255, 59, 0))
        
        # Face Treatment: High contrast black & white with geometric cropping
        face_y = margin + 300
        face_w = col_w * 3
        face_h = 700
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.4)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc.convert("RGB"), (margin, face_y))
        else:
            draw.rectangle([margin, face_y, margin + face_w, face_y + face_h], fill=(200, 200, 200))
        
        # Face border and registration mark
        draw.rectangle([margin, face_y, margin + face_w, face_y + face_h], outline=(15, 15, 15), width=2)
        draw_crosshair(draw, margin, face_y, 16, fill=(255, 59, 0))
        draw_crosshair(draw, margin + face_w, face_y + face_h, 16, fill=(255, 59, 0))
        
        # Right column metadata block
        rx = margin + col_w * 3 + 30
        draw.text((rx, face_y), "REACTION TELEMETRY", font=FONTS["display_sans"](16), fill=(15, 15, 15))
        draw.line([rx, face_y + 26, W - margin, face_y + 26], fill=(15, 15, 15), width=2)
        
        meta_items = [
            ("EMOTION", self.label),
            ("CONFIDENCE", f"{self.score}%"),
            ("INDEX ID", self.design_id),
            ("STYLE", "SWISS 1957"),
            ("GRID UNIT", "8PT MODULAR"),
            ("WEIGHT", "HEAVY BLACK"),
            ("TIMESTAMP", self.timestamp),
        ]
        
        my = face_y + 45
        for k, v in meta_items:
            draw.text((rx, my), k, font=FONTS["mono"](12), fill=(130, 130, 130))
            draw.text((rx, my + 16), v, font=FONTS["display_sans"](18), fill=(20, 20, 20))
            my += 54
            
        # Large graphical percentage
        draw.text((rx, my + 40), f"{self.score}", font=FONTS["display_condensed"](120), fill=(255, 59, 0))
        draw.text((rx + 150, my + 60), "%", font=FONTS["display_sans"](48), fill=(20, 20, 20))
        draw.text((rx, my + 170), "FACIAL COHESION INDEX", font=FONTS["mono"](12), fill=(100, 100, 100))

        # Bottom section
        bot_y = face_y + face_h + 50
        draw.line([margin, bot_y, W - margin, bot_y], fill=(15, 15, 15), width=3)
        
        p1 = ("THE PURPOSE OF THE TYPOGRAPHIC GRID IS TO FACILITATE STRUCTURAL ORDER, "
              "OBJECTIVE CLARITY, AND HARMONIOUS RHYTHM IN EXPERIMENTAL INTERFACES.")
        draw.text((margin, bot_y + 25), p1, font=FONTS["mono"](14), fill=(60, 60, 60))
        
        draw_barcode(draw, margin, bot_y + 90, width=220, height=44, fill=(15, 15, 15))
        draw.text((margin + 240, bot_y + 115), f"REACTIONARY STUDIO // {self.design_id} // VERIFIED",
                  font=FONTS["mono"](13), fill=(100, 100, 100))
        
        return poster


class BrutalistRenderer(PosterRenderer):
    """
    Brutalist / Aggressive Style:
    - Pitch black background, pure white, and neon-lime (#CCFF00) accents
    - Heavy condensed display typography, raw borders, warning badges
    - Extreme scale contrast, high-contrast face dither, barcode & raw stamps
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (10, 10, 12))
        draw = ImageDraw.Draw(poster)
        
        margin = 50
        
        # Hazard stripe top bar
        stripe_w = 24
        for sx in range(0, W, stripe_w * 2):
            draw.polygon([(sx, 0), (sx + stripe_w, 0), (sx + stripe_w - 20, 24), (sx - 20, 24)], fill=(204, 255, 0))
            draw.polygon([(sx + stripe_w, 0), (sx + stripe_w * 2, 0), (sx + stripe_w * 2 - 20, 24), (sx + stripe_w - 20, 24)], fill=(0, 0, 0))

        # Raw Header Box
        draw.rectangle([margin, 44, W - margin, 100], outline=(255, 255, 255), width=3)
        draw.text((margin + 20, 60), "WARNING // HIGH TENSION PSYCHOLOGICAL MANIFEST", font=FONTS["mono"](18), fill=(204, 255, 0))
        draw.text((W - margin - 220, 60), self.design_id, font=FONTS["mono"](20), fill=(255, 255, 255))
        
        # Massive Raw Headline
        f_huge = FONTS["display_condensed"](160)
        draw.text((margin, 120), self.label, font=f_huge, fill=(255, 255, 255))
        
        # Overlapping highlight box
        f_sub = FONTS["display_condensed"](36)
        sub_text = "AGGRESSIVE STATE DETECTED"
        bbox = draw.textbbox((margin + 16, 305), sub_text, font=f_sub)
        draw.rectangle([margin, 290, bbox[2] + 20, 348], fill=(204, 255, 0))
        draw.text((margin + 16, 305), sub_text, font=f_sub, fill=(0, 0, 0))

        # Face Treatment: High contrast thresholded dither with electric lime duotone
        face_w, face_h = 750, 750
        face_x, face_y = margin, 380
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(2.2)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # High-contrast duotone (black & neon lime)
            duo = duotone_filter(f_proc, (10, 10, 12), (204, 255, 0))
            poster.paste(duo, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(40, 40, 40))
            
        # Heavy border with corner tabs
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(255, 255, 255), width=4)
        tab_sz = 30
        draw.rectangle([face_x - 5, face_y - 5, face_x + tab_sz, face_y + tab_sz], fill=(204, 255, 0))
        draw.rectangle([face_x + face_w - tab_sz, face_y + face_h - tab_sz, face_x + face_w + 5, face_y + face_h + 5], fill=(204, 255, 0))

        # Right-side Brutalist Telemetry Stack
        tx = face_x + face_w + 30
        draw.rectangle([tx, face_y, W - margin, face_y + 260], fill=(255, 255, 255))
        draw.text((tx + 16, face_y + 20), "REACTION", font=FONTS["mono"](14), fill=(0, 0, 0))
        draw.text((tx + 16, face_y + 45), f"{self.score}%", font=FONTS["display_condensed"](90), fill=(0, 0, 0))
        draw.text((tx + 16, face_y + 145), "FORCE FACTOR", font=FONTS["mono"](14), fill=(100, 100, 100))
        draw.rectangle([tx + 16, face_y + 175, tx + 16 + int(260 * (self.score / 100.0)), face_y + 200], fill=(204, 255, 0))
        draw.rectangle([tx + 16, face_y + 175, tx + 276, face_y + 200], outline=(0, 0, 0), width=2)
        draw.text((tx + 16, face_y + 215), "THRESHOLD EXCEEDED", font=FONTS["mono"](13), fill=(200, 0, 0))

        # Barcode & Stamp
        draw_barcode(draw, tx, face_y + 300, width=int(W - margin - tx), height=55, fill=(255, 255, 255))
        draw.text((tx, face_y + 365), f"SYS.LOG // {self.timestamp}", font=FONTS["mono"](12), fill=(180, 180, 180))

        # Technical specification box
        draw.rectangle([tx, face_y + 420, W - margin, face_y + face_h], outline=(204, 255, 0), width=2)
        specs = [
            f"ID: {self.design_id}",
            "MODE: BRUTALIST",
            "GRID: UNREGULATED",
            "CONTRAST: MAXIMAL",
            "DISTORTION: +18dB",
            "STATUS: CRITICAL",
        ]
        sy = face_y + 440
        for s in specs:
            draw.text((tx + 16, sy), s, font=FONTS["mono"](14), fill=(255, 255, 255))
            sy += 42

        # Bottom Manifesto Banner
        bot_y = face_y + face_h + 40
        draw.rectangle([margin, bot_y, W - margin, bot_y + 110], fill=(204, 255, 0))
        draw.text((margin + 20, bot_y + 20), "REACTIONARY / EXPERIMENTAL GRAPHIC PRODUCTION",
                  font=FONTS["display_condensed"](34), fill=(0, 0, 0))
        draw.text((margin + 20, bot_y + 65), "FORM FOLLOWS RAW EMOTION. NO ORNAMENTATION WITHOUT FRICTION.",
                  font=FONTS["mono"](15), fill=(0, 0, 0))

        # Add heavy film grain
        poster = add_film_grain(poster, intensity=0.18)
        return poster


class Y2KRenderer(PosterRenderer):
    """
    Y2K / Cyber Chrome Style:
    - Deep midnight blue/black with cyan (#00F0FF), hot pink (#FF007A), chrome white
    - 4-pointed metallic stars, 3D wireframe perspective grids, lens flare highlights
    - Futuristic extended sans, liquid cyber frames
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (7, 11, 25))
        draw = ImageDraw.Draw(poster)
        
        # Perspective wireframe grid at bottom
        grid_start_y = H - 420
        for gy in range(grid_start_y, H, 28):
            alpha = int(40 + (gy - grid_start_y) * 0.4)
            draw.line([0, gy, W, gy], fill=(0, 180, 240), width=1)
        
        # Converging perspective lines
        vanishing_pt = (W // 2, grid_start_y - 100)
        for vx in range(-200, W + 200, 90):
            draw.line([vanishing_pt[0], vanishing_pt[1], vx, H], fill=(0, 120, 200), width=1)
            
        # Top cyber navigation
        margin = 55
        draw.text((margin, 50), "/// CYBER.GEN.00 // Y2K LAB", font=FONTS["mono"](16), fill=(0, 240, 255))
        draw.text((W - margin - 220, 50), "NET.ARCHIVE.SYS", font=FONTS["mono"](16), fill=(255, 0, 122))
        draw.line([margin, 80, W - margin, 80], fill=(0, 240, 255), width=2)
        
        # Chrome 4-point stars
        draw_chrome_star(draw, margin + 40, 130, radius=32, fill=(255, 255, 255))
        draw_chrome_star(draw, W - margin - 80, 240, radius=48, fill=(0, 240, 255))
        draw_chrome_star(draw, W - 120, 780, radius=28, fill=(255, 0, 122))

        # Main Display Title with Cyber glow shadow
        f_title = FONTS["display_condensed"](125)
        # Shadow / Glow offset
        draw.text((margin + 4, 114), self.label, font=f_title, fill=(255, 0, 122))
        draw.text((margin - 4, 106), self.label, font=f_title, fill=(0, 240, 255))
        draw.text((margin, 110), self.label, font=f_title, fill=(255, 255, 255))
        
        draw.text((margin, 255), f"HYPERSTITION PROTOCOL // {self.design_id}", font=FONTS["mono"](18), fill=(0, 240, 255))

        # Center Face Display with Chromatic Aberration and Cyber Frame
        face_w, face_h = 760, 680
        face_x = (W - face_w) // 2
        face_y = 310
        
        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.4)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # Apply chromatic aberration
            f_proc = chromatic_aberration(f_proc, offset=8)
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(20, 30, 60))
            
        # Cyber oval / bracket overlays
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(0, 240, 255), width=2)
        draw.rectangle([face_x + 10, face_y + 10, face_x + face_w - 10, face_y + face_h - 10], outline=(255, 0, 122), width=1)
        
        # Cyber corner brackets
        bw = 40
        draw.line([face_x - 10, face_y - 10, face_x - 10 + bw, face_y - 10], fill=(255, 255, 255), width=3)
        draw.line([face_x - 10, face_y - 10, face_x - 10, face_y - 10 + bw], fill=(255, 255, 255), width=3)
        draw.line([face_x + face_w + 10, face_y + face_h + 10, face_x + face_w + 10 - bw, face_y + face_h + 10], fill=(255, 255, 255), width=3)
        draw.line([face_x + face_w + 10, face_y + face_h + 10, face_x + face_w + 10, face_y + face_h + 10 - bw], fill=(255, 255, 255), width=3)

        # Bottom HUD Controls
        hud_y = face_y + face_h + 40
        # Pill Badges
        draw.rounded_rectangle([margin, hud_y, margin + 260, hud_y + 60], radius=30, fill=(0, 240, 255))
        draw.text((margin + 32, hud_y + 18), f"SHOCK INDEX: {self.score}%", font=FONTS["display_condensed"](24), fill=(0, 0, 0))
        
        draw.rounded_rectangle([margin + 280, hud_y, margin + 560, hud_y + 60], radius=30, outline=(255, 0, 122), width=2)
        draw.text((margin + 315, hud_y + 18), "NEURAL SYNC: OPTIMAL", font=FONTS["display_condensed"](24), fill=(255, 0, 122))

        draw_barcode(draw, W - margin - 220, hud_y + 8, width=220, height=45, fill=(0, 240, 255))

        # Bottom Floating Lyrics / Manifesto
        draw.text((margin, H - 90), "FUTURE IS NOW // DIGITAL ORGANISM EVOLUTION // ALL RIGHTS PRESERVED 2000-2026",
                  font=FONTS["mono"](14), fill=(160, 180, 220))
        
        return poster


class EditorialRenderer(PosterRenderer):
    """
    Luxury Editorial / Haute Couture Style:
    - Sophisticated obsidian, champagne gold (#D4AF37), and warm alabaster
    - High-fashion serif typography with extreme scale contrast
    - Generous negative space, fine editorial hairline rules, photographic grain
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (18, 18, 20))
        draw = ImageDraw.Draw(poster)
        
        margin = 70
        
        # Elegant Magazine Folio
        draw.text((margin, 60), "REACTIONARY MAGAZINE", font=FONTS["display_serif"](20), fill=(212, 175, 55))
        draw.text((W // 2 - 60, 60), "ISSUE N° 42", font=FONTS["mono"](16), fill=(160, 160, 160))
        draw.text((W - margin - 160, 60), "AUTUMN / WINTER", font=FONTS["body_sans"](16), fill=(200, 200, 200))
        draw.line([margin, 95, W - margin, 95], fill=(212, 175, 55), width=1)
        
        # Giant Dramatic Headline
        f_display = FONTS["display_serif"](130)
        draw.text((margin - 5, 120), self.label.title(), font=f_display, fill=(245, 245, 245))
        
        # Sub-headline
        f_italic = FONTS["body_serif"](26)
        draw.text((margin, 280), "A Study in Human Poise, Internal Architecture and Quiet Authority",
                  font=f_italic, fill=(212, 175, 55))

        # Face Portrait: Luxurious Monochrome with deep velvety shadows
        face_w = 680
        face_h = 880
        face_x = margin
        face_y = 350
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.3)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # Warm duotone toning (obsidian to warm ivory)
            f_proc = duotone_filter(f_proc, (18, 18, 20), (242, 238, 230))
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(35, 35, 38))
            
        # Subtle hairline frame
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(212, 175, 55), width=1)

        # Right Column Editorial Paragraph & Typography Spec
        rx = face_x + face_w + 45
        draw.text((rx, face_y + 10), "VOL. VII — PORTRAIT ESSAY", font=FONTS["mono"](13), fill=(212, 175, 55))
        draw.line([rx, face_y + 35, W - margin, face_y + 35], fill=(100, 100, 100), width=1)
        
        essay = (
            "Confidence is neither loud nor hurried. It inhabits the space between perception "
            "and reaction with absolute stillness. Through micro-movements of the brow and chin, "
            "the subject communicates an immutable composure."
        )
        
        # Wrap essay text
        words = essay.split()
        cur_y = face_y + 60
        line = ""
        for w in words:
            test = line + (" " if line else "") + w
            if len(test) > 28:
                draw.text((rx, cur_y), line, font=FONTS["body_serif"](17), fill=(210, 210, 210))
                cur_y += 28
                line = w
            else:
                line = test
        if line:
            draw.text((rx, cur_y), line, font=FONTS["body_serif"](17), fill=(210, 210, 210))
            cur_y += 35

        # Metadata Card
        draw.rectangle([rx, cur_y + 30, W - margin, cur_y + 260], outline=(70, 70, 75), width=1)
        draw.text((rx + 20, cur_y + 50), "SUBJECT METRICS", font=FONTS["mono"](13), fill=(160, 160, 160))
        draw.text((rx + 20, cur_y + 80), f"SCORE: {self.score}%", font=FONTS["display_serif"](36), fill=(212, 175, 55))
        draw.text((rx + 20, cur_y + 140), f"SPEC: {self.design_id}", font=FONTS["mono"](15), fill=(240, 240, 240))
        draw.text((rx + 20, cur_y + 175), f"TIME: {self.timestamp}", font=FONTS["mono"](12), fill=(140, 140, 140))
        draw.text((rx + 20, cur_y + 205), "EDITION: 1 OF 1 MONOPRINT", font=FONTS["mono"](12), fill=(140, 140, 140))

        # Bottom quote
        draw.line([margin, H - 140, W - margin, H - 140], fill=(212, 175, 55), width=1)
        draw.text((W - margin - 140, H - 165), self.design_id, font=FONTS["mono"](14), fill=(212, 175, 55))
        draw.text((margin, H - 105), f"“WHEN THE REACTION SPEAKS WITHOUT WORDS, DESIGN BECOMES ITS VOICE.”",
                  font=FONTS["display_serif"](20), fill=(240, 240, 240))

        # Grain
        poster = add_film_grain(poster, intensity=0.10)
        return poster


class MaximalistRenderer(PosterRenderer):
    """
    Pop / Maximalist Style:
    - Vibrant saturated palette: canary yellow (#FFD600), hot magenta (#FF2A85), cyan (#00E5FF)
    - Playful chunky typography, stickers, decorative geometric circles and organic shapes
    - High energy, layered celebratory composition
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (255, 214, 0))  # Vivid Yellow
        draw = ImageDraw.Draw(poster)
        
        # Colorful geometric background layers
        draw.polygon([(0, 0), (W, 0), (W, 400), (0, 650)], fill=(255, 42, 133))  # Magenta angle
        draw.ellipse([W - 350, 200, W + 250, 800], fill=(0, 229, 255))  # Cyan circle

        margin = 55
        
        # Floating Pop Stickers
        # Sticker 1 (Top Left)
        draw.rounded_rectangle([margin, 40, margin + 280, 100], radius=15, fill=(0, 0, 0))
        draw.text((margin + 25, 55), "★ MAXIMUM REACTION ★", font=FONTS["display_condensed"](24), fill=(255, 214, 0))
        
        # Giant Headline with 3D drop shadow
        f_title = FONTS["display_heavy"](145)
        text = self.label
        # Black 3D extrusion
        for offset in range(12, 0, -2):
            draw.text((margin + offset, 115 + offset), text, font=f_title, fill=(0, 0, 0))
        draw.text((margin, 115), text, font=f_title, fill=(255, 255, 255))
        
        # Sub badge
        draw.rounded_rectangle([margin, 290, margin + 380, 350], radius=12, fill=(0, 229, 255))
        draw.text((margin + 20, 305), f"HAPPINESS ENERGY // {self.score}% CONFIRMED",
                  font=FONTS["display_condensed"](26), fill=(0, 0, 0))

        # Face Presentation: Framed with heavy border and colorful sticker frame
        face_w, face_h = 720, 720
        face_x = (W - face_w) // 2
        face_y = 380
        
        # Drop shadow for face box
        draw.rectangle([face_x + 16, face_y + 16, face_x + face_w + 16, face_y + face_h + 16], fill=(0, 0, 0))
        
        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.6)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.2)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(255, 255, 255))
            
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(0, 0, 0), width=6)

        # Decorative Corner Stickers
        draw.ellipse([face_x - 30, face_y - 30, face_x + 70, face_y + 70], fill=(255, 42, 133))
        draw.text((face_x - 5, face_y - 5), "JOY", font=FONTS["display_condensed"](30), fill=(255, 255, 255))

        draw.rounded_rectangle([face_x + face_w - 120, face_y + face_h - 40, face_x + face_w + 40, face_y + face_h + 30],
                               radius=10, fill=(0, 229, 255))
        draw.text((face_x + face_w - 95, face_y + face_h - 25), "SMILE :)", font=FONTS["display_condensed"](24), fill=(0, 0, 0))

        # Bottom Pop Information Band
        bot_y = face_y + face_h + 40
        draw.rectangle([margin, bot_y, W - margin, bot_y + 160], fill=(0, 0, 0))
        
        draw.text((margin + 30, bot_y + 25), f"DESIGN CODE: {self.design_id}", font=FONTS["mono"](20), fill=(255, 214, 0))
        draw.text((margin + 30, bot_y + 65), "MAXIMALIST POP EXPLOSION // HIGH SATURATION REACTION ENGINE",
                  font=FONTS["display_condensed"](26), fill=(255, 255, 255))
        draw.text((margin + 30, bot_y + 110), f"TIMESTAMP: {self.timestamp} // ALL REALITY OPTIMIZED",
                  font=FONTS["mono"](14), fill=(0, 229, 255))
        
        draw_barcode(draw, W - margin - 220, bot_y + 35, width=190, height=80, fill=(255, 255, 255))

        return poster


class MinimalistRenderer(PosterRenderer):
    """
    Hyper Minimalist Style:
    - Extreme visual restraint, quiet charcoal and porcelain white
    - Abundant whitespace, microscopic technical typography, delicate hairline crosshairs
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (248, 248, 247))
        draw = ImageDraw.Draw(poster)
        
        margin = 80
        
        # Razor thin frame
        draw.rectangle([margin, margin, W - margin, H - margin], outline=(220, 220, 220), width=1)
        
        # Alignment crosshairs at corners
        draw_crosshair(draw, margin, margin, 12, fill=(180, 180, 180))
        draw_crosshair(draw, W - margin, margin, 12, fill=(180, 180, 180))
        draw_crosshair(draw, margin, H - margin, 12, fill=(180, 180, 180))
        draw_crosshair(draw, W - margin, H - margin, 12, fill=(180, 180, 180))

        # Understated header
        draw.text((margin + 30, margin + 30), "REACTIONARY — FORMAL STUDY", font=FONTS["mono"](13), fill=(120, 120, 120))
        draw.text((W - margin - 150, margin + 30), self.design_id, font=FONTS["mono"](13), fill=(120, 120, 120))

        # Subtle Single Headline
        draw.text((margin + 30, margin + 140), self.label.lower(), font=FONTS["body_sans"](72), fill=(30, 30, 30))
        draw.text((margin + 30, margin + 230), f"measured state // confidence ratio {self.score / 100:.2f}",
                  font=FONTS["mono"](14), fill=(140, 140, 140))

        # Precision Centered Portrait
        face_w = 620
        face_h = 760
        face_x = (W - face_w) // 2
        face_y = margin + 320
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.1)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc.convert("RGB"), (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(230, 230, 230))
            
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(200, 200, 200), width=1)

        # Quiet Micro-Typography
        bot_y = H - margin - 80
        draw.line([margin + 30, bot_y - 20, W - margin - 30, bot_y - 20], fill=(230, 230, 230), width=1)
        draw.text((margin + 30, bot_y), "01. ESSENCE", font=FONTS["mono"](12), fill=(100, 100, 100))
        draw.text((margin + 240, bot_y), "02. EQUILIBRIUM", font=FONTS["mono"](12), fill=(100, 100, 100))
        draw.text((margin + 480, bot_y), "03. RESTRAINT", font=FONTS["mono"](12), fill=(100, 100, 100))
        draw.text((W - margin - 180, bot_y), self.timestamp, font=FONTS["mono"](12), fill=(120, 120, 120))
        
        return poster


class CyberpunkRenderer(PosterRenderer):
    """
    Cyberpunk Telemetry Style:
    - Pitch darkness, neon cyan (#00FFCC) and magenta (#FF0055)
    - Digital scanline raster, HUD bounding boxes, coordinates and kanji accents
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (6, 8, 14))
        draw = ImageDraw.Draw(poster)
        
        # Subtle horizontal scanlines
        for sy in range(0, H, 4):
            draw.line([0, sy, W, sy], fill=(10, 14, 24), width=1)
            
        margin = 45
        
        # Cyberpunk Tech HUD Header
        draw.text((margin, 35), "NEO-METROPOLIS BIO-SENSING TERMINAL // REV 4.2", font=FONTS["mono"](14), fill=(0, 255, 204))
        draw.text((W - margin - 180, 35), "STATUS: LOCKED", font=FONTS["mono"](14), fill=(255, 0, 85))
        draw.line([margin, 60, W - margin, 60], fill=(0, 255, 204), width=1)

        # Huge Glitched Headline
        f_title = FONTS["display_heavy"](135)
        # Red/Blue offset glitch
        draw.text((margin + 6, 88), self.label, font=f_title, fill=(255, 0, 85))
        draw.text((margin - 6, 82), self.label, font=f_title, fill=(0, 255, 204))
        draw.text((margin, 85), self.label, font=f_title, fill=(255, 255, 255))
        
        # Japanese Tech glyphs + metadata
        draw.text((margin, 230), f"感情感知 // EMOTIONAL SCAN ID: {self.design_id} // LOC: 35.6762° N",
                  font=FONTS["mono"](16), fill=(0, 255, 204))

        # Face Display: Cyan/Magenta Split Duotone with scanlines
        face_w, face_h = 780, 720
        face_x = (W - face_w) // 2
        face_y = 290
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.6)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # Cyber duotone (dark purple to cyan)
            f_proc = duotone_filter(f_proc, (25, 5, 45), (0, 255, 204))
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(20, 20, 35))
            
        # HUD Reticle overlays over face
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(0, 255, 204), width=2)
        # Crosshair in center of face
        draw_crosshair(draw, face_x + face_w // 2, face_y + face_h // 2, size=32, fill=(255, 0, 85))
        
        # HUD Corner accents
        cw = 50
        draw.line([face_x, face_y + 30, face_x + 30, face_y], fill=(255, 0, 85), width=3)
        draw.line([face_x + face_w - 30, face_y + face_h, face_x + face_w, face_y + face_h - 30], fill=(255, 0, 85), width=3)

        # Telemetry Gauges Below
        hud_y = face_y + face_h + 30
        draw.rectangle([margin, hud_y, margin + 320, hud_y + 110], outline=(0, 255, 204), width=1)
        draw.text((margin + 20, hud_y + 16), "NEURAL FLUX DENSITY", font=FONTS["mono"](12), fill=(180, 180, 180))
        draw.text((margin + 20, hud_y + 40), f"{self.score}%", font=FONTS["display_heavy"](48), fill=(0, 255, 204))

        draw.rectangle([margin + 340, hud_y, W - margin, hud_y + 110], outline=(255, 0, 85), width=1)
        draw.text((margin + 360, hud_y + 16), "TARGET TELEMETRY // BIO-LOCK ENGAGED", font=FONTS["mono"](13), fill=(255, 0, 85))
        draw.text((margin + 360, hud_y + 45), f"MATCH CONFIDENCE: {self.score}% // LATENCY: 12ms", font=FONTS["mono"](15), fill=(255, 255, 255))
        draw.text((margin + 360, hud_y + 75), f"SYS ID: {self.design_id} // TIMESTAMP: {self.timestamp}", font=FONTS["mono"](13), fill=(0, 255, 204))

        # Bottom Barcode & Cyber coordinates
        draw_barcode(draw, margin, H - 70, width=280, height=35, fill=(0, 255, 204))
        draw.text((margin + 310, H - 55), "CYBER-ORGANIC VISION ARCHIVE // 2077 ED.", font=FONTS["mono"](14), fill=(120, 140, 160))

        return poster


class RetroRenderer(PosterRenderer):
    """
    Retro Risograph / Acid Print Style:
    - 2-color riso print separation with authentic misregistration
    - Coarse halftone texture, warm vintage palette (mustard #F5A623, teal #008080, terracotta #D9534F)
    - 70s-80s editorial typography and badge ornaments
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        # Cream paper background
        poster = Image.new("RGB", (W, H), (244, 240, 228))
        draw = ImageDraw.Draw(poster)
        
        margin = 60
        
        # Heavy terracotta border
        draw.rectangle([margin, margin, W - margin, H - margin], outline=(217, 83, 79), width=4)
        draw.rectangle([margin + 10, margin + 10, W - margin - 10, H - margin - 10], outline=(0, 128, 128), width=2)

        # Header Badge
        draw.text((margin + 30, margin + 25), "★ RISOGRAPH PRINT ARCHIVE ★", font=FONTS["mono"](16), fill=(0, 128, 128))
        draw.text((W - margin - 200, margin + 25), "VOL. 78 // NO. 4", font=FONTS["mono"](16), fill=(217, 83, 79))

        # Retro Display Title with Misregistration Offset
        f_title = FONTS["display_condensed"](130)
        # Offset teal shadow
        draw.text((margin + 34, margin + 74), self.label, font=f_title, fill=(0, 128, 128))
        # Main terracotta layer
        draw.text((margin + 28, margin + 70), self.label, font=f_title, fill=(217, 83, 79))

        # Face Treatment: Risograph 2-Color Halftone
        face_w, face_h = 720, 700
        face_x = (W - face_w) // 2
        face_y = margin + 240
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.8)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # Riso duotone: deep teal and terracotta
            riso = duotone_filter(f_proc, (0, 70, 70), (245, 166, 35))
            poster.paste(riso, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(200, 200, 180))
            
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(217, 83, 79), width=3)

        # Vintage Circular Badge
        badge_r = 85
        bx, by = face_x + face_w - 40, face_y + face_h - 40
        draw.ellipse([bx - badge_r, by - badge_r, bx + badge_r, by + badge_r], fill=(245, 166, 35))
        draw.ellipse([bx - badge_r + 5, by - badge_r + 5, bx + badge_r - 5, by + badge_r - 5], outline=(217, 83, 79), width=2)
        draw.text((bx - 55, by - 40), f"{self.score}%", font=FONTS["display_condensed"](42), fill=(0, 0, 0))
        draw.text((bx - 50, by + 10), "GENUINE", font=FONTS["mono"](14), fill=(217, 83, 79))
        draw.text((bx - 52, by + 28), "REACTION", font=FONTS["mono"](12), fill=(0, 0, 0))

        # Bottom Editorial Information
        bot_y = face_y + face_h + 50
        draw.text((margin + 30, bot_y), f"PLATE SPECIFICATION: {self.design_id}", font=FONTS["mono"](18), fill=(0, 128, 128))
        draw.text((margin + 30, bot_y + 35), "PRINTED VIA DUAL-DRUM ROTARY STENCIL DUPLICATOR",
                  font=FONTS["body_sans"](20), fill=(217, 83, 79))
        draw.text((margin + 30, bot_y + 70), f"RECORDED AT {self.timestamp} ON ACID-FREE RAG PAPER",
                  font=FONTS["mono"](14), fill=(100, 100, 100))

        # Add coarse film grain
        poster = add_film_grain(poster, intensity=0.14)
        return poster


class DesiMaximalismRenderer(PosterRenderer):
    """
    Desi Maximalism Style:
    - Rich marigold orange (#FF9933), royal peacock indigo (#131E3A), and ruby crimson (#C8102E)
    - Intricate ornamental corner borders, vibrant framed headers
    - Truck-art inspired typography, celebratory decorative accents
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (19, 30, 58))  # Royal Peacock Indigo
        draw = ImageDraw.Draw(poster)
        
        margin = 55
        
        # Multi-tiered decorative border
        draw.rectangle([margin, margin, W - margin, H - margin], outline=(255, 153, 51), width=5)
        draw.rectangle([margin + 12, margin + 12, W - margin - 12, H - margin - 12], outline=(200, 16, 46), width=3)
        draw.rectangle([margin + 20, margin + 20, W - margin - 20, H - margin - 20], outline=(255, 215, 0), width=1)

        # Ornate Corner Flourishes
        for cx, cy in [(margin + 20, margin + 20), (W - margin - 20, margin + 20),
                       (margin + 20, H - margin - 20), (W - margin - 20, H - margin - 20)]:
            draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(255, 153, 51))
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(200, 16, 46))

        # Grand Header Banner
        banner_y = margin + 35
        draw.rectangle([margin + 30, banner_y, W - margin - 30, banner_y + 75], fill=(200, 16, 46))
        draw.text((margin + 50, banner_y + 20), "शुभ विचार // REACTIONARY DESI MAXIMA",
                  font=FONTS["display_condensed"](32), fill=(255, 215, 0))
        draw.text((W - margin - 220, banner_y + 24), self.design_id, font=FONTS["mono"](20), fill=(255, 255, 255))

        # Main Majestic Headline
        f_head = FONTS["display_heavy"](135)
        # Gold drop shadow
        draw.text((margin + 35, banner_y + 85), self.label, font=f_head, fill=(255, 153, 51))
        draw.text((margin + 30, banner_y + 80), self.label, font=f_head, fill=(255, 255, 255))

        # Portrait with Ornate Frame
        face_w, face_h = 720, 700
        face_x = (W - face_w) // 2
        face_y = banner_y + 250
        
        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.7)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.2)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(40, 50, 80))
            
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(255, 215, 0), width=6)
        draw.rectangle([face_x - 8, face_y - 8, face_x + face_w + 8, face_y + face_h + 8], outline=(200, 16, 46), width=3)

        # Bottom Festive Information Pill
        bot_y = face_y + face_h + 40
        draw.rounded_rectangle([margin + 30, bot_y, W - margin - 30, bot_y + 110], radius=20, fill=(255, 153, 51))
        draw.text((margin + 60, bot_y + 18), f"REACTION INTENSITY: {self.score}% // TOTAL JOY",
                  font=FONTS["display_condensed"](34), fill=(19, 30, 58))
        draw.text((margin + 60, bot_y + 65), f"DECORATIVE CELEBRATION SPEC // {self.timestamp}",
                  font=FONTS["mono"](16), fill=(200, 16, 46))

        return poster


class ExperimentalRenderer(PosterRenderer):
    """
    Dramatic Experimental Style (Sadness / Overwhelmed):
    - Dark moody atmospheric palette, fractured grid, kinetic typography slices
    - High-contrast dramatic face treatment, layered text masks, deep grain
    """
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (13, 13, 17))
        draw = ImageDraw.Draw(poster)
        
        margin = 55
        
        # Asymmetrical diagonal guides
        draw.line([margin, 300, W - margin, 180], fill=(45, 45, 55), width=1)
        draw.line([margin, 950, W - margin, 1100], fill=(45, 45, 55), width=1)

        # Header
        draw.text((margin, 50), "EXPERIMENTAL POSTER LABORATORY", font=FONTS["mono"](14), fill=(140, 140, 160))
        draw.text((W - margin - 180, 50), self.design_id, font=FONTS["mono"](14), fill=(121, 40, 202))

        # Massive Sliced & Layered Headline
        f_title = FONTS["display_condensed"](165)
        # Background ghostly offset
        draw.text((margin - 20, 110), self.label, font=f_title, fill=(28, 28, 38))
        draw.text((margin, 125), self.label, font=f_title, fill=(240, 240, 245))

        # Dramatic Face Treatment: Deep shadowy B&W with high grain and crop
        face_w, face_h = 760, 760
        face_x = (W - face_w) // 2
        face_y = 310
        
        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.8)
            f_proc = ImageOps.fit(f_proc, (face_w, face_h), method=Image.Resampling.LANCZOS)
            # Moody violet duotone (obsidian to muted indigo)
            f_proc = duotone_filter(f_proc, (13, 13, 17), (180, 185, 215))
            poster.paste(f_proc, (face_x, face_y))
        else:
            draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], fill=(25, 25, 32))
            
        # Experimental framing (offset floating rectangles)
        draw.rectangle([face_x, face_y, face_x + face_w, face_y + face_h], outline=(121, 40, 202), width=2)
        draw.rectangle([face_x + 25, face_y - 25, face_x + face_w + 25, face_y + face_h - 25], outline=(255, 255, 255), width=1)

        # Overlay vertical typography on face edge
        draw.text((face_x + 15, face_y + 30), "OVERWHELMED STATE ARCHITECTURE", font=FONTS["mono"](14), fill=(255, 255, 255))
        draw.text((face_x + 15, face_y + 60), f"SCORE: {self.score}% // INTERNAL PRESSURE", font=FONTS["mono"](14), fill=(121, 40, 202))

        # Bottom Abstract Grid Metadata
        bot_y = face_y + face_h + 40
        draw.text((margin, bot_y), "01 / DISSOLUTION OF FORM", font=FONTS["mono"](14), fill=(180, 180, 190))
        draw.text((margin, bot_y + 30), "WHEN PERCEPTION SURPASSES PROCESSING CAPACITY, THE INTERFACE FRACTURES.",
                  font=FONTS["display_serif"](24), fill=(220, 220, 230))
        draw.text((margin, bot_y + 75), f"REACTIONARY RES. PROTOCOL // {self.timestamp}", font=FONTS["mono"](13), fill=(100, 100, 120))

        draw_barcode(draw, W - margin - 220, bot_y + 20, width=220, height=45, fill=(240, 240, 245))

        # Heavy Film Grain
        poster = add_film_grain(poster, intensity=0.18)
        return poster


# Registry of renderers
RENDERERS = {
    "Swiss": SwissRenderer,
    "Brutalist": BrutalistRenderer,
    "Y2K": Y2KRenderer,
    "Editorial": EditorialRenderer,
    "Maximalist": MaximalistRenderer,
    "Minimalist": MinimalistRenderer,
    "Cyberpunk": CyberpunkRenderer,
    "Retro": RetroRenderer,
    "Desi Maximalism": DesiMaximalismRenderer,
    "Experimental": ExperimentalRenderer,
}


def generate_poster(frame, emotion_data, style_name="Swiss", box=None, params=None):
    """
    Main entry point to generate a graphic design poster.
    
    Args:
      frame: OpenCV BGR image from webcam (or None).
      emotion_data: Dict containing emotion, label, confidence, etc.
      style_name: One of the 10 supported styles.
      box: (x0, y0, x1, y1) face bounding box tuple.
      params: Dict of optional tuning parameters (grain, contrast, etc.).
      
    Returns:
      PIL Image of the generated poster.
    """
    style = style_name if style_name in RENDERERS else "Swiss"
    
    # Extract & prepare cropped face if frame is supplied
    face_img = None
    if frame is not None:
        try:
            face_img = crop_face_area(frame, box)
        except Exception as e:
            print(f"Face crop error: {e}")
            face_img = None
            
    render_data = {
        "emotion": emotion_data.get("emotion", "NEUTRAL"),
        "label": emotion_data.get("label", "UNIMPRESSED"),
        "style": style,
        "confidence": emotion_data.get("confidence", 85),
        "design_id": f"#RX-{random.randint(100, 999)}",
        "face_img": face_img,
        "params": params or {},
    }
    
    renderer_cls = RENDERERS[style]
    renderer = renderer_cls(render_data)
    poster = renderer.render()
    return poster, render_data
