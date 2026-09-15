#!/usr/bin/env python3
"""
poster_generator.py — Graphic design meme synthesis engine for REACTIONARY.

Generates visually rich, social-media-ready meme posters (1200 x 1600)
based on detected or overridden facial emotions across 10 styles:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Fear
  - Disgust
  - Neutral
  - Confused
  - Excited
  - Embarrassed

Performs final brand compositing:
  - Bottom-Left: Graphica Club Logo Watermark (proportions preserved, high contrast)
  - Bottom-Right: Graphica Instagram QR Code (proportions preserved, fully scannable)
"""

import os
import random
import time
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

POSTER_WIDTH = 1200
POSTER_HEIGHT = 1600

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR = "C:/Windows/Fonts"

# Helper to find and load available system fonts
def get_font(candidates, size):
    for name in candidates:
        p = os.path.join(FONTS_DIR, name)
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.truetype(os.path.join(FONTS_DIR, "arial.ttf"), size)
    except Exception:
        return ImageFont.load_default()

FONTS = {
    "impact": lambda s: get_font(["impact.ttf", "arialbd.ttf"], s),
    "heavy": lambda s: get_font(["impact.ttf", "arialbd.ttf", "segoeuib.ttf"], s),
    "condensed": lambda s: get_font(["ARIALNB.TTF", "impact.ttf", "arialbd.ttf"], s),
    "sans_bold": lambda s: get_font(["arialbd.ttf", "segoeuib.ttf", "trebucbd.ttf"], s),
    "sans": lambda s: get_font(["arial.ttf", "segoeui.ttf"], s),
    "serif_bold": lambda s: get_font(["georgiab.ttf", "timesbd.ttf"], s),
    "serif": lambda s: get_font(["georgia.ttf", "times.ttf"], s),
    "mono_bold": lambda s: get_font(["consolab.ttf", "courbd.ttf"], s),
    "mono": lambda s: get_font(["consola.ttf", "cour.ttf"], s),
    "comic": lambda s: get_font(["comicbd.ttf", "comic.ttf", "arialbd.ttf"], s),
}

# --- Drawing & Visual Effect Utilities ---

def add_film_grain(img, intensity=0.10):
    """Adds subtle film grain texture."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def duotone_filter(img, dark_rgb, light_rgb):
    """Converts image to stylized two-color gradient map."""
    gray = img.convert("L")
    arr = np.array(gray).astype(np.float32) / 255.0
    r = (dark_rgb[0] * (1.0 - arr) + light_rgb[0] * arr).astype(np.uint8)
    g = (dark_rgb[1] * (1.0 - arr) + light_rgb[1] * arr).astype(np.uint8)
    b = (dark_rgb[2] * (1.0 - arr) + light_rgb[2] * arr).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def chromatic_aberration(img, offset=7):
    """RGB channel shift for chaotic/shock vibes."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    out = np.zeros_like(arr)
    out[:, offset:w, 0] = arr[:, 0:w - offset, 0]
    out[:, :, 1] = arr[:, :, 1]
    out[:, 0:w - offset, 2] = arr[:, offset:w, 2]
    return Image.fromarray(out)


def crop_face_portrait(frame, box=None):
    """Extracts recognizable face and upper shoulders from camera frame."""
    h, w = frame.shape[:2]
    if box:
        x0, y0, x1, y1 = box
        fw, fh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        crop_w = int(fw * 1.8)
        crop_h = int(fh * 2.3)
        nx0 = max(0, cx - crop_w // 2)
        ny0 = max(0, cy - int(crop_h * 0.45))
        nx1 = min(w, nx0 + crop_w)
        ny1 = min(h, ny0 + crop_h)
        face_roi = frame[ny0:ny1, nx0:nx1]
    else:
        ch, cw = int(h * 0.85), int(h * 0.85 * 0.8)
        nx0 = max(0, (w - cw) // 2)
        ny0 = max(0, (h - ch) // 2)
        face_roi = frame[ny0:ny0 + ch, nx0:nx0 + cw]

    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_text_with_outline(draw, pos, text, font, text_color, outline_color, outline_width=3):
    """Draws classic impact meme text with a bold dark outline."""
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)


def draw_comic_starburst(draw, cx, cy, num_points=12, r_inner=60, r_outer=130, fill=(255, 220, 0), outline=(0, 0, 0)):
    """Draws an explosive comic-book starburst sticker."""
    pts = []
    angle_step = (2 * math.pi) / (num_points * 2)
    for i in range(num_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        ang = i * angle_step
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=fill, outline=outline)


def draw_speech_bubble(draw, x0, y0, x1, y1, tail_pos, fill=(255, 255, 255), outline=(0, 0, 0), width=3):
    """Draws a clean comic speech bubble."""
    draw.rounded_rectangle([x0, y0, x1, y1], radius=20, fill=fill, outline=outline, width=width)
    # Triangle tail
    tx, ty = tail_pos
    bx = (x0 + x1) // 2
    by = y1
    draw.polygon([(bx - 20, by - 2), (bx + 20, by - 2), (tx, ty)], fill=fill)
    draw.line([(bx - 20, by), (tx, ty), (bx + 20, by)], fill=outline, width=width)


# --- 10 DISTINCT EMOTION MEME RENDERERS ---

class BaseMemeRenderer:
    def __init__(self, face_img, emotion_data, params=None):
        self.face_img = face_img
        self.emotion = emotion_data.get("emotion", "HAPPY")
        self.label = emotion_data.get("label", "Happy")
        self.caption = emotion_data.get("default_caption", "POV: YOU GOT THE MEME")
        self.confidence = emotion_data.get("confidence", 90)
        self.params = params or {}
        self.design_id = f"#RX-{random.randint(100, 999)}"
        self.timestamp = time.strftime("%Y.%m.%d // %H:%M")


class HappyMemeRenderer(BaseMemeRenderer):
    """Bright, sunny, wholesome pop meme with joyful stickers and energetic typography."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (255, 222, 40))
        draw = ImageDraw.Draw(poster)

        # Sunburst ray background
        cx, cy = W // 2, 600
        for deg in range(0, 360, 20):
            rad1 = math.radians(deg)
            rad2 = math.radians(deg + 10)
            p1 = (cx + 1400 * math.cos(rad1), cy + 1400 * math.sin(rad1))
            p2 = (cx + 1400 * math.cos(rad2), cy + 1400 * math.sin(rad2))
            draw.polygon([(cx, cy), p1, p2], fill=(255, 198, 10))

        # Top Meme Banner
        draw.rectangle([50, 45, W - 50, 150], fill=(255, 42, 133))
        draw.rectangle([50, 45, W - 50, 150], outline=(0, 0, 0), width=5)
        f_top = FONTS["impact"](64)
        draw.text((75, 60), "POV: YOUR CODE COMPILES ON FIRST TRY", font=f_top, fill=(255, 255, 255))

        # Face Display with Fun Sticker Frame
        fw, fh = 800, 800
        fx, fy = (W - fw) // 2, 220
        draw.rectangle([fx + 18, fy + 18, fx + fw + 18, fy + fh + 18], fill=(0, 0, 0))

        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.6)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.2)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(255, 255, 255))
        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(0, 0, 0), width=8)

        # Floating Stickers
        draw_comic_starburst(draw, fx - 20, fy + 40, num_points=10, r_inner=45, r_outer=95, fill=(0, 229, 255), outline=(0, 0, 0))
        draw.text((fx - 70, fy + 20), "100%", font=FONTS["impact"](44), fill=(0, 0, 0))

        draw_comic_starburst(draw, fx + fw + 20, fy + 120, num_points=12, r_inner=50, r_outer=100, fill=(255, 42, 133), outline=(0, 0, 0))
        draw.text((fx + fw - 35, fy + 95), "VIBES", font=FONTS["impact"](38), fill=(255, 255, 255))

        # Bottom Meme Punchline
        draw.rectangle([50, 1070, W - 50, 1280], fill=(255, 255, 255))
        draw.rectangle([50, 1070, W - 50, 1280], outline=(0, 0, 0), width=6)
        draw.text((80, 1095), "IMMACULATE SEROTONIN DETECTED", font=FONTS["impact"](68), fill=(0, 0, 0))
        draw.text((80, 1185), f"REACTION LEVEL: {self.confidence}% // ALL SYSTEMS PURE JOY", font=FONTS["sans_bold"](32), fill=(255, 42, 133))

        return poster


class SadMemeRenderer(BaseMemeRenderer):
    """Dramatic, over-the-top melancholic meme with rain streaks, cold blues, and dramatic type."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (15, 23, 42))
        draw = ImageDraw.Draw(poster)

        # Dramatic rain streaks
        for rx in range(30, W, 45):
            draw.line([rx, 0, rx - 35, H], fill=(30, 41, 59), width=2)

        # Header
        draw.text((60, 50), "EMOTIONAL DAMAGE ARCHIVE // VOL. 404", font=FONTS["mono_bold"](22), fill=(56, 189, 248))
        f_title = FONTS["serif_bold"](86)
        draw.text((60, 95), "It’s Fine. Everything Is Fine.", font=f_title, fill=(241, 245, 249))

        # Face Display with Moody Vignette & Deep Blue Duotone
        fw, fh = 820, 820
        fx, fy = (W - fw) // 2, 230

        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.4)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            # Cold deep blue duotone
            f_proc = duotone_filter(f_proc, (15, 23, 42), (147, 197, 253))
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(30, 41, 59))
        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(56, 189, 248), width=3)

        # Big Dramatic Teardrop Sticker
        draw.rounded_rectangle([fx + 30, fy + 30, fx + 260, fy + 90], radius=12, fill=(15, 23, 42))
        draw.rectangle([fx + 30, fy + 30, fx + 260, fy + 90], outline=(56, 189, 248), width=2)
        draw.text((fx + 50, fy + 45), f"SADNESS: {self.confidence}%", font=FONTS["mono_bold"](24), fill=(56, 189, 248))

        # Bottom Caption Block
        draw.rectangle([60, 1100, W - 60, 1270], fill=(30, 41, 59))
        draw.rectangle([60, 1100, W - 60, 1270], outline=(100, 116, 139), width=2)
        draw.text((90, 1125), "WHEN YOU DROP THE PRODUCTION DATABASE ON A FRIDAY", font=FONTS["impact"](46), fill=(255, 255, 255))
        draw.text((90, 1195), "“I have made a severe and continuous lapse in judgement.”", font=FONTS["serif"](28), fill=(148, 163, 184))

        poster = add_film_grain(poster, intensity=0.12)
        return poster


class AngryMemeRenderer(BaseMemeRenderer):
    """Rage, aggressive, chaotic meme with brutalist red, neon-lime, hazard blocks, and heavy type."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (14, 14, 16))
        draw = ImageDraw.Draw(poster)

        # Top Hazard Stripes
        stripe_w = 26
        for sx in range(0, W, stripe_w * 2):
            draw.polygon([(sx, 0), (sx + stripe_w, 0), (sx + stripe_w - 20, 30), (sx - 20, 30)], fill=(255, 23, 68))
            draw.polygon([(sx + stripe_w, 0), (sx + stripe_w * 2, 0), (sx + stripe_w * 2 - 20, 30), (sx + stripe_w - 20, 30)], fill=(0, 0, 0))

        # Massive Header
        draw.rectangle([50, 50, W - 50, 120], fill=(255, 23, 68))
        draw.text((70, 62), "WARNING // CRITICAL RAGE OVERFLOW", font=FONTS["mono_bold"](26), fill=(0, 0, 0))

        f_huge = FONTS["impact"](110)
        draw.text((50, 135), "PEACE WAS NEVER AN OPTION", font=f_huge, fill=(255, 255, 255))

        # Face Display with High-Contrast Red & Acid Lime Duotone
        fw, fh = 820, 780
        fx, fy = (W - fw) // 2, 275

        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(2.3)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            f_proc = duotone_filter(f_proc, (20, 0, 10), (255, 23, 68))
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(30, 10, 15))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(204, 255, 0), width=6)

        # Corner Rage Stamps
        draw.rectangle([fx - 15, fy + 50, fx + 160, fy + 115], fill=(204, 255, 0))
        draw.text((fx - 5, fy + 65), "ANGRY!", font=FONTS["impact"](42), fill=(0, 0, 0))

        # Bottom Punchline
        draw.rectangle([50, 1100, W - 50, 1270], fill=(204, 255, 0))
        draw.text((75, 1120), "ABSOLUTELY CRASHING OUT", font=FONTS["impact"](76), fill=(0, 0, 0))
        draw.text((75, 1205), f"TENSION FACTOR: {self.confidence}% // STAND BACK 100 METERS", font=FONTS["mono_bold"](26), fill=(255, 23, 68))

        poster = add_film_grain(poster, intensity=0.15)
        return poster


class SurprisedMemeRenderer(BaseMemeRenderer):
    """Shocked, unexpected pop-art explosion with comic speed lines, cyan/pink gradient, and big text."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (11, 15, 30))
        draw = ImageDraw.Draw(poster)

        # Comic Speed Lines radiating from center
        cx, cy = W // 2, 600
        for i in range(0, 360, 6):
            rad = math.radians(i)
            px = cx + 1200 * math.cos(rad)
            py = cy + 1200 * math.sin(rad)
            draw.line([cx, cy, px, py], fill=(25, 35, 70), width=2)

        # Header Pill
        draw.rounded_rectangle([50, 45, 450, 105], radius=30, fill=(0, 240, 255))
        draw.text((80, 60), "LIVE REACTION: UNPRECEDENTED", font=FONTS["impact"](30), fill=(0, 0, 0))

        f_huge = FONTS["impact"](125)
        draw_text_with_outline(draw, (50, 115), "WAIT... WHAT?!", f_huge, (255, 255, 255), (255, 0, 122), outline_width=6)

        # Face Display with Chromatic Aberration & Pop Glitch
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 275

        if self.face_img:
            f_proc = ImageOps.fit(self.face_img, (fw, fh), method=Image.Resampling.LANCZOS)
            f_proc = chromatic_aberration(f_proc, offset=9)
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(30, 40, 60))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(0, 240, 255), width=5)

        # Speech Bubble
        draw_speech_bubble(draw, fx + fw - 280, fy + 40, fx + fw + 30, fy + 180, (fx + fw - 70, fy + 220), fill=(255, 0, 122), outline=(255, 255, 255), width=3)
        draw.text((fx + fw - 250, fy + 80), "BRUH NO WAY", font=FONTS["impact"](38), fill=(255, 255, 255))

        # Bottom Meme Punchline
        draw.rectangle([50, 1100, W - 50, 1270], fill=(0, 240, 255))
        draw.text((75, 1125), "REALITY.EXE HAS UNEXPECTEDLY STOPPED", font=FONTS["impact"](54), fill=(0, 0, 0))
        draw.text((75, 1205), f"SHOCK INDEX: {self.confidence}% // ALL CALCULATIONS SHATTERED", font=FONTS["mono_bold"](26), fill=(255, 0, 122))

        return poster


class FearMemeRenderer(BaseMemeRenderer):
    """Scared, panic meme with horror vignette, caution tape, and trembling text."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (10, 14, 20))
        draw = ImageDraw.Draw(poster)

        # Caution Tape Stripes Top & Bottom
        draw.rectangle([0, 40, W, 85], fill=(234, 179, 8))
        for x in range(0, W, 50):
            draw.polygon([(x, 40), (x + 25, 40), (x + 5, 85), (x - 20, 85)], fill=(0, 0, 0))

        f_huge = FONTS["impact"](110)
        draw.text((50, 100), "PANIK MODE ACTIVATED", font=f_huge, fill=(239, 68, 68))

        # Face Display with Eerie Green Noir Lighting
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 245

        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.8)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            f_proc = duotone_filter(f_proc, (10, 20, 15), (74, 222, 128))
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(15, 25, 20))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(239, 68, 68), width=5)

        # Distress Box
        draw.rounded_rectangle([fx + 30, fy + 30, fx + 280, fy + 95], radius=10, fill=(239, 68, 68))
        draw.text((fx + 50, fy + 45), "THREAT DETECTED", font=FONTS["impact"](34), fill=(255, 255, 255))

        # Bottom Meme Punchline
        draw.rectangle([50, 1080, W - 50, 1270], fill=(239, 68, 68))
        draw.text((75, 1105), "MOM PLEASE COME PICK ME UP I'M SCARED", font=FONTS["impact"](54), fill=(255, 255, 255))
        draw.text((75, 1185), f"FEAR QUOTIENT: {self.confidence}% // SURVIVAL CHANCE: 12%", font=FONTS["mono_bold"](26), fill=(0, 0, 0))

        poster = add_film_grain(poster, intensity=0.16)
        return poster


class DisgustMemeRenderer(BaseMemeRenderer):
    """Disgusted reaction meme with toxic slime green/purple accents and rejection stamps."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (18, 24, 18))
        draw = ImageDraw.Draw(poster)

        # Top Slime Banner
        draw.rectangle([50, 45, W - 50, 125], fill=(34, 197, 94))
        draw.text((70, 60), "BIOHAZARD // INSTANT RECOIL", font=FONTS["mono_bold"](28), fill=(0, 0, 0))

        f_huge = FONTS["impact"](92)
        draw.text((50, 140), "EW BROTHER EW... WHAT'S THAT?!", font=f_huge, fill=(240, 253, 244))

        # Face Display with Toxic Green Glow Frame
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 265

        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.6)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            f_proc = duotone_filter(f_proc, (18, 24, 18), (134, 239, 172))
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(30, 40, 30))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(168, 85, 247), width=6)

        # Big "CERTIFIED NASTY" Stamp
        draw.rounded_rectangle([fx + fw - 280, fy + 40, fx + fw + 20, fy + 120], radius=15, fill=(168, 85, 247))
        draw.text((fx + fw - 260, fy + 55), "CERTIFIED NASTY", font=FONTS["impact"](38), fill=(255, 255, 255))

        # Bottom Punchline
        draw.rectangle([50, 1090, W - 50, 1270], fill=(34, 197, 94))
        draw.text((75, 1115), "ABSOLUTELY REJECTED BY ALL SENSES", font=FONTS["impact"](62), fill=(0, 0, 0))
        draw.text((75, 1195), f"DISGUST LEVEL: {self.confidence}% // CANNOT UNSEE THIS", font=FONTS["mono_bold"](26), fill=(168, 85, 247))

        poster = add_film_grain(poster, intensity=0.12)
        return poster


class NeutralMemeRenderer(BaseMemeRenderer):
    """Deadpan, unimpressed meme with Swiss minimal layout, loading spinner, and flat stare caption."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (244, 244, 245))
        draw = ImageDraw.Draw(poster)

        # Subtle Grid Lines
        for gx in range(50, W, 100):
            draw.line([gx, 50, gx, H - 50], fill=(228, 228, 231), width=1)

        # Header
        draw.text((60, 50), "ISO 404 / EMOTIONLESS TEST BENCH", font=FONTS["mono"](18), fill=(113, 113, 122))
        f_title = FONTS["sans_bold"](100)
        draw.text((60, 85), "COMPLETELY UNBOTHERED.", font=f_title, fill=(24, 24, 27))

        # Face Display: Crisp High-Contrast Black & White
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 230

        if self.face_img:
            f_proc = ImageOps.grayscale(self.face_img)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.3)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc.convert("RGB"), (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(220, 220, 220))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(24, 24, 27), width=3)

        # Loading Spinner Indicator
        draw.text((fx + 30, fy + 30), "[ SYSTEM STATUS: 0% GIVEN ]", font=FONTS["mono_bold"](20), fill=(255, 59, 0))

        # Bottom Caption
        draw.rectangle([60, 1060, W - 60, 1260], fill=(24, 24, 27))
        draw.text((85, 1090), "LACK OF REACTION DETECTED", font=FONTS["impact"](64), fill=(255, 255, 255))
        draw.text((85, 1180), f"NEUTRALITY SCORE: {self.confidence}% // NOT IMPRESSED IN THE SLIGHTEST", font=FONTS["mono"](24), fill=(255, 59, 0))

        return poster


class ConfusedMemeRenderer(BaseMemeRenderer):
    """Confused meme with floating ??? question marks, math formulas, and chaotic layout."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (26, 22, 43))
        draw = ImageDraw.Draw(poster)

        # Floating Math & Logic Formulas
        formulas = ["E = mc² ???", "x = (-b ± √(b² - 4ac)) / 2a", "404 LOGIC NOT FOUND", "WHERE IS THE FLAVOR?", "∫ f(x)dx = WHY"]
        for i, form in enumerate(formulas):
            draw.text((60 + (i * 180) % 800, 70 + i * 40), form, font=FONTS["mono"](20), fill=(80, 70, 120))

        f_huge = FONTS["impact"](110)
        draw.text((50, 110), "WHAT IS BLUD EVEN DOING?!", font=f_huge, fill=(245, 158, 11))

        # Face Display with Question Marks & Offset Shadow
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 245
        draw.rectangle([fx + 16, fy + 16, fx + fw + 16, fy + fh + 16], fill=(139, 92, 246))

        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.4)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(40, 35, 60))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(245, 158, 11), width=5)

        # Huge Floating Question Marks
        f_q = FONTS["impact"](140)
        draw_text_with_outline(draw, (fx - 40, fy + 20), "?", f_q, (245, 158, 11), (0, 0, 0), outline_width=5)
        draw_text_with_outline(draw, (fx + fw - 80, fy + 80), "???", f_q, (139, 92, 246), (255, 255, 255), outline_width=4)

        # Bottom Punchline
        draw.rectangle([50, 1080, W - 50, 1270], fill=(139, 92, 246))
        draw.text((75, 1105), "NO THOUGHTS. HEAD COMPLETELY EMPTY.", font=FONTS["impact"](58), fill=(255, 255, 255))
        draw.text((75, 1185), f"CONFUSION COEFFICIENT: {self.confidence}% // BUFFER OVERLOAD", font=FONTS["mono_bold"](26), fill=(245, 158, 11))

        return poster


class ExcitedMemeRenderer(BaseMemeRenderer):
    """Chaotic, energetic meme with explosive neon confetti, lightning, and maximalist typography."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (30, 27, 75))
        draw = ImageDraw.Draw(poster)

        # Dynamic Confetti & Streaks
        random.seed(42)
        for _ in range(60):
            rx, ry = random.randint(20, W - 20), random.randint(20, H - 20)
            rw, rh = random.randint(8, 24), random.randint(8, 24)
            color = random.choice([(244, 63, 94), (251, 191, 36), (6, 182, 212), (168, 85, 247)])
            draw.rectangle([rx, ry, rx + rw, ry + rh], fill=color)

        f_huge = FONTS["impact"](140)
        draw_text_with_outline(draw, (50, 70), "LETS GOOOOOOOOO!", f_huge, (251, 191, 36), (244, 63, 94), outline_width=6)

        # Face Display with Hyper-Saturated Pop Styling
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 230

        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.8)
            f_proc = ImageEnhance.Contrast(f_proc).enhance(1.25)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(40, 30, 80))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(6, 182, 212), width=6)

        # Starburst Stickers
        draw_comic_starburst(draw, fx + fw - 20, fy + 80, num_points=12, r_inner=45, r_outer=95, fill=(244, 63, 94), outline=(255, 255, 255))
        draw.text((fx + fw - 65, fy + 55), "HYPE", font=FONTS["impact"](42), fill=(255, 255, 255))

        # Bottom Punchline
        draw.rectangle([50, 1070, W - 50, 1270], fill=(244, 63, 94))
        draw.text((75, 1095), "MAXIMUM OVERDRIVE ACHIEVED", font=FONTS["impact"](72), fill=(255, 255, 255))
        draw.text((75, 1185), f"ENERGY LEVEL: {self.confidence}% // CANNOT BE CONTAINED", font=FONTS["mono_bold"](28), fill=(251, 191, 36))

        return poster


class EmbarrassedMemeRenderer(BaseMemeRenderer):
    """Awkward cringe meme with anime blush cheeks, sweat drops, and cringe meter."""
    def render(self):
        W, H = POSTER_WIDTH, POSTER_HEIGHT
        poster = Image.new("RGB", (W, H), (42, 24, 32))
        draw = ImageDraw.Draw(poster)

        # Header
        draw.rounded_rectangle([50, 45, 420, 100], radius=15, fill=(251, 113, 133))
        draw.text((75, 60), "AWKWARD MOMENT // CAUGHT IN 4K", font=FONTS["impact"](26), fill=(0, 0, 0))

        f_huge = FONTS["impact"](68)
        draw.text((50, 125), "DYING OF SECONDHAND EMBARRASSMENT", font=f_huge, fill=(255, 241, 242))

        # Face Display with Soft Blush Framing
        fw, fh = 800, 780
        fx, fy = (W - fw) // 2, 235

        if self.face_img:
            f_proc = ImageEnhance.Color(self.face_img).enhance(1.4)
            f_proc = ImageOps.fit(f_proc, (fw, fh), method=Image.Resampling.LANCZOS)
            # Soft blush pink tint overlay
            arr = np.array(f_proc).astype(np.float32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.15, 0, 255)
            f_proc = Image.fromarray(arr.astype(np.uint8))
            poster.paste(f_proc, (fx, fy))
        else:
            draw.rectangle([fx, fy, fx + fw, fy + fh], fill=(50, 30, 40))

        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(251, 113, 133), width=5)

        # Cringe Badge
        draw.rounded_rectangle([fx + 30, fy + 30, fx + 260, fy + 95], radius=12, fill=(244, 114, 182))
        draw.text((fx + 50, fy + 45), "CRINGE: 9000+", font=FONTS["impact"](36), fill=(0, 0, 0))

        # Bottom Punchline
        draw.rectangle([50, 1070, W - 50, 1270], fill=(251, 113, 133))
        draw.text((75, 1095), "I WANT TO DISAPPEAR INTO THE FLOOR", font=FONTS["impact"](62), fill=(0, 0, 0))
        draw.text((75, 1180), f"SECONDHAND RATING: {self.confidence}% // UNRECOVERABLE DAMAGE", font=FONTS["mono_bold"](26), fill=(255, 255, 255))

        return poster


# Registry of meme renderers
RENDERERS = {
    "HAPPY": HappyMemeRenderer,
    "SAD": SadMemeRenderer,
    "ANGRY": AngryMemeRenderer,
    "SURPRISED": SurprisedMemeRenderer,
    "FEAR": FearMemeRenderer,
    "DISGUST": DisgustMemeRenderer,
    "NEUTRAL": NeutralMemeRenderer,
    "CONFUSED": ConfusedMemeRenderer,
    "EXCITED": ExcitedMemeRenderer,
    "EMBARRASSED": EmbarrassedMemeRenderer,
}


# --- FINAL BRAND COMPOSITING STEP ---

def composite_club_branding(poster):
    """
    Composites:
      - Bottom-Left: Instagram QR ("insta logo.jpeg") in a clean white rounded card.
      - Bottom-Right: Graphica club logo ("g logo.jpeg") + "GRAPHICA" text.
    Preserves exact proportions, works cleanly on all background palettes.
    """
    W, H = poster.size
    draw = ImageDraw.Draw(poster)

    badge_h = 170
    margin_x = 55
    margin_y = 55
    by = H - margin_y - badge_h

    # ---------------------------------------------------------
    # 1. BOTTOM-LEFT: INSTAGRAM QR CODE ("insta logo.jpeg")
    # ---------------------------------------------------------
    insta_paths = [
        os.path.join(HERE, "insta logo.jpeg"),
        os.path.join(HERE, "assets", "club_instagram_qr.jpeg"),
        os.path.join(HERE, "assets", "club_instagram_qr.png"),
    ]
    insta_file = next((p for p in insta_paths if os.path.isfile(p)), None)

    if insta_file:
        try:
            insta_img = Image.open(insta_file).convert("RGB")
            iw, ih = insta_img.size
            aspect = iw / ih

            target_insta_h = 145
            target_insta_w = int(target_insta_h * aspect)
            insta_scaled = insta_img.resize((target_insta_w, target_insta_h), Image.Resampling.LANCZOS)

            card_w = target_insta_w + 30
            card_x = margin_x

            # White card with drop shadow
            draw.rectangle([card_x + 5, by + 5, card_x + card_w + 5, by + badge_h + 5],
                           fill=(0, 0, 0, 140))
            draw.rounded_rectangle([card_x, by, card_x + card_w, by + badge_h],
                                   radius=14, fill=(255, 255, 255), outline=(0, 0, 0), width=3)

            # Center QR inside the card
            qr_x = card_x + (card_w - target_insta_w) // 2
            qr_y = by + (badge_h - target_insta_h) // 2
            poster.paste(insta_scaled, (qr_x, qr_y))
        except Exception as e:
            print(f"[BRANDING] Error loading Instagram QR: {e}")

    # ---------------------------------------------------------
    # 2. BOTTOM-RIGHT: CLUB LOGO + "GRAPHICA" ("g logo.jpeg")
    # ---------------------------------------------------------
    logo_paths = [
        os.path.join(HERE, "assets", "club_logo.jpeg"),
        os.path.join(HERE, "g logo.jpeg"),
        os.path.join(HERE, "assets", "club_logo.png"),
    ]
    logo_file = next((p for p in logo_paths if os.path.isfile(p)), None)

    if logo_file:
        try:
            logo_img = Image.open(logo_file).convert("RGBA")
            lw, lh = logo_img.size

            # Scale logo proportionally to fit inside badge
            target_logo_h = 110
            aspect = lw / lh
            target_logo_w = int(target_logo_h * aspect)
            logo_scaled = logo_img.resize((target_logo_w, target_logo_h), Image.Resampling.LANCZOS)

            # Measure "GRAPHICA" text width to size badge correctly
            f_name = FONTS["impact"](42)
            try:
                bbox = f_name.getbbox("GRAPHICA")
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = 160

            padding = 18
            gap = 14
            badge_w = padding + target_logo_w + gap + text_w + padding

            # Position at bottom-right
            badge_x = W - margin_x - badge_w

            # White card with drop shadow
            draw.rectangle([badge_x + 5, by + 5, badge_x + badge_w + 5, by + badge_h + 5],
                           fill=(0, 0, 0, 140))
            draw.rounded_rectangle([badge_x, by, badge_x + badge_w, by + badge_h],
                                   radius=14, fill=(255, 255, 255), outline=(0, 0, 0), width=3)

            # Paste logo on the left side of the badge
            logo_x = badge_x + padding
            logo_y = by + (badge_h - target_logo_h) // 2
            if logo_scaled.mode == "RGBA":
                poster.paste(logo_scaled, (logo_x, logo_y), mask=logo_scaled)
            else:
                poster.paste(logo_scaled, (logo_x, logo_y))

            # "GRAPHICA" text on the right side of the logo
            text_x = logo_x + target_logo_w + gap
            text_y = by + (badge_h - 42) // 2
            draw.text((text_x, text_y), "GRAPHICA", font=f_name, fill=(0, 0, 0))

        except Exception as e:
            print(f"[BRANDING] Error loading logo: {e}")
            badge_w = 240
            badge_x = W - margin_x - badge_w
            draw.rounded_rectangle([badge_x, by, badge_x + badge_w, by + badge_h],
                                   radius=14, fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            draw.text((badge_x + 24, by + 62), "GRAPHICA", font=FONTS["impact"](42), fill=(0, 0, 0))
    else:
        badge_w = 240
        badge_x = W - margin_x - badge_w
        draw.rounded_rectangle([badge_x, by, badge_x + badge_w, by + badge_h],
                               radius=14, fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        draw.text((badge_x + 24, by + 62), "GRAPHICA", font=FONTS["impact"](42), fill=(0, 0, 0))

    return poster


def generate_meme(frame, emotion_data, target_emotion=None, box=None, params=None):
    """
    Main pipeline to generate a graphic design meme.
    
    Args:
      frame: OpenCV BGR image from webcam.
      emotion_data: Dict with detected emotion details.
      target_emotion: Overridden emotion if user manually selected one.
      box: (x0, y0, x1, y1) face bounding box.
      params: Dict of options.
      
    Returns:
      (PIL Image, metadata dict)
    """
    emotion_key = (target_emotion or emotion_data.get("emotion", "HAPPY")).upper()
    if emotion_key not in RENDERERS:
        emotion_key = "HAPPY"

    face_img = None
    if frame is not None:
        try:
            face_img = crop_face_portrait(frame, box)
        except Exception as e:
            print(f"Face crop error: {e}")

    # Initialize renderer
    renderer_cls = RENDERERS[emotion_key]
    renderer = renderer_cls(face_img, emotion_data, params)
    
    # 1. Render Meme Composition
    poster = renderer.render()
    
    # 2. Composite Final Club Branding (Logo + Instagram QR)
    poster = composite_club_branding(poster)

    meta = {
        "emotion": emotion_key,
        "label": renderer.label,
        "caption": renderer.caption,
        "confidence": renderer.confidence,
        "design_id": renderer.design_id,
        "timestamp": renderer.timestamp,
        "is_override": target_emotion is not None and target_emotion.upper() != emotion_data.get("emotion", "").upper(),
    }

    return poster, meta
