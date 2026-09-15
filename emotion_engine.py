#!/usr/bin/env python3
"""
emotion_engine.py — Reaction and emotion classification for REACTIONARY.

Continuously analyzes facial blendshapes, head pose, and resting calibration
to classify 10 distinct emotional states in real time:
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
"""

import json
import os
import numpy as np

# 10 Primary Emotions supported by REACTIONARY Meme Booth
EMOTIONS = {
    "HAPPY": {
        "label": "Happy",
        "symbol": "😄",
        "tagline": "Joyful / Goofy / Wholesome",
        "description": "Warm smile, relaxed forehead, pure serotonin",
        "palette": ["#FFD700", "#FF4081", "#00E5FF", "#18181B"],
        "default_caption": "POV: YOUR CODE COMPILES ON THE FIRST TRY",
        "accent": "#FFD700",
    },
    "SAD": {
        "label": "Sad",
        "symbol": "😢",
        "tagline": "Dramatic / Depressed / Melancholic",
        "description": "Inner brows raised, downturned mouth, existential crisis",
        "palette": ["#1E293B", "#38BDF8", "#64748B", "#0F172A"],
        "default_caption": "IT'S FINE. EVERYTHING IS FINE.",
        "accent": "#38BDF8",
    },
    "ANGRY": {
        "label": "Angry",
        "symbol": "😡",
        "tagline": "Rage / Aggressive / Chaotic",
        "description": "Furrowed brow, flared nose, clenched jaw",
        "palette": ["#0F0F10", "#FF1744", "#CCFF00", "#FFFFFF"],
        "default_caption": "PEACE WAS NEVER AN OPTION",
        "accent": "#FF1744",
    },
    "SURPRISED": {
        "label": "Surprised",
        "symbol": "😲",
        "tagline": "Shocked / Unexpected / Stunned",
        "description": "Wide open jaw, raised brows, startled gaze",
        "palette": ["#070B19", "#00F0FF", "#FF007A", "#FFFFFF"],
        "default_caption": "WAIT... WHAT JUST HAPPENED?!",
        "accent": "#00F0FF",
    },
    "FEAR": {
        "label": "Fear",
        "symbol": "😨",
        "tagline": "Scared / Panic / Danger",
        "description": "Wide eyes, lateral mouth stretch, pure dread",
        "palette": ["#0A0E17", "#10B981", "#EF4444", "#E2E8F0"],
        "default_caption": "PANIK MODE ACTIVATED",
        "accent": "#10B981",
    },
    "DISGUST": {
        "label": "Disgust",
        "symbol": "🤢",
        "tagline": "Disgusted / Reaction / Ick",
        "description": "Wrinkled nose, curled upper lip, instant recoil",
        "palette": ["#121811", "#22C55E", "#A855F7", "#F8FAFC"],
        "default_caption": "EW BROTHER EW... WHAT'S THAT?!",
        "accent": "#22C55E",
    },
    "NEUTRAL": {
        "label": "Neutral",
        "symbol": "😐",
        "tagline": "Deadpan / Unimpressed / Void",
        "description": "Flat stare, zero expression, completely unbothered",
        "palette": ["#18181B", "#71717A", "#E4E4E7", "#FF3B00"],
        "default_caption": "LACK OF REACTION DETECTED",
        "accent": "#71717A",
    },
    "CONFUSED": {
        "label": "Confused",
        "symbol": "🤔",
        "tagline": "What Is Happening / Processing",
        "description": "Asymmetric eyebrow, head tilt, math equation vibes",
        "palette": ["#1A162B", "#8B5CF6", "#F59E0B", "#F1F5F9"],
        "default_caption": "WHAT IS BLUD EVEN DOING?!",
        "accent": "#8B5CF6",
    },
    "EXCITED": {
        "label": "Excited",
        "symbol": "🤩",
        "tagline": "Chaotic / Energetic / Hype",
        "description": "Huge grin, wide eyes, maximum overdrive",
        "palette": ["#1E1B4B", "#F43F5E", "#FBBF24", "#06B6D4"],
        "default_caption": "LETS GOOOOOOOOO!",
        "accent": "#F43F5E",
    },
    "EMBARRASSED": {
        "label": "Embarrassed",
        "symbol": "😳",
        "tagline": "Awkward / Cringe / Blush",
        "description": "Nervous tight grin, looking away, dying inside",
        "palette": ["#2A1820", "#FB7185", "#F472B6", "#FFF1F2"],
        "default_caption": "DYING OF SECONDHAND EMBARRASSMENT",
        "accent": "#FB7185",
    },
}

ALL_EMOTIONS_LIST = [
    {"id": k, "label": v["label"], "symbol": v["symbol"], "tagline": v["tagline"], "accent": v["accent"]}
    for k, v in EMOTIONS.items()
]

GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05, "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05,
    "browInnerUp": 0.06, "browOuterUpLeft": 0.10, "browOuterUpRight": 0.10,
    "eyeWideLeft": 0.02, "eyeWideRight": 0.02,
    "mouthStretchLeft": 0.02, "mouthStretchRight": 0.02, "mouthPressLeft": 0.02, "mouthPressRight": 0.02,
    "eyeLookDownLeft": 0.05, "eyeLookDownRight": 0.05,
}
GENERIC_SIGMA = 0.035


class Baseline:
    """Resting face baseline calibration wrapper."""
    def __init__(self, mean=None, sigma=None, samples=0, made=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.generic = not self.mean

    def z(self, name, value):
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        s = max(self.sigma.get(name, 0.08), 0.015)
        return (value - m) / s

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not data.get("mean"):
                return cls()
            return cls(data["mean"], data.get("sigma", {}), data.get("samples", 0), data.get("made"))
        except Exception:
            return cls()


class EmotionEngine:
    """Calculates real-time classification across all 10 emotional states."""
    def __init__(self, calib_path=None):
        self.calib_path = calib_path or os.path.join(os.path.dirname(__file__), "calibration.json")
        self.baseline = Baseline.load(self.calib_path)
        self.smoothed_scores = {e: 0.0 for e in EMOTIONS}

    def reload_baseline(self):
        self.baseline = Baseline.load(self.calib_path)

    def analyze(self, blendshapes, turn_signed=0.0):
        """
        Takes blendshapes {category: score} and computes dominant emotion from 10 classes.
        """
        b = blendshapes.get
        base = self.baseline

        def z_ch(name):
            return base.z(name, b(name, 0.0))

        def z_pair(name):
            vl = base.z(name + "Left", b(name + "Left", 0.0))
            vr = base.z(name + "Right", b(name + "Right", 0.0))
            return (vl + vr) / 2.0

        def raw_pair(name):
            return (b(name + "Left", 0.0) + b(name + "Right", 0.0)) / 2.0

        # Channels
        jaw_open = b("jawOpen", 0.0)
        z_jaw = z_ch("jawOpen")

        sneer = raw_pair("noseSneer")
        z_sneer = z_pair("noseSneer")

        brow_down = raw_pair("browDown")
        z_brow_down = z_pair("browDown")

        brow_inner_up = b("browInnerUp", 0.0)
        z_brow_inner_up = z_ch("browInnerUp")

        brow_outer_up_l = b("browOuterUpLeft", 0.0)
        brow_outer_up_r = b("browOuterUpRight", 0.0)
        brow_outer_up = (brow_outer_up_l + brow_outer_up_r) / 2.0

        frown = raw_pair("mouthFrown")
        z_frown = z_pair("mouthFrown")

        smile = raw_pair("mouthSmile")
        z_smile = z_pair("mouthSmile")

        eye_wide = raw_pair("eyeWide")
        z_eye_wide = z_pair("eyeWide")

        squint = max(raw_pair("eyeSquint"), raw_pair("eyeBlink"))
        z_squint = max(z_pair("eyeSquint"), z_pair("eyeBlink"))

        stretch = raw_pair("mouthStretch")
        z_stretch = z_pair("mouthStretch")

        press = raw_pair("mouthPress")
        look_down = raw_pair("eyeLookDown")

        turn = abs(turn_signed - base.neutral_turn)

        # Compute raw scores for each of the 10 emotions
        raw = {}

        # 1. HAPPY: High smile, cheek squint, low brow down
        happy_score = z_smile * 2.2 + z_squint * 0.4 - z_frown * 0.6
        if smile > 0.15:
            happy_score += 1.5
        raw["HAPPY"] = max(0.0, happy_score)

        # 2. SAD: Raised inner brow, mouth frown, depressed corners
        sad_score = z_brow_inner_up * 1.8 + z_frown * 1.5 - z_smile * 1.2
        if brow_inner_up > 0.15 and smile < 0.1:
            sad_score += 1.2
        raw["SAD"] = max(0.0, sad_score)

        # 3. ANGRY: Brow down, nose sneer, mouth press
        angry_score = z_brow_down * 1.8 + z_sneer * 1.4 + press * 3.0 - z_smile * 0.8
        if brow_down > 0.15:
            angry_score += 1.2
        raw["ANGRY"] = max(0.0, angry_score)

        # 4. SURPRISED: Wide open jaw, wide eyes, raised outer brows
        surprised_score = z_jaw * 1.8 + z_eye_wide * 1.6 + brow_outer_up * 4.0
        if jaw_open > 0.35:
            surprised_score += 2.0
        raw["SURPRISED"] = max(0.0, surprised_score)

        # 5. FEAR: Wide eyes, inner brow up, lateral mouth stretch
        fear_score = z_eye_wide * 1.5 + z_brow_inner_up * 1.4 + z_stretch * 1.6
        if eye_wide > 0.2 and stretch > 0.1:
            fear_score += 1.5
        raw["FEAR"] = max(0.0, fear_score)

        # 6. DISGUST: Intense nose sneer, upper lip raised, squint
        disgust_score = z_sneer * 2.5 + z_ch("mouthUpperUpLeft") * 1.5 + z_squint * 0.8
        if sneer > 0.08:
            disgust_score += 2.0
        raw["DISGUST"] = max(0.0, disgust_score)

        # 7. CONFUSED: Asymmetric brow (one down, one up) + head turn/tilt + squint
        asym_brow = abs(b("browDownLeft", 0.0) - b("browDownRight", 0.0)) + abs(brow_outer_up_l - brow_outer_up_r)
        confused_score = asym_brow * 6.0 + turn * 4.5 + z_squint * 0.6
        if asym_brow > 0.08:
            confused_score += 1.8
        raw["CONFUSED"] = max(0.0, confused_score)

        # 8. EXCITED: Combination of high smile + open jaw + wide eyes
        excited_score = z_smile * 1.5 + z_jaw * 1.3 + z_eye_wide * 1.2
        if smile > 0.3 and jaw_open > 0.2:
            excited_score += 2.5
        raw["EXCITED"] = max(0.0, excited_score)

        # 9. EMBARRASSED: Tight press smile + looking down or away + slight blush tension
        embarrassed_score = press * 4.0 + z_smile * 0.8 + look_down * 3.5 - z_jaw * 1.0
        if press > 0.15 and smile > 0.05:
            embarrassed_score += 1.6
        raw["EMBARRASSED"] = max(0.0, embarrassed_score)

        # 10. NEUTRAL: Dominates when other expressions have low excitement
        total_energy = sum(raw.values())
        raw["NEUTRAL"] = max(0.0, 3.2 - total_energy * 0.45)

        # Temporal smoothing (alpha = 0.35)
        for e in EMOTIONS:
            self.smoothed_scores[e] = 0.65 * self.smoothed_scores.get(e, 0.0) + 0.35 * raw[e]

        # Winner
        best_emotion = max(self.smoothed_scores, key=self.smoothed_scores.get)
        best_val = self.smoothed_scores[best_emotion]

        total_s = sum(self.smoothed_scores.values()) + 1e-6
        confidence = min(98, max(58, int((best_val / total_s) * 110.0 + 20)))

        info = EMOTIONS[best_emotion]

        telemetry = {
            "jaw": round(float(jaw_open), 3),
            "z_jaw": round(float(z_jaw), 1),
            "brow_down": round(float(brow_down), 3),
            "z_brow_down": round(float(z_brow_down), 1),
            "brow_inner_up": round(float(brow_inner_up), 3),
            "z_brow_inner_up": round(float(z_brow_inner_up), 1),
            "smile": round(float(smile), 3),
            "z_smile": round(float(z_smile), 1),
            "frown": round(float(frown), 3),
            "squint": round(float(squint), 3),
            "turn": round(float(turn), 3),
        }

        return {
            "emotion": best_emotion,
            "label": info["label"],
            "symbol": info["symbol"],
            "tagline": info["tagline"],
            "description": info["description"],
            "palette": info["palette"],
            "default_caption": info["default_caption"],
            "accent": info["accent"],
            "confidence": confidence,
            "telemetry": telemetry,
            "is_calibrated": not self.baseline.generic,
        }
