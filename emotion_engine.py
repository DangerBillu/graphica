#!/usr/bin/env python3
"""
emotion_engine.py — Reaction and emotion classification for REACTIONARY.

Translates facial blendshapes, head pose, and resting calibration into
primary emotional states, confidence scores, and graphic design style mappings.
"""

import json
import os
import numpy as np

# Primary emotions supported by REACTIONARY
EMOTIONS = {
    "NEUTRAL": {
        "label": "Unimpressed",
        "symbol": "😐",
        "default_style": "Swiss",
        "description": "Restrained, balanced, minimal expression",
        "palette": ["#000000", "#FFFFFF", "#FF3B00", "#8E8E93"],
        "typography": "Helvetica / Grotesque",
    },
    "OVERWHELMED": {
        "label": "Overwhelmed",
        "symbol": "😭",
        "default_style": "Experimental",
        "description": "Dramatic, inward-turning, intense depth",
        "palette": ["#0D0D11", "#1F2232", "#E0E2EC", "#5A67D8"],
        "typography": "Serif Display + Distorted Sans",
    },
    "ANGRY": {
        "label": "Aggressive",
        "symbol": "😡",
        "default_style": "Brutalist",
        "description": "High tension, sharp geometric friction",
        "palette": ["#0A0A0A", "#FFFFFF", "#CCFF00", "#FF1A1A"],
        "typography": "Ultra Condensed Display",
    },
    "SHOCKED": {
        "label": "Shocked",
        "symbol": "😳",
        "default_style": "Y2K",
        "description": "Chaotic surge, wide-open perception",
        "palette": ["#070B19", "#00F0FF", "#FF007A", "#E0E7FF"],
        "typography": "Futuristic Extended Sans",
    },
    "AMUSED": {
        "label": "Amused",
        "symbol": "😂",
        "default_style": "Maximalist",
        "description": "Playful, celebratory, vibrant warmth",
        "palette": ["#FFD600", "#FF2A85", "#00E5FF", "#18181B"],
        "typography": "Chunky Rounded Gothic",
    },
    "SUSPICIOUS": {
        "label": "Suspicious",
        "symbol": "🤨",
        "default_style": "Retro",
        "description": "Asymmetrical scrutiny, investigative intrigue",
        "palette": ["#121316", "#C9182B", "#D1D5DB", "#854D0E"],
        "typography": "Noir Monospace & Stencil",
    },
    "CONFIDENT": {
        "label": "Confident",
        "symbol": "😎",
        "default_style": "Editorial",
        "description": "Composed authority, refined poise",
        "palette": ["#111111", "#D4AF37", "#F3F4F6", "#374151"],
        "typography": "High-Contrast Luxury Serif",
    },
}

ALL_STYLES = [
    {"id": "Swiss", "name": "Swiss / International", "era": "1950s Modernism", "accent": "#FF3B00"},
    {"id": "Brutalist", "name": "Brutalist / Aggressive", "era": "Raw Contemporary", "accent": "#CCFF00"},
    {"id": "Y2K", "name": "Y2K / Cyber Chrome", "era": "Late 90s Cyber", "accent": "#00F0FF"},
    {"id": "Editorial", "name": "Luxury Editorial", "era": "Haute Couture", "accent": "#D4AF37"},
    {"id": "Maximalist", "name": "Pop / Maximalist", "era": "Playful Chaos", "accent": "#FF2A85"},
    {"id": "Minimalist", "name": "Hyper Minimalist", "era": "Negative Space", "accent": "#FFFFFF"},
    {"id": "Cyberpunk", "name": "Cyberpunk Telemetry", "era": "Neo-Tokyo 2077", "accent": "#00FF66"},
    {"id": "Retro", "name": "Retro Risograph", "era": "Acid Print 70s", "accent": "#FF6B4A"},
    {"id": "Desi Maximalism", "name": "Desi Maximalism", "era": "Folk Truck Art", "accent": "#FF9933"},
    {"id": "Experimental", "name": "Dramatic Experimental", "era": "Avant-Garde", "accent": "#7928CA"},
]

GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05, "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05,
    "browInnerUp": 0.06, "eyeWideLeft": 0.02, "eyeWideRight": 0.02,
}
GENERIC_SIGMA = 0.035


class Baseline:
    """Resting face calibration wrapper."""
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
    """Evaluates facial blendshapes against personal baseline to determine emotion and style."""
    def __init__(self, calib_path=None):
        self.calib_path = calib_path or os.path.join(os.path.dirname(__file__), "calibration.json")
        self.baseline = Baseline.load(self.calib_path)
        self.smoothed_scores = {e: 0.0 for e in EMOTIONS}
        self.history = []

    def reload_baseline(self):
        self.baseline = Baseline.load(self.calib_path)

    def analyze(self, blendshapes, turn_signed=0.0):
        """
        Analyze blendshape dictionary {category_name: score}.
        Returns dict with:
          - emotion: primary emotion key (e.g. 'ANGRY')
          - label: human readable label ('Aggressive')
          - symbol: emoji
          - style: recommended design style name ('Brutalist')
          - confidence: score between 0 and 100
          - scores: dict of all emotion intensities
          - telemetry: key blendshape channels for UI visualization
        """
        b = blendshapes.get
        base = self.baseline

        # Helper to compute Z-scores across left & right channels
        def z_ch(name):
            return base.z(name, b(name, 0.0))

        def z_pair(name):
            vl = base.z(name + "Left", b(name + "Left", 0.0))
            vr = base.z(name + "Right", b(name + "Right", 0.0))
            return (vl + vr) / 2.0

        def raw_pair(name):
            return (b(name + "Left", 0.0) + b(name + "Right", 0.0)) / 2.0

        jaw_open = b("jawOpen", 0.0)
        z_jaw = z_ch("jawOpen")

        sneer = raw_pair("noseSneer")
        z_sneer = z_pair("noseSneer")

        brow_down = raw_pair("browDown")
        z_brow_down = z_pair("browDown")

        brow_inner_up = b("browInnerUp", 0.0)
        z_brow_inner_up = z_ch("browInnerUp")

        frown = raw_pair("mouthFrown")
        z_frown = z_pair("mouthFrown")

        smile = raw_pair("mouthSmile")
        z_smile = z_pair("mouthSmile")

        eye_wide = raw_pair("eyeWide")
        z_eye_wide = z_pair("eyeWide")

        squint = max(raw_pair("eyeSquint"), raw_pair("eyeBlink"))
        z_squint = max(z_pair("eyeSquint"), z_pair("eyeBlink"))

        turn = abs(turn_signed - base.neutral_turn)

        # Raw emotion intensities
        raw_scores = {}

        # 1. ANGER: furrowed brow, nose sneer, tight mouth / pressed lips
        raw_scores["ANGRY"] = max(0.0, (z_brow_down * 1.5 + z_sneer * 1.2 + z_frown * 0.8) / 3.0)

        # 2. OVERWHELMED / SADNESS: inner brows up, mouth frown, droop
        raw_scores["OVERWHELMED"] = max(0.0, (z_brow_inner_up * 1.6 + z_frown * 1.4 - z_smile * 0.8) / 2.5)

        # 3. SHOCKED: wide open jaw, raised brows, wide eyes
        raw_scores["SHOCKED"] = max(0.0, (z_jaw * 1.4 + z_eye_wide * 1.3 + z_brow_inner_up * 0.8) / 3.0)

        # 4. AMUSED / HAPPINESS: smile, cheek squint, dimples
        raw_scores["AMUSED"] = max(0.0, (z_smile * 1.8 + z_squint * 0.6) / 2.0)

        # 5. CONFIDENT: subtle smile, steady posture, low tension, slight chin elevation
        confidence_base = max(0.0, (z_smile * 0.7 - z_brow_down * 0.5 - z_frown * 0.8))
        if 0.05 <= smile <= 0.35 and brow_down < 0.2:
            confidence_base += 1.0
        raw_scores["CONFIDENT"] = max(0.0, confidence_base)

        # 6. SUSPICIOUS: head turn combined with squint / asymmetric brow
        suspicious_score = (turn * 6.0) + (z_squint * 0.8)
        if turn > 0.08 and squint > 0.15:
            suspicious_score += 1.5
        raw_scores["SUSPICIOUS"] = max(0.0, suspicious_score)

        # 7. NEUTRAL: inverse of expression excitement
        total_energy = sum(raw_scores.values())
        neutral_score = max(0.0, 3.5 - total_energy * 0.5)
        raw_scores["NEUTRAL"] = neutral_score

        # Temporal smoothing (alpha = 0.35)
        for e in EMOTIONS:
            self.smoothed_scores[e] = 0.65 * self.smoothed_scores.get(e, 0.0) + 0.35 * raw_scores[e]

        # Determine winner
        best_emotion = max(self.smoothed_scores, key=self.smoothed_scores.get)
        best_val = self.smoothed_scores[best_emotion]

        # Convert to confidence (normalized 0-100)
        total_s = sum(self.smoothed_scores.values()) + 1e-6
        raw_conf = (best_val / total_s) * 100.0
        # Scaled realistic confidence (between 65% and 98%)
        confidence = min(98, max(58, int(raw_conf * 1.4 + 20)))

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
            "z_frown": round(float(z_frown), 1),
            "squint": round(float(squint), 3),
            "z_squint": round(float(z_squint), 1),
            "turn": round(float(turn), 3),
        }

        normalized_scores = {
            e: round(float(self.smoothed_scores[e]), 2) for e in EMOTIONS
        }

        return {
            "emotion": best_emotion,
            "label": info["label"],
            "symbol": info["symbol"],
            "style": info["default_style"],
            "description": info["description"],
            "palette": info["palette"],
            "typography": info["typography"],
            "confidence": confidence,
            "scores": normalized_scores,
            "telemetry": telemetry,
            "is_calibrated": not self.baseline.generic,
        }

