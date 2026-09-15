#!/usr/bin/env python3
"""
reactionary.py — Core application server for the REACTIONARY AI Meme Booth.

1. Captures live webcam stream and runs MediaPipe FaceLandmarker in real time.
2. Performs 10-emotion classification (Happy, Sad, Angry, Surprised, Fear, Disgust,
   Neutral, Confused, Excited, Embarrassed).
3. Detects face presence and multiple face warnings:
   - "Please position your face inside the camera frame."
   - "Please make sure only one person is in the frame."
4. Allows user emotion override in UI before meme synthesis.
5. Generates high-resolution meme posters with embedded Graphica logo and Instagram QR.
6. Generates a separate, unique dynamic QR code for phone download (linking to /m/<token>).
7. Serves responsive mobile download page and cleans up expired memes.
"""

import argparse
import base64
import io
import json
import os
import socket
import sys
import threading
import time
import uuid
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

import cv2
import numpy as np
import qrcode
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import emotion_engine
import poster_generator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
WEB_DIR = os.path.join(HERE, "web")

# In-memory storage for meme tokens with expiration (2 hours)
MEME_STORE = {}
MEME_EXPIRY_SECONDS = 7200  # 2 hours


def get_local_ip():
    """Detects local LAN IP address for phone hotspot/Wi-Fi connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def generate_qr_data_url(text):
    """Generates a high-contrast QR code as a base64 PNG data URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=3,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class Clock:
    """Monotonic timestamp generator for MediaPipe video mode."""
    def __init__(self):
        self.t0 = time.monotonic()
        self.last = -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class CameraWorker(threading.Thread):
    """Continuously captures frames and tracks face presence and 10 emotions."""
    def __init__(self, camera_idx=0, size="640x480", no_flip=False):
        super().__init__(daemon=True)
        self.camera_idx = camera_idx
        self.size = size
        self.no_flip = no_flip
        self.running = True
        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_jpeg = None
        self.has_face = False
        self.num_faces = 0
        self.warning_msg = "Please position your face inside the camera frame."
        self.face_box = None
        self.frame_w = 640
        self.frame_h = 480
        self.fps = 0.0
        
        self.emotion_engine = emotion_engine.EmotionEngine()
        self.emotion_data = {
            "emotion": "NEUTRAL",
            "label": "Neutral",
            "symbol": "😐",
            "tagline": "Deadpan / Unimpressed",
            "description": "Flat stare, zero expression",
            "palette": ["#18181B", "#71717A", "#E4E4E7", "#FF3B00"],
            "default_caption": "LACK OF REACTION DETECTED",
            "accent": "#71717A",
            "confidence": 85,
            "telemetry": {},
            "is_calibrated": not self.emotion_engine.baseline.generic,
        }

    def run(self):
        print(f"[REACTIONARY] Opening camera index {self.camera_idx}...")
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam device {self.camera_idx}.")
            with self.lock:
                self.warning_msg = "Camera permission denied or camera unavailable."
            return

        if "x" in self.size:
            sw, sh = self.size.lower().split("x")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(sw))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(sh))

        model_path = os.path.join(HERE, "models", "face_landmarker.task")
        clock = Clock()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=3,
            output_face_blendshapes=True
        )
        face_detector = vision.FaceLandmarker.create_from_options(options)

        frame_counter = 0
        t_start = time.time()

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                if not self.no_flip:
                    frame = cv2.flip(frame, 1)

                H, W = frame.shape[:2]
                ts = clock.next()

                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = face_detector.detect_for_video(mp_img, ts)

                has_face = False
                num_faces = len(result.face_landmarks) if result.face_landmarks else 0
                face_box = None
                warning = None

                if num_faces == 0:
                    warning = "Please position your face inside the camera frame."
                    e_res = self.emotion_data
                elif num_faces > 1:
                    warning = f"Please make sure only one person is in the frame. ({num_faces} detected)"
                    has_face = True
                    # Track largest/primary face
                    lms = result.face_landmarks[0]
                    pts = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
                    x0, y0 = pts.min(0)
                    x1, y1 = pts.max(0)
                    face_box = (int(x0), int(y0), int(x1), int(y1))
                    
                    bs_dict = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
                    cl, cr, nose = pts[234], pts[454], pts[1]
                    turn_signed = float((nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
                    e_res = self.emotion_engine.analyze(bs_dict, turn_signed)
                else:
                    # Exactly 1 face
                    has_face = True
                    warning = None
                    lms = result.face_landmarks[0]
                    pts = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
                    x0, y0 = pts.min(0)
                    x1, y1 = pts.max(0)
                    face_box = (int(x0), int(y0), int(x1), int(y1))

                    bs_dict = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
                    cl, cr, nose = pts[234], pts[454], pts[1]
                    turn_signed = float((nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
                    e_res = self.emotion_engine.analyze(bs_dict, turn_signed)

                _, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                frame_counter += 1
                if time.time() - t_start >= 1.0:
                    self.fps = round(frame_counter / (time.time() - t_start), 1)
                    frame_counter = 0
                    t_start = time.time()

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_jpeg = jpeg_buf.tobytes()
                    self.has_face = has_face
                    self.num_faces = num_faces
                    self.warning_msg = warning
                    self.face_box = face_box
                    self.frame_w = W
                    self.frame_h = H
                    self.emotion_data = e_res

                time.sleep(0.008)
        finally:
            cap.release()
            face_detector.close()

    def get_state(self):
        with self.lock:
            return {
                "has_face": self.has_face,
                "num_faces": self.num_faces,
                "warning_msg": self.warning_msg,
                "face_box": self.face_box,
                "frame_w": self.frame_w,
                "frame_h": self.frame_h,
                "fps": self.fps,
                **self.emotion_data
            }

    def get_frame_and_emotion(self):
        with self.lock:
            frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
            return frame_copy, self.face_box, dict(self.emotion_data)


WORKER = None
SERVER_PORT = 8000
SERVER_IP = "127.0.0.1"


def cleanup_expired_memes():
    """Periodically removes generated memes older than 2 hours."""
    now = time.time()
    expired_tokens = []
    for token, item in list(MEME_STORE.items()):
        if now - item.get("created_at", now) > MEME_EXPIRY_SECONDS:
            expired_tokens.append(token)
            
    for t in expired_tokens:
        item = MEME_STORE.pop(t, None)
        if item and "filepath" in item and os.path.isfile(item["filepath"]):
            try:
                os.remove(item["filepath"])
            except Exception:
                pass


def build_mobile_page_html(token, meta):
    """Constructs a responsive, mobile-friendly download page for phone visitors."""
    caption = meta.get("caption", "GRAPHICA MEME")
    emotion = meta.get("emotion", "HAPPY")
    symbol = meta.get("symbol", "✨")
    download_url = f"/api/download/{token}"
    img_url = f"/api/meme/{token}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>Your Meme — Graphica AI Booth</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: #0A0B10;
      color: #F8FAFC;
      font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px 16px 40px;
    }}
    .header {{
      width: 100%;
      max-width: 480px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0 16px;
      border-bottom: 1px solid #1E293B;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 800;
      font-size: 17px;
      letter-spacing: 0.5px;
    }}
    .brand-badge {{
      background: #CCFF00;
      color: #000;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .club-tag {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #94A3B8;
    }}
    .hero-card {{
      width: 100%;
      max-width: 480px;
      margin-top: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .meme-wrap {{
      width: 100%;
      background: #000;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid #334155;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.7);
    }}
    .meme-img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .info-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #11131A;
      border: 1px solid #1E293B;
      border-radius: 10px;
      padding: 12px 16px;
    }}
    .emotion-tag {{
      font-size: 15px;
      font-weight: 700;
      color: #CCFF00;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .design-id {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #64748B;
    }}
    .action-stack {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 4px;
    }}
    .btn {{
      width: 100%;
      border: none;
      border-radius: 12px;
      padding: 16px 20px;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      text-decoration: none;
      transition: transform 0.1s ease, filter 0.2s ease;
    }}
    .btn-download {{
      background: linear-gradient(135deg, #CCFF00 0%, #00F0FF 100%);
      color: #000;
      box-shadow: 0 6px 20px rgba(0, 240, 255, 0.35);
    }}
    .btn-download:active {{ transform: scale(0.98); }}
    .btn-share {{
      background: #1E293B;
      color: #F8FAFC;
      border: 1px solid #334155;
    }}
    .footer-note {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: #64748B;
      text-align: center;
      margin-top: 24px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <header class="header">
    <div class="brand">
      <span class="brand-badge">RX</span>
      <span>REACTIONARY</span>
    </div>
    <div class="club-tag">@GRAPHICA.CLUB</div>
  </header>

  <main class="hero-card">
    <div class="meme-wrap">
      <img src="{img_url}" alt="{caption}" class="meme-img" id="memeImage">
    </div>

    <div class="info-bar">
      <div class="emotion-tag">{symbol} {emotion} MEME</div>
      <div class="design-id">{meta.get('design_id', '#RX-042')}</div>
    </div>

    <div class="action-stack">
      <a href="{download_url}" download class="btn btn-download">
        <span>📥</span> DOWNLOAD MEME IMAGE
      </a>
      <button class="btn btn-share" id="shareBtn">
        <span>🔗</span> SHARE MEME
      </button>
    </div>

    <p class="footer-note">
      Crafted live at Graphica Creative-Tech Booth.<br>
      Tag <strong>@graphica.club</strong> on Instagram!
    </p>
  </main>

  <script>
    const shareBtn = document.getElementById('shareBtn');
    const currentUrl = window.location.href;

    function copyFallback(text) {{
      /* Works over plain HTTP where clipboard API is unavailable */
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;pointer-events:none;';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {{
        document.execCommand('copy');
        showToast('Link copied! Paste it to share.');
      }} catch (e) {{
        prompt('Copy this link to share:', text);
      }}
      document.body.removeChild(ta);
    }}

    function showToast(msg) {{
      const t = document.createElement('div');
      t.textContent = msg;
      t.style.cssText = [
        'position:fixed', 'bottom:24px', 'left:50%', 'transform:translateX(-50%)',
        'background:#CCFF00', 'color:#000', 'font-weight:700',
        'padding:12px 24px', 'border-radius:30px',
        'font-size:14px', 'z-index:9999', 'pointer-events:none',
        'transition:opacity 0.3s ease'
      ].join(';');
      document.body.appendChild(t);
      setTimeout(() => {{ t.style.opacity = '0'; setTimeout(() => document.body.removeChild(t), 300); }}, 2800);
    }}

    shareBtn.addEventListener('click', async () => {{
      /* Try native share first (iOS Safari, Android Chrome — requires HTTPS or localhost) */
      if (navigator.share && (location.protocol === 'https:' || location.hostname === 'localhost')) {{
        try {{
          await navigator.share({{
            title: 'My Reactionary Meme — Graphica Club',
            text: '{caption}',
            url: currentUrl,
          }});
          return;
        }} catch (e) {{
          if (e.name === 'AbortError') return; /* User cancelled — do nothing */
        }}
      }}
      /* Fallback: copy URL to clipboard (works over plain HTTP) */
      copyFallback(currentUrl);
    }});
  </script>
</body>
</html>
"""


class ReactionaryHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving Web UI, API, video stream, and mobile download page."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/video_feed":
            self.stream_video()
            return

        elif path == "/api/status":
            self.send_json(WORKER.get_state())
            return

        elif path == "/api/emotions":
            self.send_json({"emotions": emotion_engine.ALL_EMOTIONS_LIST})
            return

        elif path.startswith("/m/"):
            # Mobile download landing page: /m/<token>
            token = path[3:].strip("/")
            item = MEME_STORE.get(token)
            if item and os.path.isfile(item["filepath"]):
                html = build_mobile_page_html(token, item["metadata"])
                self.send_html_response(html)
            else:
                self.send_html_response("<h2>Meme link has expired or does not exist.</h2><p>Visit the booth to generate a new one!</p>", status=404)
            return

        elif path.startswith("/api/meme/"):
            # Direct image serving
            token = path[10:].strip("/")
            item = MEME_STORE.get(token)
            if item and os.path.isfile(item["filepath"]):
                self.send_file_response(item["filepath"], content_type="image/png")
            else:
                self.send_error(404, "Meme image not found or expired")
            return

        elif path.startswith("/api/download/"):
            # File download attachment
            token = path[14:].strip("/")
            item = MEME_STORE.get(token)
            if item and os.path.isfile(item["filepath"]):
                filename = f"GRAPHICA_MEME_{item['metadata'].get('emotion', 'RX')}_{item['metadata'].get('design_id', '042').replace('#', '')}.png"
                self.send_file_response(item["filepath"], content_type="image/png", as_attachment=True, filename=filename)
            else:
                self.send_error(404, "File not found or expired")
            return

        # Default static file serving from web/
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/generate":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode('utf-8')) if body else {}
            except Exception:
                payload = {}

            target_emotion = payload.get("emotion")
            override = payload.get("override", False)
            params = payload.get("params", {})

            # Capture live frame and emotion
            frame, box, e_data = WORKER.get_frame_and_emotion()

            # Synthesize graphic design meme poster with brand watermarks
            poster_pil, meta = poster_generator.generate_meme(
                frame=frame,
                emotion_data=e_data,
                target_emotion=target_emotion,
                box=box,
                params=params
            )

            # Assign unique token for this specific meme
            token = uuid.uuid4().hex[:12]
            filename = f"meme_{meta['emotion'].lower()}_{token}.png"
            file_path = os.path.join(OUTPUTS_DIR, filename)
            poster_pil.save(file_path, format="PNG")

            # Mobile download URL (points to phone-accessible LAN IP)
            mobile_url = f"http://{SERVER_IP}:{SERVER_PORT}/m/{token}"

            # Generate dynamic Phone Download QR Code (QR Code 2)
            download_qr_b64 = generate_qr_data_url(mobile_url)

            # In-memory token cache with timestamp
            MEME_STORE[token] = {
                "filepath": file_path,
                "created_at": time.time(),
                "metadata": {**meta, "symbol": emotion_engine.EMOTIONS.get(meta["emotion"], {}).get("symbol", "✨")}
            }

            # Periodic cleanup
            cleanup_expired_memes()

            # Base64 data for immediate desktop preview
            buffered = io.BytesIO()
            poster_pil.save(buffered, format="PNG")
            img_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

            response_data = {
                "success": True,
                "token": token,
                "image_url": f"/api/meme/{token}",
                "download_url": f"/api/download/{token}",
                "image_b64": img_b64,
                "download_qr_b64": download_qr_b64,
                "mobile_url": mobile_url,
                "emotion": meta["emotion"],
                "label": meta["label"],
                "symbol": emotion_engine.EMOTIONS.get(meta["emotion"], {}).get("symbol", "✨"),
                "caption": meta["caption"],
                "confidence": meta["confidence"],
                "design_id": meta["design_id"],
                "timestamp": meta["timestamp"],
                "is_override": meta["is_override"],
                "palette": emotion_engine.EMOTIONS.get(meta["emotion"], {}).get("palette", []),
            }
            self.send_json(response_data)
            return

        self.send_error(404, "Endpoint not found")

    def stream_video(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()

        try:
            while True:
                with WORKER.lock:
                    jpeg_bytes = WORKER.latest_jpeg
                if jpeg_bytes is not None:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b'\r\n')
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_html_response(self, html_str, status=200):
        body = html_str.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file_response(self, file_path, content_type="application/octet-stream", as_attachment=False, filename=None):
        with open(file_path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        if as_attachment and filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        if "/video_feed" in str(args) or "/api/status" in str(args):
            return
        super().log_message(format, *args)


def main():
    global WORKER, SERVER_PORT, SERVER_IP
    parser = argparse.ArgumentParser(description="REACTIONARY — AI Emotion-to-Meme Generator")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--ip", type=str, default=None, help="Custom server IP for mobile phone QR code")
    parser.add_argument("--size", default="640x480", help="Webcam capture size")
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror camera frame")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    SERVER_PORT = args.port
    SERVER_IP = args.ip or get_local_ip()

    print("=" * 72)
    print("           R E A C T I O N A R Y  —  A I  M E M E  S T U D I O")
    print("                  GRAPHICA CREATIVE-TECH BOOTH")
    print("=" * 72)

    WORKER = CameraWorker(camera_idx=args.camera, size=args.size, no_flip=args.no_flip)
    WORKER.start()

    server_address = ('', SERVER_PORT)
    httpd = ThreadingHTTPServer(server_address, ReactionaryHTTPHandler)

    local_url = f"http://localhost:{SERVER_PORT}"
    network_url = f"http://{SERVER_IP}:{SERVER_PORT}"

    print(f"\n[DESKTOP UI]  {local_url}")
    print(f"[PHONE ACCESS] {network_url}")
    print("  - Real-Time 10-Emotion Detection")
    print("  - Interactive Emotion Override Selector")
    print("  - Graphic-Design Meme Synthesis with Club Logo & Instagram QR")
    print("  - Phone Download QR Code & Mobile Web Landing Page (/m/<token>)\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(local_url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down REACTIONARY...")
    finally:
        WORKER.running = False
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
