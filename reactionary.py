#!/usr/bin/env python3
"""
reactionary.py — Core application server for REACTIONARY.

Transforms facial reactions into graphic design posters:
1. Captures live webcam video and processes facial landmarks via MediaPipe.
2. Extracts blendshapes and calculates resting-face deviation via EmotionEngine.
3. Translates reaction into visual design languages (Auto or Manual mode).
4. Serves an interactive experimental graphic design web tool on localhost:8000.
5. Synthesizes high-resolution editorial posters on demand.
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import emotion_engine
import poster_generator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
WEB_DIR = os.path.join(HERE, "web")


class Clock:
    """Strictly increasing monotonic timestamps for MediaPipe."""
    def __init__(self):
        self.t0 = time.monotonic()
        self.last = -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class CameraWorker(threading.Thread):
    """Background worker continuously capturing webcam frames and tracking face/emotions."""
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
        self.face_box = None
        self.frame_w = 640
        self.frame_h = 480
        self.fps = 0.0
        
        self.emotion_engine = emotion_engine.EmotionEngine()
        self.emotion_data = {
            "emotion": "NEUTRAL",
            "label": "Unimpressed",
            "symbol": "😐",
            "style": "Swiss",
            "description": "Restrained, balanced, minimal expression",
            "palette": ["#000000", "#FFFFFF", "#FF3B00", "#8E8E93"],
            "typography": "Helvetica / Grotesque",
            "confidence": 85,
            "telemetry": {},
            "is_calibrated": not self.emotion_engine.baseline.generic,
        }

    def run(self):
        print(f"[REACTIONARY] Initializing Camera Index {self.camera_idx} ...")
        cap = cv2.VideoCapture(self.camera_idx)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_idx}.")
            return
            
        if "x" in self.size:
            sw, sh = self.size.lower().split("x")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(sw))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(sh))
            
        # Ensure model is present
        model_path = os.path.join(HERE, "models", "face_landmarker.task")
        if not os.path.isfile(model_path):
            print(f"[ERROR] Missing model at {model_path}")
            return
            
        clock = Clock()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
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
                
                # MediaPipe inference
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = face_detector.detect_for_video(mp_img, ts)
                
                has_face = False
                num_faces = len(result.face_landmarks) if result.face_landmarks else 0
                face_box = None
                
                if num_faces > 0:
                    has_face = True
                    lms = result.face_landmarks[0]
                    pts = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
                    x0, y0 = pts.min(0)
                    x1, y1 = pts.max(0)
                    face_box = (int(x0), int(y0), int(x1), int(y1))
                    
                    # Compute turn
                    cl, cr, nose = pts[234], pts[454], pts[1]
                    turn_signed = float((nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
                    
                    # Blendshapes
                    bs_dict = {}
                    if result.face_blendshapes:
                        for c in result.face_blendshapes[0]:
                            bs_dict[c.category_name] = c.score
                            
                    # Analyze via EmotionEngine
                    e_res = self.emotion_engine.analyze(bs_dict, turn_signed)
                else:
                    e_res = self.emotion_data

                # Encode JPEG for live MJPEG streaming
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


# Global worker reference
WORKER = None
POSTER_CACHE = {}


class ReactionaryHTTPHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler serving Web UI, API, and MJPEG stream."""
    
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
            
        elif path == "/api/styles":
            self.send_json({"styles": emotion_engine.ALL_STYLES})
            return
            
        elif path.startswith("/api/poster/"):
            filename = os.path.basename(path)
            file_path = os.path.join(OUTPUTS_DIR, filename)
            if os.path.isfile(file_path):
                self.send_file_response(file_path, content_type="image/png")
            else:
                self.send_error(404, "Poster not found")
            return
            
        elif path.startswith("/api/download/"):
            filename = os.path.basename(path)
            file_path = os.path.join(OUTPUTS_DIR, filename)
            if os.path.isfile(file_path):
                self.send_file_response(file_path, content_type="image/png", as_attachment=True, filename=filename)
            else:
                self.send_error(404, "File not found")
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
                
            style = payload.get("style", "Swiss")
            params = payload.get("params", {})
            
            # Capture latest frame and reaction
            frame, box, e_data = WORKER.get_frame_and_emotion()
            
            # Synthesize poster
            poster_pil, meta = poster_generator.generate_poster(
                frame=frame,
                emotion_data=e_data,
                style_name=style,
                box=box,
                params=params
            )
            
            # Save to outputs directory
            design_id_clean = meta["design_id"].replace("#", "")
            filename = f"poster_{design_id_clean}_{int(time.time())}.png"
            file_path = os.path.join(OUTPUTS_DIR, filename)
            poster_pil.save(file_path, format="PNG")
            
            # Buffer base64 for immediate display
            buffered = io.BytesIO()
            poster_pil.save(buffered, format="PNG")
            img_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            response_data = {
                "success": True,
                "image_url": f"/api/poster/{filename}",
                "download_url": f"/api/download/{filename}",
                "image_b64": img_b64,
                "emotion": meta["emotion"],
                "label": meta["label"],
                "style": meta["style"],
                "confidence": meta["confidence"],
                "design_id": meta["design_id"],
                "timestamp": meta.get("timestamp", time.strftime("%Y.%m.%d // %H:%M:%S")),
                "palette": emotion_engine.EMOTIONS.get(meta["emotion"], {}).get("palette", []),
                "typography": emotion_engine.EMOTIONS.get(meta["emotion"], {}).get("typography", ""),
            }
            self.send_json(response_data)
            return
            
        self.send_error(404, "Endpoint not found")

    def stream_video(self):
        """Stream continuous MJPEG feed to client."""
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
                time.sleep(0.033)  # ~30 fps
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
        # Suppress routine GET logs for clean console
        if "/video_feed" in str(args) or "/api/status" in str(args):
            return
        super().log_message(format, *args)


def main():
    global WORKER
    parser = argparse.ArgumentParser(description="REACTIONARY — Your face. Your emotion. Your design.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--size", default="640x480", help="Webcam resolution, e.g. 640x480 or 1280x720")
    parser.add_argument("--no-flip", action="store_true", help="Do not mirror the camera feed")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser")
    args = parser.parse_args()

    print("=" * 70)
    print("                R E A C T I O N A R Y")
    print("      “Your face. Your emotion. Your design.”")
    print("=" * 70)
    
    # Start Camera & Tracking Worker
    WORKER = CameraWorker(camera_idx=args.camera, size=args.size, no_flip=args.no_flip)
    WORKER.start()
    
    # Start Web Server
    server_address = ('', args.port)
    httpd = ThreadingHTTPServer(server_address, ReactionaryHTTPHandler)
    url = f"http://localhost:{args.port}"
    print(f"\n[SERVER] REACTIONARY Studio running at: {url}")
    print("  - Auto / Manual Design Style Translation")
    print("  - 10 Graphic Design Languages (Swiss, Brutalist, Y2K, Editorial, etc.)")
    print("  - High-Resolution (1200x1600) Poster Synthesis")
    print("\nPress Ctrl+C in terminal to stop.")
    
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down REACTIONARY...")
    finally:
        WORKER.running = False
        httpd.server_close()
        print("Done.")


if __name__ == "__main__":
    main()

