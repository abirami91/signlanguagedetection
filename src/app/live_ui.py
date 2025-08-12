# src/app/live_ui.py
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import mediapipe as mp, cv2, time, os

# --------- tunables ---------
W, H = 640, 480          # try 800x600 if your Pi keeps up; drop to 424x240 if it lags
SLEEP = 0.04             # ~25 fps; use 0.03 (~33 fps) if smooth
JPEG_QUALITY = 80        # 60–85 is usually a good tradeoff
# ----------------------------

# --- camera ---
picam2 = Picamera2()
cfg = picam2.create_preview_configuration({"format": "RGB888", "size": (W, H)})
picam2.configure(cfg)
picam2.start()

# --- mediapipe ---
hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, model_complexity=0,
    min_detection_confidence=0.4, min_tracking_confidence=0.4
)
draw = mp.solutions.drawing_utils
current_text = "Starting…"

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Sign Language — Live Demo</title>
<style>
:root{--bg:#0f172a;--fg:#e2e8f0;--muted:#94a3b8;--card:#111827;--border:#1f2937;--accent:#22d3ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
.header{padding:24px 16px;text-align:center}
.title{font-weight:700;letter-spacing:.2px;font-size:clamp(22px,3vw,32px)}
.subtitle{color:var(--muted);margin-top:6px}
.main{display:flex;align-items:center;justify-content:center;padding:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.35);max-width:min(98vw,1200px)}
.status{display:flex;gap:8px;align-items:center;margin-bottom:12px}
.pill{font-weight:600;padding:6px 10px;border-radius:999px;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.3);color:#67e8f9}
img{display:block;width:100%;height:auto;border-radius:10px}
.footer{text-align:center;color:var(--muted);padding:18px;font-size:12px}
</style>
</head>
<body>
  <div class="header">
    <div class="title">Sign Language — Live Demo</div>
    <div class="subtitle">Landmarks streamed from your Raspberry Pi camera</div>
  </div>
  <main class="main">
    <div class="card">
      <div class="status">
        <span class="pill" id="badge">{{t}}</span>
      </div>
      <img id="live" src="/video" alt="Live stream"/>
    </div>
  </main>
  <div class="footer">Tip: good lighting and centered hands improve detection.</div>
<script>
  const badge = document.getElementById('badge');
  setInterval(()=>fetch('/text').then(r=>r.text()).then(x=>badge.textContent=x), 200);
</script>
</body>
</html>
"""

def frame_generator():
    global current_text
    while True:
        rgb = picam2.capture_array()          # HxWx3 RGB for MediaPipe
        res = hands.process(rgb)

        # Draw on BGR (for OpenCV colors) and JPEG-encode
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                draw.draw_landmarks(bgr, lm, mp.solutions.hands.HAND_CONNECTIONS)
            current_text = f"Hands: {len(res.multi_hand_landmarks)}"
        else:
            current_text = "No hand"

        ok, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")
        time.sleep(SLEEP)

@app.route("/")
def index():
    return render_template_string(PAGE, t=current_text)

@app.route("/video")
def video():
    return Response(frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/text")
def text():
    return current_text

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        hands.close()
        picam2.stop()
