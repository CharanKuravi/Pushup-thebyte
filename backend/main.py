"""
FastAPI backend for AI Workout Tracker.
Supports: Push-ups, Plank, Squats, Abs (Crunches), Wall Sit.
Accepts base64-encoded frames via WebSocket and streams back rep/time count + form data.
"""

import cv2
import numpy as np
import base64
import time
import os
import csv
import urllib.request
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from coach import SessionTracker, remember_session

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

def ensure_model():
    if not os.path.isfile(MODEL_PATH):
        print("[INFO] Downloading pose model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model ready.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pushup thresholds
PUSHUP_DOWN_THRESH  = 95    # elbow angle below this = bottom position
PUSHUP_UP_THRESH    = 155   # elbow angle above this = top position
PUSHUP_BODY_MIN     = 165   # shoulder-hip-ankle below this = hips sagging
PUSHUP_BODY_MAX     = 185   # above this = hips piking
PUSHUP_FLARE_MAX    = 75    # elbow-shoulder-hip above this = elbows flaring
PUSHUP_WRIST_MIN    = 160   # elbow-wrist-finger below this = bent wrist

# MediaPipe landmark indices
R_EAR=8;  R_SHOULDER=12; R_ELBOW=14; R_WRIST=16; R_HIP=24; R_KNEE=26; R_ANKLE=28
L_EAR=7;  L_SHOULDER=11; L_ELBOW=13; L_WRIST=15; L_HIP=23; L_KNEE=25; L_ANKLE=27
L_INDEX=19; R_INDEX=20   # index finger tips for wrist alignment check

# Squat thresholds — exact values from joint angle analysis
SQUAT_KNEE_DOWN      = 95    # knee angle below this triggers DOWN state
SQUAT_KNEE_UP        = 155   # knee angle above this triggers UP state (rep counted)
SQUAT_HIP_MIN        = 65    # hip must hinge this far down (shoulder-hip-knee)
SQUAT_HIP_MAX        = 95    # hip angle above this means no real hip hinge
SQUAT_TORSO_MIN      = 60    # minimum torso-vs-vertical angle (below = too much lean)
SQUAT_TORSO_MAX      = 90    # above this = too upright (only possible with heel raise)
SQUAT_ANKLE_MAX      = 88    # ankle dorsiflexion beyond this = heel rising
SQUAT_VALGUS_RATIO   = 0.85  # knee width must be >= 85% of hip width

# Plank thresholds — angle-based, side camera
PLANK_BODY_MIN   = 160   # shoulder-hip-ankle below this = hips sagging (loosened from 165)
PLANK_BODY_MAX   = 195   # above this = hips piking (loosened from 185)
PLANK_NECK_MIN   = 160   # ear-shoulder-hip below this = head drooping
PLANK_NECK_MAX   = 195   # above this = craning neck up
PLANK_ELBOW_MIN  = 75    # shoulder-elbow-wrist below this = elbows too far forward
PLANK_ELBOW_MAX  = 105   # above this = elbows too far back

# Abs / crunch thresholds
ABS_UP_THRESH    = 85    # hip angle below this = crunched up
ABS_DOWN_THRESH  = 150   # hip angle above this = flat (rep counted)
ABS_MIN_DELTA    = 45    # minimum angle swing required to register a state change
ABS_NECK_MIN     = 140   # ear-shoulder-hip below this = neck pulling

# Wall sit thresholds — all 3 joints must be in range simultaneously
WALL_KNEE_MIN   = 80     # hip-knee-ankle below this = too deep
WALL_KNEE_MAX   = 100    # above this = too shallow
WALL_HIP_MIN    = 80     # shoulder-hip-knee below this = leaning forward
WALL_HIP_MAX    = 100    # above this = not sitting deep enough
WALL_TORSO_MIN  = 80     # vertical-hip-shoulder below this = back off wall

HISTORY_FILE = os.environ.get("HISTORY_FILE", "workout_history.csv")
CSV_HEADERS  = ["Date", "Exercise", "Total_Reps", "Duration_Seconds"]

# ---------------------------------------------------------------------------
# Math helpers (same as pushup_counter.py)
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    a = np.array([a[0], a[1], 0.0])
    b = np.array([b[0], b[1], 0.0])
    c = np.array([c[0], c[1], 0.0])
    ba = a - b; bc = c - b
    cross = np.cross(ba, bc)[2]
    dot   = np.dot(ba, bc)
    return abs(np.degrees(np.arctan2(cross, dot)))


def angle_arccos(a, b, c):
    """
    Exact formula: arccos of the dot product.
    a, b, c are [x, y] coords. b is the vertex joint.
    Used for all squat joint checks.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))

def point_to_line_distance(p, line_a, line_b):
    p = np.array(p, dtype=float)
    line_a = np.array(line_a, dtype=float)
    line_b = np.array(line_b, dtype=float)
    d = line_b - line_a
    norm = np.linalg.norm(d)
    if norm == 0:
        return float(np.linalg.norm(p - line_a))
    return float(abs(np.cross(d, line_a - p)) / norm)

def check_hip_alignment(shoulder, hip, ankle):
    dist = point_to_line_distance(hip, shoulder, ankle)
    return dist, dist < HIP_SAG_PX

def check_head_position(ear, shoulder, hip):
    mid_y = (shoulder[1] + hip[1]) / 2.0
    diff  = mid_y - ear[1]
    return diff, diff > -HEAD_DROP_MARGIN

def check_wrist_stack(shoulder, wrist):
    diff = abs(shoulder[0] - wrist[0])
    return diff, diff < WRIST_ALIGN_TOL

def get_side_landmarks(lms, w, h):
    def vis(idx):
        return lms[idx].visibility if lms[idx].visibility is not None else 0.0
    right_score = vis(R_SHOULDER) + vis(R_ELBOW) + vis(R_HIP) + vis(R_ANKLE)
    left_score  = vis(L_SHOULDER) + vis(L_ELBOW) + vis(L_HIP) + vis(L_ANKLE)
    use_right   = right_score >= left_score
    ids = (R_EAR, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE, R_INDEX) if use_right else \
          (L_EAR, L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE, L_INDEX)
    def c(idx):
        lm = lms[idx]
        return [lm.x * w, lm.y * h]
    return {k: c(ids[i]) for i, k in enumerate(["ear","shoulder","elbow","wrist","hip","knee","ankle","finger"])}


def get_front_landmarks(lms, w, h):
    """Average left+right for front-facing exercises (squats, wall sit, abs)."""
    def c(idx):
        lm = lms[idx]
        return [lm.x * w, lm.y * h]
    def avg(a, b):
        pa, pb = c(a), c(b)
        return [(pa[0]+pb[0])/2, (pa[1]+pb[1])/2]
    # Foot index landmarks: 31 = left foot index, 32 = right foot index
    L_FOOT_IDX = 31
    R_FOOT_IDX = 32
    return {
        "ear":       avg(L_EAR, R_EAR),
        "shoulder":  avg(L_SHOULDER, R_SHOULDER),
        "elbow":     avg(L_ELBOW, R_ELBOW),
        "wrist":     avg(L_WRIST, R_WRIST),
        "hip":       avg(L_HIP, R_HIP),
        "knee":      avg(L_KNEE, R_KNEE),
        "ankle":     avg(L_ANKLE, R_ANKLE),
        "foot":      avg(L_FOOT_IDX, R_FOOT_IDX),
        # individual sides for valgus and symmetry checks
        "l_knee":    c(L_KNEE),
        "r_knee":    c(R_KNEE),
        "l_hip":     c(L_HIP),
        "r_hip":     c(R_HIP),
        "l_ankle":   c(L_ANKLE),
        "r_ankle":   c(R_ANKLE),
        "l_foot":    c(L_FOOT_IDX),
        "r_foot":    c(R_FOOT_IDX),
    }


def save_session(total_reps, duration_seconds, exercise="pushup"):
    file_exists = os.path.isfile(HISTORY_FILE)
    # Migrate old CSV that lacks Exercise column
    if file_exists:
        with open(HISTORY_FILE, newline="") as f:
            sample = f.read(256)
        if "Exercise" not in sample:
            # rewrite with new header
            with open(HISTORY_FILE, newline="") as f:
                old_rows = list(csv.DictReader(f))
            with open(HISTORY_FILE, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                for r in old_rows:
                    writer.writerow({
                        "Date": r.get("Date",""),
                        "Exercise": "pushup",
                        "Total_Reps": r.get("Total_Reps","0"),
                        "Duration_Seconds": r.get("Duration_Seconds","0"),
                    })
            file_exists = True

    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Exercise": exercise,
            "Total_Reps": total_reps,
            "Duration_Seconds": round(duration_seconds, 2),
        })

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REST: workout history
# ---------------------------------------------------------------------------

@app.get("/history")
def get_history():
    if not os.path.isfile(HISTORY_FILE):
        return []
    rows = []
    with open(HISTORY_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # back-compat: add Exercise field if missing
            if "Exercise" not in row:
                row["Exercise"] = "pushup"
            rows.append(row)
    return rows


@app.delete("/history/{date}")
def delete_session(date: str):
    if not os.path.isfile(HISTORY_FILE):
        return {"ok": False}
    rows = []
    with open(HISTORY_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["Date"] != date]
    with open(HISTORY_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return {"ok": True}


# ---------------------------------------------------------------------------
# REST: AI coaching — recalls Hindsight memory, routes via cascadeflow
# ---------------------------------------------------------------------------

@app.get("/coach/{user_id}/{exercise}")
async def get_coach_message(user_id: str, exercise: str, reps: int = 0):
    """
    Returns a personalized coaching message for the user.
    Hindsight recalls past sessions. cascadeflow routes to the right model.
    """
    tracker = SessionTracker(user_id=user_id, exercise=exercise)
    message = await tracker.coaching_message(current_reps=reps)
    return {"message": message}

# ---------------------------------------------------------------------------
# WebSocket: Push-up detection
# Side camera, hip height, 1-1.5m away.
# 4 angles: elbow (primary rep trigger), body line, shoulder flare, wrist.
# 5-frame smoothing on elbow angle. 10-frame lockout after each rep.
# Rep logic: up -> down -> up = 1 rep. Form issues shown as feedback only.
# Visibility gate: elbow and hip must be confidently detected.
# ---------------------------------------------------------------------------

@app.websocket("/ws/pushup")
async def pushup_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    rep_count  = 0
    state      = "up"    # "up" = top | "down" = bottom
    lockout    = 0
    start_time = time.time()
    feedback   = ""
    tracker    = SessionTracker(user_id="default", exercise="pushup")

    from collections import deque
    elbow_buffer: deque = deque(maxlen=5)

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            response = {
                "reps": rep_count,
                "state": state.upper(),
                "feedback": feedback,
                "pose_detected": False,
                "checks": {},
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = get_side_landmarks(lms, w, h)

                shoulder = pts["shoulder"]
                elbow    = pts["elbow"]
                wrist    = pts["wrist"]
                hip      = pts["hip"]
                ankle    = pts["ankle"]
                finger   = pts["finger"]

                # Visibility gate — elbow and hip must be visible
                def vis(idx):
                    v = lms[idx].visibility
                    return v if v is not None else 0.0
                elbow_vis = max(vis(L_ELBOW), vis(R_ELBOW))
                hip_vis   = max(vis(L_HIP),   vis(R_HIP))
                body_visible = elbow_vis > 0.5 and hip_vis > 0.5

                # Angle 1: elbow — shoulder-elbow-wrist (primary, smoothed)
                raw_elbow = angle_arccos(shoulder, elbow, wrist)
                elbow_buffer.append(raw_elbow)
                elbow_angle = float(np.mean(elbow_buffer))

                # Angle 2: body line — shoulder-hip-ankle
                body_line = angle_arccos(shoulder, hip, ankle)

                # Angle 3: shoulder flare — elbow-shoulder-hip
                shoulder_ang = angle_arccos(elbow, shoulder, hip)

                # Angle 4: wrist alignment — elbow-wrist-finger
                wrist_ang = angle_arccos(elbow, wrist, finger)

                # Form checks — feedback only, never block rep
                form_notes = []
                if body_line < PUSHUP_BODY_MIN:
                    form_notes.append("Hips sagging — squeeze glutes and core")
                elif body_line > PUSHUP_BODY_MAX:
                    form_notes.append("Hips too high — lower your butt")
                if shoulder_ang > PUSHUP_FLARE_MAX:
                    form_notes.append("Elbows flaring — tuck them in")
                if wrist_ang < PUSHUP_WRIST_MIN:
                    form_notes.append("Straighten your wrists")
                if state == "down" and elbow_angle > 105:
                    form_notes.append("Go lower — chest closer to ground")

                if body_visible:
                    if lockout > 0:
                        lockout -= 1
                    else:
                        # Top -> bottom
                        if state == "up" and elbow_angle < PUSHUP_DOWN_THRESH:
                            state = "down"

                        # Bottom -> top = 1 rep
                        elif state == "down" and elbow_angle > PUSHUP_UP_THRESH:
                            state     = "up"
                            rep_count += 1
                            lockout   = 10
                            if form_notes:
                                tracker.record_issue(form_notes[0])
                                feedback = f"Rep {rep_count} — {form_notes[0]}"
                            else:
                                feedback = f"Rep {rep_count}"
                else:
                    feedback = "Point camera at side view, hip height"

                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]
                response = {
                    "reps": rep_count,
                    "state": state.upper(),
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "elbow_angle":  round(elbow_angle, 1),
                        "body_line":    round(body_line, 1),
                        "shoulder_ang": round(shoulder_ang, 1),
                        "wrist_ang":    round(wrist_ang, 1),
                        "hip_ok":       PUSHUP_BODY_MIN <= body_line <= PUSHUP_BODY_MAX,
                        "flare_ok":     shoulder_ang <= PUSHUP_FLARE_MAX,
                        "wrist_ok":     wrist_ang >= PUSHUP_WRIST_MIN,
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(rep_count, elapsed, "pushup")
        await tracker.save(rep_count, elapsed)
        landmarker.close()
        print(f"[INFO] Pushup session ended. Reps: {rep_count}")


# ---------------------------------------------------------------------------
# WebSocket: Squat detection
# Front-facing camera.
# 4 joints tracked: knee, hip, torso vs vertical, ankle dorsiflexion.
# Valgus check: knee width vs hip width.
# 5-frame rolling average on knee angle to kill false reps from noise.
# Rep state machine: knee < 95 = DOWN, knee > 155 = UP -> rep counted.
# Hip depth validation: hip angle must drop below 95 to confirm real squat.
# ---------------------------------------------------------------------------

@app.websocket("/ws/squat")
async def squat_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    rep_count        = 0
    state            = "up"   # "up" | "down"
    form_failed      = False
    form_fail_reason = ""
    start_time       = time.time()
    feedback         = ""
    hip_depth_reached = False
    tracker          = SessionTracker(user_id="default", exercise="squat")

    # 5-frame rolling buffer for knee angle smoothing
    knee_buffer: list[float] = []

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            response = {
                "reps": rep_count, "state": state.upper(),
                "feedback": feedback, "pose_detected": False, "checks": {},
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = get_front_landmarks(lms, w, h)

                shoulder = pts["shoulder"]
                hip      = pts["hip"]
                knee     = pts["knee"]
                ankle    = pts["ankle"]
                foot     = pts["foot"]

                # Joint 1: Knee angle (hip -> knee -> ankle) — primary rep trigger
                raw_knee = angle_arccos(hip, knee, ankle)
                knee_buffer.append(raw_knee)
                if len(knee_buffer) > 5:
                    knee_buffer.pop(0)
                knee_angle = float(np.mean(knee_buffer))

                # Joint 2: Hip angle (shoulder -> hip -> knee) — depth validation
                # Knee can bend without real hip hinge; this catches shallow squats
                hip_angle = angle_arccos(shoulder, hip, knee)

                # Joint 3: Torso angle vs vertical
                # Build a point directly above hip to represent the vertical axis
                vertical_ref = [hip[0], hip[1] - 100]
                torso_angle  = angle_arccos(vertical_ref, hip, shoulder)

                # Joint 4: Ankle dorsiflexion (knee -> ankle -> foot index)
                ankle_angle = angle_arccos(knee, ankle, foot)

                # Valgus check: knee horizontal width vs hip horizontal width
                hip_width  = abs(pts["r_hip"][0]  - pts["l_hip"][0])
                knee_width = abs(pts["r_knee"][0] - pts["l_knee"][0])
                valgus_ok  = knee_width >= hip_width * SQUAT_VALGUS_RATIO

                # Derived form booleans
                hip_ok     = hip_angle <= SQUAT_HIP_MAX          # hip actually hinged
                torso_ok   = torso_angle >= SQUAT_TORSO_MIN       # not too much forward lean
                ankle_ok   = ankle_angle <= SQUAT_ANKLE_MAX       # heel not rising
                depth_ok   = knee_angle <= SQUAT_KNEE_DOWN        # reached parallel

                # State machine
                # DOWN: person goes from standing into squat (knee crosses below 95)
                if state == "up" and knee_angle < SQUAT_KNEE_DOWN:
                    state             = "down"
                    form_failed       = False
                    form_fail_reason  = ""
                    hip_depth_reached = False   # reset for this rep

                # While in DOWN phase: run form checks + track hip depth
                if state == "down":
                    # Track hip depth — check if hip hinged at any point during descent
                    if hip_angle <= SQUAT_HIP_MAX:
                        hip_depth_reached = True

                    # Form checks in priority order (most dangerous first)
                    if not torso_ok and not form_failed:
                        form_failed      = True
                        form_fail_reason = "Too much forward lean"
                    elif not valgus_ok and not form_failed:
                        form_failed      = True
                        form_fail_reason = "Knees caving in — push them out"
                    elif not ankle_ok and not form_failed:
                        form_failed      = True
                        form_fail_reason = "Heel rising — work on ankle mobility"

                # UP: person stands back up (knee crosses above 155)
                # 1 DOWN + 1 UP = 1 rep — always counted, form issues shown as feedback only
                if state == "down" and knee_angle > SQUAT_KNEE_UP:
                    state = "up"
                    rep_count += 1
                    if form_fail_reason:
                        tracker.record_issue(form_fail_reason)
                        feedback = f"Rep {rep_count} — {form_fail_reason}"
                    elif not hip_depth_reached:
                        feedback = f"Rep {rep_count} — go deeper next time"
                    else:
                        feedback = f"Rep {rep_count}"

                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]
                response = {
                    "reps": rep_count,
                    "state": state.upper(),
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "knee_angle":  round(knee_angle, 1),
                        "hip_angle":   round(hip_angle, 1),
                        "torso_angle": round(torso_angle, 1),
                        "ankle_angle": round(ankle_angle, 1),
                        "hip_ok":      hip_ok,
                        "torso_ok":    torso_ok,
                        "ankle_ok":    ankle_ok,
                        "valgus_ok":   valgus_ok,
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(rep_count, elapsed, "squat")
        await tracker.save(rep_count, elapsed)
        landmarker.close()
        print(f"[INFO] Squat session ended. Reps: {rep_count}")


# ---------------------------------------------------------------------------
# WebSocket: Plank detection
# Side-facing camera. Timed hold, no reps.
# 3 angles: body line (shoulder-hip-ankle), neck line (ear-shoulder-hip),
# elbow angle (shoulder-elbow-wrist).
# Timer only accumulates when all 3 angles are in range.
# Pauses and resumes — counts quality hold time, not suffering time.
# ---------------------------------------------------------------------------

@app.websocket("/ws/plank")
async def plank_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    hold_start     = None    # wall-clock time when good form started
    total_hold_sec = 0.0     # accumulated good-form seconds
    in_position    = False
    start_time     = time.time()
    feedback       = ""

    from collections import deque
    body_buffer: deque = deque(maxlen=3)   # smooth body line angle

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            # Current live hold time = accumulated + ongoing segment
            live_hold = total_hold_sec
            if hold_start is not None:
                live_hold += time.time() - hold_start

            response = {
                "reps": round(live_hold, 1),
                "state": "HOLD" if in_position else "REST",
                "feedback": feedback, "pose_detected": False, "checks": {},
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = get_side_landmarks(lms, w, h)

                shoulder = pts["shoulder"]
                hip      = pts["hip"]
                ankle    = pts["ankle"]
                ear      = pts["ear"]
                elbow    = pts["elbow"]
                wrist    = pts["wrist"]

                # Visibility gate — hip and ankle must be confidently detected.
                # Side-view plank requires the full body in frame.
                def vis(idx):
                    v = lms[idx].visibility
                    return v if v is not None else 0.0

                hip_vis   = max(vis(L_HIP),   vis(R_HIP))
                ankle_vis = max(vis(L_ANKLE), vis(R_ANKLE))
                body_visible = hip_vis > 0.5 and ankle_vis > 0.5

                # Angle 1: body line — shoulder-hip-ankle, smoothed
                raw_body  = angle_arccos(shoulder, hip, ankle)
                body_buffer.append(raw_body)
                body_line = float(np.mean(body_buffer))

                # Angle 2: neck line — ear-shoulder-hip
                neck_line = angle_arccos(ear, shoulder, hip)

                # Angle 3: elbow angle — shoulder-elbow-wrist
                elbow_angle = angle_arccos(shoulder, elbow, wrist)

                body_ok  = PLANK_BODY_MIN  <= body_line  <= PLANK_BODY_MAX
                neck_ok  = PLANK_NECK_MIN  <= neck_line  <= PLANK_NECK_MAX
                elbow_ok = PLANK_ELBOW_MIN <= elbow_angle <= PLANK_ELBOW_MAX

                all_ok = body_visible and body_ok and neck_ok and elbow_ok

                now = time.time()
                if all_ok:
                    if hold_start is None:
                        hold_start = now
                    in_position = True
                    live_hold   = total_hold_sec + (now - hold_start)
                    feedback    = f"{round(live_hold, 1)}s"
                else:
                    if hold_start is not None:
                        total_hold_sec += now - hold_start
                        hold_start = None
                    in_position = False
                    live_hold   = total_hold_sec
                    if not body_visible:
                        feedback = "Point camera at full body — side view"
                    elif not body_ok:
                        if body_line < PLANK_BODY_MIN:
                            feedback = "Hips sagging — squeeze your core"
                        else:
                            feedback = "Hips too high — lower your butt"
                    elif not neck_ok:
                        if neck_line < PLANK_NECK_MIN:
                            feedback = "Head drooping — look at the floor"
                        else:
                            feedback = "Don't crane your neck up"
                    elif not elbow_ok:
                        if elbow_angle < PLANK_ELBOW_MIN:
                            feedback = "Elbows too far forward"
                        else:
                            feedback = "Bring elbows under shoulders"

                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]
                response = {
                    "reps": round(live_hold, 1),
                    "state": "HOLD" if in_position else "REST",
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "body_line":   round(body_line, 1),
                        "neck_line":   round(neck_line, 1),
                        "elbow_angle": round(elbow_angle, 1),
                        "body_ok":     body_ok,
                        "neck_ok":     neck_ok,
                        "elbow_ok":    elbow_ok,
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(round(total_hold_sec), elapsed, "plank")
        await remember_session("default", "plank", round(total_hold_sec), elapsed, [])
        landmarker.close()
        print(f"[INFO] Plank session ended. Hold: {round(total_hold_sec, 1)}s")


# ---------------------------------------------------------------------------
# WebSocket: Abs / Crunch detection
# Side camera, lying on back.
# Primary angle: shoulder-hip-knee (hip flexion).
#   Flat = 170-180, good crunch top = 55-80, state machine triggers at 80/140.
# Neck angle: ear-shoulder-hip. Below 140 = pulling neck. Shown as feedback only.
# 5-frame deque smoothing on hip angle. 8-frame lockout after each rep.
# Rep logic: down (flat) -> up (crunched) -> back to down = 1 rep.
# Form issues never block the count.
# ---------------------------------------------------------------------------

@app.websocket("/ws/abs")
async def abs_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    rep_count  = 0
    state      = "down"
    lockout    = 0
    start_time = time.time()
    feedback   = ""
    peak_angle = None
    base_angle = None
    tracker    = SessionTracker(user_id="default", exercise="abs")

    from collections import deque
    hip_buffer: deque = deque(maxlen=3)

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            response = {
                "reps": rep_count, "state": state.upper(),
                "feedback": feedback, "pose_detected": False, "checks": {},
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                # Use averaged left+right landmarks — more stable when lying on back
                pts = get_front_landmarks(lms, w, h)

                shoulder = pts["shoulder"]
                hip      = pts["hip"]
                knee     = pts["knee"]
                ear      = pts["ear"]

                # Visibility gate — hip and knee landmarks must both be confidently
                # detected on at least one side. If the camera only sees the head,
                # these will be near zero and we skip the state machine entirely.
                def vis(idx):
                    v = lms[idx].visibility
                    return v if v is not None else 0.0

                hip_vis  = max(vis(L_HIP),  vis(R_HIP))
                knee_vis = max(vis(L_KNEE), vis(R_KNEE))
                body_visible = hip_vis > 0.5 and knee_vis > 0.5

                # Primary angle: shoulder -> hip -> knee (hip flexion)
                # Flat on ground = ~170-180, good crunch = ~55-90
                raw_hip  = angle_arccos(shoulder, hip, knee)
                hip_buffer.append(raw_hip)
                hip_angle = float(np.mean(hip_buffer))

                # Neck angle: ear -> shoulder -> hip
                neck_angle = angle_arccos(ear, shoulder, hip)
                neck_ok    = neck_angle >= ABS_NECK_MIN

                # Only run state machine when body is actually visible
                if body_visible:
                    # Track the extremes of each phase
                    if state == "down":
                        if base_angle is None or hip_angle > base_angle:
                            base_angle = hip_angle
                    if state == "up":
                        if peak_angle is None or hip_angle < peak_angle:
                            peak_angle = hip_angle

                    if lockout > 0:
                        lockout -= 1
                    else:
                        # Flat -> crunched: must cross UP_THRESH
                        if state == "down" and hip_angle < ABS_UP_THRESH:
                            state      = "up"
                            peak_angle = hip_angle

                        # Crunched -> flat: must cross DOWN_THRESH AND
                        # total swing must be >= MIN_DELTA
                        elif state == "up" and hip_angle > ABS_DOWN_THRESH:
                            swing = (base_angle or ABS_DOWN_THRESH) - (peak_angle or ABS_UP_THRESH)
                            if swing >= ABS_MIN_DELTA:
                                state      = "down"
                                rep_count += 1
                                lockout    = 6
                                base_angle = hip_angle
                                if not neck_ok:
                                    tracker.record_issue("neck pulling")
                                    feedback = f"Rep {rep_count} — stop pulling your neck"
                                else:
                                    feedback = f"Rep {rep_count}"
                            else:
                                state      = "down"
                                base_angle = hip_angle
                else:
                    feedback = "Point camera at full body — side view"

                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]
                response = {
                    "reps": rep_count,
                    "state": state.upper(),
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "hip_angle":  round(hip_angle, 1),
                        "neck_angle": round(neck_angle, 1),
                        "neck_ok":    neck_ok,
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(rep_count, elapsed, "abs")
        await tracker.save(rep_count, elapsed)
        landmarker.close()
        print(f"[INFO] Abs session ended. Reps: {rep_count}")


# ---------------------------------------------------------------------------
# WebSocket: Wall Sit detection
# Front-facing camera. Timed hold, no reps.
# 3 angles: knee (hip-knee-ankle), hip (shoulder-hip-knee),
# torso vs vertical (vertical-hip-shoulder).
# All 3 must be in range simultaneously for timer to tick.
# One joint going out of range pauses the clock instantly.
# Timer accumulates across good-form segments. Best hold tracked.
# 5-frame smoothing on knee angle.
# ---------------------------------------------------------------------------

@app.websocket("/ws/wallsit")
async def wallsit_ws(websocket: WebSocket):
    await websocket.accept()
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    hold_start     = None
    total_hold_sec = 0.0
    best_hold_sec  = 0.0
    in_position    = False
    start_time     = time.time()
    feedback       = ""

    from collections import deque
    knee_buffer: deque = deque(maxlen=5)

    try:
        while True:
            data      = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            live_hold = total_hold_sec
            if hold_start is not None:
                live_hold += time.time() - hold_start

            response = {
                "reps": round(live_hold, 1),
                "state": "HOLD" if in_position else "REST",
                "feedback": feedback, "pose_detected": False, "checks": {},
            }

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                pts = get_front_landmarks(lms, w, h)

                shoulder = pts["shoulder"]
                hip      = pts["hip"]
                knee     = pts["knee"]
                ankle    = pts["ankle"]

                # Angle 1: knee — hip-knee-ankle (primary)
                raw_knee  = angle_arccos(hip, knee, ankle)
                knee_buffer.append(raw_knee)
                knee_angle = float(np.mean(knee_buffer))

                # Angle 2: hip — shoulder-hip-knee
                hip_angle  = angle_arccos(shoulder, hip, knee)

                # Angle 3: torso vs vertical — vertical point above hip
                vertical_ref = [hip[0], hip[1] - 100]
                torso_angle  = angle_arccos(vertical_ref, hip, shoulder)

                knee_ok  = WALL_KNEE_MIN  <= knee_angle  <= WALL_KNEE_MAX
                hip_ok   = WALL_HIP_MIN   <= hip_angle   <= WALL_HIP_MAX
                torso_ok = torso_angle >= WALL_TORSO_MIN

                all_ok = knee_ok and hip_ok and torso_ok

                now = time.time()
                if all_ok:
                    if hold_start is None:
                        hold_start = now
                    in_position = True
                    live_hold   = total_hold_sec + (now - hold_start)
                    if live_hold > best_hold_sec:
                        best_hold_sec = live_hold
                    feedback = f"{round(live_hold, 1)}s"
                else:
                    if hold_start is not None:
                        total_hold_sec += now - hold_start
                        hold_start = None
                    in_position = False
                    live_hold   = total_hold_sec
                    # Most critical feedback first
                    if not knee_ok:
                        if knee_angle > WALL_KNEE_MAX:
                            feedback = "Sit deeper — knees should be 90°"
                        else:
                            feedback = "Too deep — ease up slightly"
                    elif not hip_ok:
                        if hip_angle > WALL_HIP_MAX:
                            feedback = "Push back flat against the wall"
                        else:
                            feedback = "Don't lean forward"
                    elif not torso_ok:
                        feedback = "Back leaving the wall — straighten up"

                landmarks = [{"x": lm.x, "y": lm.y} for lm in lms]
                response = {
                    "reps": round(live_hold, 1),
                    "state": "HOLD" if in_position else "REST",
                    "feedback": feedback,
                    "pose_detected": True,
                    "checks": {
                        "knee_angle":  round(knee_angle, 1),
                        "hip_angle":   round(hip_angle, 1),
                        "torso_angle": round(torso_angle, 1),
                        "knee_ok":     knee_ok,
                        "hip_ok":      hip_ok,
                        "torso_ok":    torso_ok,
                        "best_hold":   round(best_hold_sec, 1),
                    },
                    "landmarks": landmarks,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        save_session(round(total_hold_sec), elapsed, "wallsit")
        await remember_session("default", "wallsit", round(total_hold_sec), elapsed, [])
        landmarker.close()
        print(f"[INFO] Wall sit session ended. Hold: {round(total_hold_sec, 1)}s")
