"""
AI Pushup Counter — Streamlit Web App
Runs in browser at http://localhost:8501
"""

import streamlit as st
import cv2
import numpy as np
import csv
import os
import time
import urllib.request
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def ensure_model():
    if not os.path.isfile(MODEL_PATH):
        with st.spinner("Downloading pose model (~30 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------

R_EAR = 8;  R_SHOULDER = 12; R_ELBOW = 14; R_WRIST = 16; R_HIP = 24; R_ANKLE = 28
L_EAR = 7;  L_SHOULDER = 11; L_ELBOW = 13; L_WRIST = 15; L_HIP = 23; L_ANKLE = 27

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ELBOW_DOWN      = 90
ELBOW_UP        = 160
HIP_SAG_PX      = 50
HEAD_DROP_MARGIN= 30
WRIST_ALIGN_TOL = 80

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    a = np.array([a[0], a[1], 0.0])
    b = np.array([b[0], b[1], 0.0])
    c = np.array([c[0], c[1], 0.0])
    ba, bc = a - b, c - b
    return abs(np.degrees(np.arctan2(np.cross(ba, bc)[2], np.dot(ba, bc))))


def point_to_line_distance(p, la, lb):
    p, la, lb = np.array(p, float), np.array(la, float), np.array(lb, float)
    d = lb - la
    n = np.linalg.norm(d)
    return float(np.linalg.norm(p - la)) if n == 0 else float(abs(np.cross(d, la - p)) / n)


# ---------------------------------------------------------------------------
# Form checks
# ---------------------------------------------------------------------------

def check_hip(shoulder, hip, ankle):
    d = point_to_line_distance(hip, shoulder, ankle)
    return d, d < HIP_SAG_PX

def check_head(ear, shoulder, hip):
    mid_y = (shoulder[1] + hip[1]) / 2.0
    diff  = mid_y - ear[1]
    return diff, diff > -HEAD_DROP_MARGIN

def check_stack(shoulder, wrist):
    diff = abs(shoulder[0] - wrist[0])
    return diff, diff < WRIST_ALIGN_TOL

def get_landmarks(lms, w, h):
    def vis(i): return lms[i].visibility or 0.0
    use_right = (vis(R_SHOULDER)+vis(R_ELBOW)+vis(R_HIP)+vis(R_ANKLE)) >= \
                (vis(L_SHOULDER)+vis(L_ELBOW)+vis(L_HIP)+vis(L_ANKLE))
    ids = (R_EAR,R_SHOULDER,R_ELBOW,R_WRIST,R_HIP,R_ANKLE) if use_right \
        else (L_EAR,L_SHOULDER,L_ELBOW,L_WRIST,L_HIP,L_ANKLE)
    def c(i): return [lms[i].x*w, lms[i].y*h]
    return {k: c(i) for k,i in zip(["ear","shoulder","elbow","wrist","hip","ankle"], ids)}

# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

CONNECTIONS = [
    (R_EAR,R_SHOULDER),(R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),
    (R_SHOULDER,R_HIP),(R_HIP,R_ANKLE),
    (L_EAR,L_SHOULDER),(L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
    (L_SHOULDER,L_HIP),(L_HIP,L_ANKLE),
]

def draw_skeleton(frame, lms, w, h):
    pts = {}
    for i, lm in enumerate(lms):
        pts[i] = (int(lm.x*w), int(lm.y*h))
        cv2.circle(frame, pts[i], 4, (245,117,66), -1)
    for a,b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (245,66,230), 2)

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(frame, reps, state, ea, hip_ok, hip_d, head_ok, stack_ok, feedback):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0,0), (200,80), (0,0,0), -1)
    cv2.putText(frame, "REPS", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(reps), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3, cv2.LINE_AA)

    sc = (0,255,255) if state=="UP" else (0,120,255)
    cv2.rectangle(frame, (w-160,0), (w,80), (0,0,0), -1)
    cv2.putText(frame, "STATE", (w-150,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, state or "---", (w-150,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, sc, 3, cv2.LINE_AA)

    cv2.rectangle(frame, (0,85), (270,210), (25,25,25), -1)
    ea_c = (0,200,0) if ea <= ELBOW_DOWN else (0,200,200)
    cv2.putText(frame, f"Elbow: {int(ea)} deg", (8,108), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ea_c, 1, cv2.LINE_AA)

    def row(label, ok, y):
        cv2.putText(frame, f"{label}: {'OK' if ok else 'FIX'}", (8,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,200,0) if ok else (0,50,220), 1, cv2.LINE_AA)

    row(f"Hip line (d={int(hip_d)}px)", hip_ok,   132)
    row("Head position",               head_ok,  156)
    row("Wrist under shoulder",        stack_ok, 180)

    if feedback:
        fc = (0,220,80) if "Good" in feedback else (0,50,220)
        ts = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = (w - ts[0]) // 2
        cv2.rectangle(frame, (tx-8, h-54), (tx+ts[0]+8, h-18), (0,0,0), -1)
        cv2.putText(frame, feedback, (tx, h-26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_session(reps, duration):
    exists = os.path.isfile("workout_history.csv")
    with open("workout_history.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date","Total_Reps","Duration_Seconds"])
        if not exists:
            w.writeheader()
        w.writerow({"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Total_Reps": reps,
                    "Duration_Seconds": round(duration, 2)})

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Pushup Counter", layout="wide")
st.title("AI Pushup Counter")
st.caption("Place camera at SIDE VIEW, floor level, so your full body is visible.")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Session")
    rep_display   = st.empty()
    state_display = st.empty()
    form_display  = st.empty()
    st.markdown("---")
    start_btn = st.button("Start", type="primary", use_container_width=True)
    stop_btn  = st.button("Stop & Save", use_container_width=True)

with col1:
    frame_window = st.empty()

# Session state
if "running"    not in st.session_state: st.session_state.running    = False
if "rep_count"  not in st.session_state: st.session_state.rep_count  = 0
if "state"      not in st.session_state: st.session_state.state      = None
if "start_time" not in st.session_state: st.session_state.start_time = None
if "landmarker" not in st.session_state: st.session_state.landmarker = None
if "cap"        not in st.session_state: st.session_state.cap        = None

if start_btn:
    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    st.session_state.landmarker  = mp_vision.PoseLandmarker.create_from_options(options)
    st.session_state.cap         = cv2.VideoCapture(0)
    st.session_state.running     = True
    st.session_state.rep_count   = 0
    st.session_state.state       = None
    st.session_state.start_time  = time.time()
    st.session_state.form_failed = False
    st.session_state.fail_reason = ""
    st.session_state.feedback    = ""

if stop_btn and st.session_state.running:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
    if st.session_state.landmarker:
        st.session_state.landmarker.close()
    elapsed = time.time() - (st.session_state.start_time or time.time())
    save_session(st.session_state.rep_count, elapsed)
    st.success(f"Session saved! Reps: {st.session_state.rep_count} | Duration: {round(elapsed,1)}s")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if st.session_state.running:
    cap        = st.session_state.cap
    landmarker = st.session_state.landmarker
    start_time = st.session_state.start_time

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        ea = 0; hip_ok = True; hip_d = 0; head_ok = True; stack_ok = True

        if result.pose_landmarks:
            lms  = result.pose_landmarks[0]
            pts  = get_landmarks(lms, w, h)
            shoulder, elbow, wrist = pts["shoulder"], pts["elbow"], pts["wrist"]
            hip, ankle, ear        = pts["hip"], pts["ankle"], pts["ear"]

            ea                  = calculate_angle(shoulder, elbow, wrist)
            hip_d,  hip_ok      = check_hip(shoulder, hip, ankle)
            _,      head_ok     = check_head(ear, shoulder, hip)
            stack_d, stack_ok   = check_stack(shoulder, wrist)

            # State machine
            if ea <= ELBOW_DOWN and st.session_state.state != "DOWN":
                st.session_state.state       = "DOWN"
                st.session_state.form_failed = False
                st.session_state.fail_reason = ""

            if st.session_state.state == "DOWN":
                if not hip_ok and not st.session_state.form_failed:
                    st.session_state.form_failed = True
                    st.session_state.fail_reason = "Hip sagging — keep body straight"
                elif not head_ok and not st.session_state.form_failed:
                    st.session_state.form_failed = True
                    st.session_state.fail_reason = "Head dropping — keep neck neutral"

            if ea >= ELBOW_UP and st.session_state.state == "DOWN":
                st.session_state.state = "UP"
                if not stack_ok and not st.session_state.form_failed:
                    st.session_state.form_failed = True
                    st.session_state.fail_reason = "Wrist not under shoulder"

                if not st.session_state.form_failed:
                    st.session_state.rep_count += 1
                    st.session_state.feedback   = f"Good rep! #{st.session_state.rep_count}"
                else:
                    st.session_state.feedback = st.session_state.fail_reason

            draw_skeleton(frame, lms, w, h)
            ep = (int(elbow[0])+10, int(elbow[1]))
            cv2.putText(frame, f"{int(ea)}deg", ep,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            cv2.line(frame, (int(shoulder[0]),int(shoulder[1])),
                     (int(ankle[0]),int(ankle[1])), (0,180,255), 1)

        draw_hud(frame, st.session_state.rep_count, st.session_state.state,
                 ea, hip_ok, hip_d, head_ok, stack_ok,
                 st.session_state.get("feedback",""))

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

        rep_display.metric("Reps", st.session_state.rep_count)
        state_display.metric("State", st.session_state.state or "---")

        # Rerun to grab next frame
        time.sleep(0.03)
        st.rerun()
