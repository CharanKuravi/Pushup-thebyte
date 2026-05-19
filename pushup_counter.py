"""
AI-Powered Pushup Counter - v2.0
Camera: Place it at SIDE VIEW (to your right or left), at floor level.

Rep is counted only when ALL of these pass:
  1. Elbow angle (Shoulder->Elbow->Wrist) drops to <= 90 deg  --> DOWN state
  2. Elbow angle rises back to >= 160 deg                     --> UP state + count
  3. Hip stays on the Shoulder-Ankle line (no sag/pike)
  4. Ear stays above the torso line (no head drop)
  5. Wrist is roughly under shoulder at the top position
"""

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
# Model download
# ---------------------------------------------------------------------------

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def ensure_model():
    """Download the MediaPipe pose model if not already present."""
    if not os.path.isfile(MODEL_PATH):
        print("[INFO] Downloading pose landmarker model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose)
# ---------------------------------------------------------------------------

R_EAR      = 8
R_SHOULDER = 12
R_ELBOW    = 14
R_WRIST    = 16
R_HIP      = 24
R_ANKLE    = 28

L_EAR      = 7
L_SHOULDER = 11
L_ELBOW    = 13
L_WRIST    = 15
L_HIP      = 23
L_ANKLE    = 27

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ELBOW_DOWN       = 90    # elbow must bend to <= this for DOWN state
ELBOW_UP         = 160   # elbow must extend to >= this for UP state
HIP_SAG_PX       = 50    # max perpendicular distance (px) of hip from shoulder-ankle line
HEAD_DROP_MARGIN = 30    # ear must stay within this many px above torso midpoint
WRIST_ALIGN_TOL  = 80    # wrist X must be within this many px of shoulder X at top

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    """
    Angle at vertex B formed by points A-B-C.
    Uses 3D vectors to avoid NumPy 2D deprecation warning.
    Returns degrees (0-180).
    """
    a = np.array([a[0], a[1], 0.0])
    b = np.array([b[0], b[1], 0.0])
    c = np.array([c[0], c[1], 0.0])
    ba = a - b
    bc = c - b
    cross = np.cross(ba, bc)[2]
    dot   = np.dot(ba, bc)
    return abs(np.degrees(np.arctan2(cross, dot)))


def point_to_line_distance(p, line_a, line_b):
    """
    Perpendicular distance from point p to the infinite line through line_a and line_b.
    Used to detect hip sag or pike.
    """
    p      = np.array(p,      dtype=float)
    line_a = np.array(line_a, dtype=float)
    line_b = np.array(line_b, dtype=float)
    d    = line_b - line_a
    norm = np.linalg.norm(d)
    if norm == 0:
        return float(np.linalg.norm(p - line_a))
    return float(abs(np.cross(d, line_a - p)) / norm)


# ---------------------------------------------------------------------------
# Form checks
# ---------------------------------------------------------------------------

def check_hip_alignment(shoulder, hip, ankle):
    """
    Hip must stay close to the shoulder-ankle line (straight plank).
    Returns (distance_px, is_ok).
    """
    dist = point_to_line_distance(hip, shoulder, ankle)
    return dist, dist < HIP_SAG_PX


def check_head_position(ear, shoulder, hip):
    """
    Ear must stay above (lower Y) the midpoint of shoulder and hip.
    Image Y increases downward, so ear_y < mid_y is good.
    Returns (diff_px, is_ok).
    """
    mid_y = (shoulder[1] + hip[1]) / 2.0
    diff  = mid_y - ear[1]   # positive means ear is above midpoint
    return diff, diff > -HEAD_DROP_MARGIN


def check_wrist_stack(shoulder, wrist):
    """
    At the top of the pushup the wrist X should be close to shoulder X (side view).
    Returns (diff_px, is_ok).
    """
    diff = abs(shoulder[0] - wrist[0])
    return diff, diff < WRIST_ALIGN_TOL


# ---------------------------------------------------------------------------
# Side selection — use whichever side has better visibility
# ---------------------------------------------------------------------------

def get_side_landmarks(lms, w, h):
    """
    Picks left or right side based on total landmark visibility score.
    Returns a dict with pixel coordinates for each key joint.
    """
    def vis(idx):
        return lms[idx].visibility if lms[idx].visibility is not None else 0.0

    right_score = vis(R_SHOULDER) + vis(R_ELBOW) + vis(R_HIP) + vis(R_ANKLE)
    left_score  = vis(L_SHOULDER) + vis(L_ELBOW) + vis(L_HIP) + vis(L_ANKLE)
    use_right   = right_score >= left_score

    if use_right:
        ids = (R_EAR, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_ANKLE)
    else:
        ids = (L_EAR, L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_ANKLE)

    def c(idx):
        lm = lms[idx]
        return [lm.x * w, lm.y * h]

    return {
        "ear":      c(ids[0]),
        "shoulder": c(ids[1]),
        "elbow":    c(ids[2]),
        "wrist":    c(ids[3]),
        "hip":      c(ids[4]),
        "ankle":    c(ids[5]),
    }


# ---------------------------------------------------------------------------
# CSV history
# ---------------------------------------------------------------------------

HISTORY_FILE = "workout_history.csv"
CSV_HEADERS  = ["Date", "Total_Reps", "Duration_Seconds"]


def save_session(total_reps, duration_seconds):
    """Append session data to workout_history.csv, creating header if needed."""
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Date":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Reps":       total_reps,
            "Duration_Seconds": round(duration_seconds, 2),
        })
    print(f"\n[OK] Session saved to {HISTORY_FILE}")
    print(f"     Reps: {total_reps}  |  Duration: {round(duration_seconds, 2)}s")


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------

def draw_hud(frame, rep_count, state, checks, feedback):
    """
    Draws rep count, state, form indicators, and feedback on the frame.
    checks keys: elbow_angle, hip_ok, hip_dist, head_ok, stack_ok, stack_diff
    """
    h, w = frame.shape[:2]

    # Rep counter — top left
    cv2.rectangle(frame, (0, 0), (200, 80), (0, 0, 0), -1)
    cv2.putText(frame, "REPS", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, str(rep_count), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)

    # State — top right
    sc = (0, 255, 255) if state == "UP" else (0, 120, 255)
    cv2.rectangle(frame, (w - 160, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, "STATE", (w - 150, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, state if state else "---", (w - 150, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, sc, 3, cv2.LINE_AA)

    # Form panel — below rep counter
    cv2.rectangle(frame, (0, 85), (270, 210), (25, 25, 25), -1)

    elbow_angle = checks.get("elbow_angle", 0)
    hip_ok      = checks.get("hip_ok",   True)
    hip_dist    = checks.get("hip_dist",  0)
    head_ok     = checks.get("head_ok",  True)
    stack_ok    = checks.get("stack_ok", True)

    # Elbow angle — yellow when bending, green when fully down
    ea_color = (0, 200, 0) if elbow_angle <= ELBOW_DOWN else (0, 200, 200)
    cv2.putText(frame, f"Elbow: {int(elbow_angle)} deg", (8, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ea_color, 1, cv2.LINE_AA)

    def form_row(label, ok, y):
        color = (0, 200, 0) if ok else (0, 50, 220)
        cv2.putText(frame, f"{label}: {'OK' if ok else 'FIX'}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    form_row(f"Hip line (d={int(hip_dist)}px)", hip_ok,   132)
    form_row("Head position",                   head_ok,  156)
    form_row("Wrist under shoulder",            stack_ok, 180)

    # Feedback message — bottom centre
    if feedback:
        is_good  = "Good" in feedback
        fc       = (0, 220, 80) if is_good else (0, 50, 220)
        ts       = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx       = (w - ts[0]) // 2
        cv2.rectangle(frame, (tx - 8, h - 54), (tx + ts[0] + 8, h - 18), (0, 0, 0), -1)
        cv2.putText(frame, feedback, (tx, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Side view  |  Press Q to quit", (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

CONNECTIONS = [
    (R_EAR, R_SHOULDER),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (R_SHOULDER, R_HIP),
    (R_HIP, R_ANKLE),
    (L_EAR, L_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (L_SHOULDER, L_HIP),
    (L_HIP, L_ANKLE),
]


def draw_skeleton(frame, lms, w, h):
    """Draw pose skeleton dots and lines on the frame."""
    pts = {}
    for i, lm in enumerate(lms):
        pts[i] = (int(lm.x * w), int(lm.y * h))
        cv2.circle(frame, pts[i], 4, (245, 117, 66), -1)
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (245, 66, 230), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_model()

    # Build MediaPipe pose landmarker (VIDEO mode for per-frame timestamps)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam at index 0.")
        landmarker.close()
        return

    print("[INFO] Camera opened.")
    print("[INFO] Place camera at SIDE VIEW, at floor level, to see your full body.")
    print(f"[INFO] DOWN: elbow <= {ELBOW_DOWN} deg  |  UP: elbow >= {ELBOW_UP} deg")
    print("[INFO] Press Q to quit.\n")

    rep_count        = 0
    movement_state   = None    # None -> "DOWN" -> "UP"
    form_failed      = False   # tracks if form broke during current rep
    form_fail_reason = ""
    start_time       = time.time()
    feedback         = ""
    checks           = {
        "elbow_angle": 0, "hip_ok": True, "hip_dist": 0,
        "head_ok": True,  "stack_ok": True, "stack_diff": 0,
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame.")
            break

        # Mirror the frame so it feels natural
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Run pose detection
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            pts = get_side_landmarks(lms, w, h)

            shoulder = pts["shoulder"]
            elbow    = pts["elbow"]
            wrist    = pts["wrist"]
            hip      = pts["hip"]
            ankle    = pts["ankle"]
            ear      = pts["ear"]

            # Compute all measurements
            elbow_angle             = calculate_angle(shoulder, elbow, wrist)
            hip_dist,  hip_ok       = check_hip_alignment(shoulder, hip, ankle)
            head_diff, head_ok      = check_head_position(ear, shoulder, hip)
            stack_diff, stack_ok    = check_wrist_stack(shoulder, wrist)

            checks = {
                "elbow_angle": elbow_angle,
                "hip_ok":      hip_ok,
                "hip_dist":    hip_dist,
                "head_ok":     head_ok,
                "stack_ok":    stack_ok,
                "stack_diff":  stack_diff,
            }

            # ----------------------------------------------------------------
            # State machine
            # ----------------------------------------------------------------

            # 1. Entering DOWN — elbow bends past threshold
            if elbow_angle <= ELBOW_DOWN and movement_state != "DOWN":
                movement_state   = "DOWN"
                form_failed      = False
                form_fail_reason = ""

            # 2. While DOWN — continuously check form
            if movement_state == "DOWN":
                if not hip_ok and not form_failed:
                    form_failed      = True
                    form_fail_reason = "Hip sagging — keep body straight"
                elif not head_ok and not form_failed:
                    form_failed      = True
                    form_fail_reason = "Head dropping — keep neck neutral"

            # 3. Entering UP — elbow extends back past threshold
            if elbow_angle >= ELBOW_UP and movement_state == "DOWN":
                movement_state = "UP"

                # Check wrist stack at the top
                if not stack_ok and not form_failed:
                    form_failed      = True
                    form_fail_reason = "Wrist not under shoulder"

                if not form_failed:
                    rep_count += 1
                    feedback   = f"Good rep! #{rep_count}"
                    print(f"  [OK] Rep #{rep_count}  "
                          f"elbow={elbow_angle:.1f}deg  "
                          f"hip_dist={hip_dist:.1f}px  "
                          f"stack={stack_diff:.1f}px")
                else:
                    feedback = form_fail_reason
                    print(f"  [SKIP] {form_fail_reason}")

            # ----------------------------------------------------------------
            # Draw skeleton and measurements
            # ----------------------------------------------------------------
            draw_skeleton(frame, lms, w, h)

            # Elbow angle label near elbow joint
            ep = (int(elbow[0]) + 10, int(elbow[1]))
            cv2.putText(frame, f"{int(elbow_angle)}deg", ep,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Shoulder-to-ankle alignment reference line
            cv2.line(frame,
                     (int(shoulder[0]), int(shoulder[1])),
                     (int(ankle[0]),    int(ankle[1])),
                     (0, 180, 255), 1)

        # Draw HUD and show frame
        draw_hud(frame, rep_count, movement_state, checks, feedback)
        cv2.imshow("AI Pushup Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    elapsed = time.time() - start_time
    save_session(rep_count, elapsed)


if __name__ == "__main__":
    main()
