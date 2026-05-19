import { useRef, useEffect, useState, useCallback } from "react";
import { EXERCISES, type ExerciseType } from "../App";

const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const FRAME_INTERVAL_MS = 60;

// WebSocket route per exercise
const WS_ROUTES: Record<ExerciseType, string> = {
  pushup:  "/ws/pushup",
  squat:   "/ws/squat",
  plank:   "/ws/plank",
  abs:     "/ws/abs",
  wallsit: "/ws/wallsit",
};

interface Checks {
  // pushup
  elbow_angle?: number;
  body_line?: number;
  shoulder_ang?: number;
  wrist_ang?: number;
  hip_ok?: boolean;
  flare_ok?: boolean;
  wrist_ok?: boolean;
  // squat
  knee_angle?: number;
  hip_angle?: number;
  torso_angle?: number;
  ankle_angle?: number;
  torso_ok?: boolean;
  ankle_ok?: boolean;
  valgus_ok?: boolean;
  // plank
  body_line?: number;
  neck_line?: number;
  body_ok?: boolean;
  neck_ok?: boolean;
  elbow_ok?: boolean;
  // abs
  hip_angle_abs?: number;
  neck_angle?: number;
  neck_ok?: boolean;
  // wallsit
  knee_ok?: boolean;
  hip_ok?: boolean;
  torso_ok?: boolean;
  best_hold?: number;
}

interface PoseData {
  reps: number;          // for timed exercises this is hold seconds
  state: string | null;
  feedback: string;
  pose_detected: boolean;
  checks: Checks;
}

interface Props {
  exercise: ExerciseType;
}

export default function ExerciseCounter({ exercise }: Props) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const wsRef      = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [running, setRunning]   = useState(false);
  const [poseData, setPoseData] = useState<PoseData | null>(null);
  const [error, setError]       = useState("");
  const [coaching, setCoaching] = useState("");

  const meta = EXERCISES.find((e) => e.id === exercise)!;
  const wsUrl = BACKEND.replace("https://", "wss://").replace("http://", "ws://") + WS_ROUTES[exercise];

  const stopSession = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setRunning(false);
  }, []);

  const startSession = useCallback(async () => {
    setError("");
    setPoseData(null);
    setCoaching("");
    try {
      // Fetch personalized coaching from Hindsight memory before starting
      fetch(`${BACKEND}/coach/default/${exercise}?reps=0`)
        .then((r) => r.json())
        .then((d) => { if (d.message) setCoaching(d.message); })
        .catch(() => {});

      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        const data: PoseData = JSON.parse(e.data);
        setPoseData(data);
      };
      ws.onerror = () => setError("Cannot connect to backend. Make sure the server is running.");
      ws.onclose = () => setRunning(false);

      ws.onopen = () => {
        setRunning(true);
        intervalRef.current = window.setInterval(() => {
          if (!canvasRef.current || !videoRef.current || ws.readyState !== WebSocket.OPEN) return;
          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) return;
          canvasRef.current.width  = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0);
          canvasRef.current.toBlob((blob) => {
            if (!blob) return;
            blob.arrayBuffer().then((buf) => {
              const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
              ws.send(b64);
            });
          }, "image/jpeg", 0.7);
        }, FRAME_INTERVAL_MS);
      };
    } catch {
      setError("Camera access denied. Please allow camera permissions.");
    }
  }, [wsUrl]);

  useEffect(() => () => stopSession(), [stopSession]);

  const checks = poseData?.checks;

  // Label for the main counter — reps for rep-based, seconds for timed
  const counterLabel = meta.timed ? "HOLD TIME" : "REPS";
  const counterValue = meta.timed
    ? `${poseData?.reps ?? 0}s`
    : (poseData?.reps ?? 0);

  return (
    <div className="counter-page">
      <div className="video-wrapper">
        <video ref={videoRef} className="video-feed" muted playsInline />
        <canvas ref={canvasRef} style={{ display: "none" }} />
        {!running && (
          <div className="video-overlay">
            <p><strong>{meta.label}</strong></p>
            <p style={{ marginTop: 8, fontSize: "0.85rem" }}>{meta.cameraHint}</p>
          </div>
        )}
      </div>

      <div className="stats-panel">
        {/* Main counter */}
        <div className="rep-box">
          <span className="rep-label">{counterLabel}</span>
          <span className="rep-count">{counterValue}</span>
        </div>

        {/* State badge */}
        <div className={`state-box ${poseData?.state === "UP" || poseData?.state === "HOLD" ? "up" : "down"}`}>
          <span className="state-label">STATE</span>
          <span className="state-value">{poseData?.state ?? "---"}</span>
        </div>

        {/* Form checks — rendered per exercise */}
        {checks && Object.keys(checks).length > 0 && (
          <div className="form-checks">
            <h3>FORM</h3>
            <FormChecks exercise={exercise} checks={checks} />
          </div>
        )}

        {/* Feedback */}
        {poseData?.feedback && (
          <div className={`feedback ${poseData.feedback.startsWith("Rep") || poseData.feedback.startsWith("Hold") ? "good" : "bad"}`}>
            {poseData.feedback}
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {coaching && (
          <div className="coaching-panel">
            <span className="coaching-label">Coach</span>
            <p>{coaching}</p>
          </div>
        )}

        <button
          className={`btn ${running ? "btn-stop" : "btn-start"}`}
          onClick={running ? stopSession : startSession}
        >
          {running ? "Stop Session" : "Start Session"}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-exercise form check rows
// ---------------------------------------------------------------------------

function FormChecks({ exercise, checks }: { exercise: ExerciseType; checks: Checks }) {
  switch (exercise) {
    case "pushup":
      return (
        <>
          <div className="check-row">
            <span>Elbow</span>
            <span className="check-value">{checks.elbow_angle}°</span>
          </div>
          <div className="check-row">
            <span>Body line</span>
            <span className="check-value">{checks.body_line}°</span>
          </div>
          <div className="check-row">
            <span>Shoulder flare</span>
            <span className="check-value">{checks.shoulder_ang}°</span>
          </div>
          <div className="check-row">
            <span>Wrist</span>
            <span className="check-value">{checks.wrist_ang}°</span>
          </div>
          <FormRow label="Body straight" ok={checks.hip_ok    ?? true} />
          <FormRow label="Elbow tuck"    ok={checks.flare_ok  ?? true} />
          <FormRow label="Wrist straight" ok={checks.wrist_ok ?? true} />
        </>
      );

    case "squat":
      return (
        <>
          <div className="check-row">
            <span>Knee</span>
            <span className="check-value">{checks.knee_angle}°</span>
          </div>
          <div className="check-row">
            <span>Hip</span>
            <span className="check-value">{checks.hip_angle}°</span>
          </div>
          <div className="check-row">
            <span>Torso lean</span>
            <span className="check-value">{checks.torso_angle}°</span>
          </div>
          <div className="check-row">
            <span>Ankle flex</span>
            <span className="check-value">{checks.ankle_angle}°</span>
          </div>
          <FormRow label="Hip depth"     ok={checks.hip_ok    ?? true} />
          <FormRow label="Torso upright" ok={checks.torso_ok  ?? true} />
          <FormRow label="Ankle flex"    ok={checks.ankle_ok  ?? true} />
          <FormRow label="Knee tracking" ok={checks.valgus_ok ?? true} />
        </>
      );

    case "plank":
      return (
        <>
          <div className="check-row">
            <span>Body line</span>
            <span className="check-value">{checks.body_line}°</span>
          </div>
          <div className="check-row">
            <span>Neck line</span>
            <span className="check-value">{checks.neck_line}°</span>
          </div>
          <div className="check-row">
            <span>Elbow angle</span>
            <span className="check-value">{checks.elbow_angle}°</span>
          </div>
          <FormRow label="Body straight" ok={checks.body_ok  ?? true} />
          <FormRow label="Neck neutral"  ok={checks.neck_ok  ?? true} />
          <FormRow label="Elbow position" ok={checks.elbow_ok ?? true} />
        </>
      );

    case "abs":
      return (
        <>
          <div className="check-row">
            <span>Hip angle</span>
            <span className="check-value">{checks.hip_angle}°</span>
          </div>
          <div className="check-row">
            <span>Neck angle</span>
            <span className="check-value">{checks.neck_angle}°</span>
          </div>
          <FormRow label="Neck neutral" ok={checks.neck_ok ?? true} />
        </>
      );

    case "wallsit":
      return (
        <>
          <div className="check-row">
            <span>Knee</span>
            <span className="check-value">{checks.knee_angle}°</span>
          </div>
          <div className="check-row">
            <span>Hip</span>
            <span className="check-value">{checks.hip_angle}°</span>
          </div>
          <div className="check-row">
            <span>Torso</span>
            <span className="check-value">{checks.torso_angle}°</span>
          </div>
          <FormRow label="Knee at 90°"    ok={checks.knee_ok  ?? true} />
          <FormRow label="Hip position"   ok={checks.hip_ok   ?? true} />
          <FormRow label="Back on wall"   ok={checks.torso_ok ?? true} />
          {checks.best_hold !== undefined && (
            <div className="check-row">
              <span>Best hold</span>
              <span className="check-value">{checks.best_hold}s</span>
            </div>
          )}
        </>
      );
  }
}

function FormRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="check-row">
      <span>{label}</span>
      <span className={`check-badge ${ok ? "ok" : "fix"}`}>{ok ? "✓ OK" : "✗ FIX"}</span>
    </div>
  );
}
