import { useRef, useEffect, useState, useCallback } from "react";

const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const WS_URL = BACKEND.replace("https://", "wss://").replace("http://", "ws://") + "/ws/pushup";
const FRAME_INTERVAL_MS = 100; // send 10 frames/sec to backend

interface Checks {
  elbow_angle: number;
  hip_ok: boolean;
  hip_dist: number;
  head_ok: boolean;
  stack_ok: boolean;
  stack_diff: number;
}

interface PoseData {
  reps: number;
  state: string | null;
  feedback: string;
  pose_detected: boolean;
  checks: Checks;
}

export default function PushupCounter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [poseData, setPoseData] = useState<PoseData | null>(null);
  const [error, setError] = useState("");

  const stopSession = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setRunning(false);
  }, []);

  const startSession = useCallback(async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        const data: PoseData = JSON.parse(e.data);
        setPoseData(data);
      };

      ws.onerror = () => setError("Cannot connect to backend. Make sure the server is running.");
      ws.onclose = () => setRunning(false);

      ws.onopen = () => {
        setRunning(true);
        // Send frames at fixed interval
        intervalRef.current = window.setInterval(() => {
          if (!canvasRef.current || !videoRef.current || ws.readyState !== WebSocket.OPEN) return;
          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) return;
          canvasRef.current.width = videoRef.current.videoWidth;
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
  }, []);

  useEffect(() => () => stopSession(), [stopSession]);

  const checks = poseData?.checks;

  return (
    <div className="counter-page">
      <div className="video-wrapper">
        <video ref={videoRef} className="video-feed" muted playsInline />
        <canvas ref={canvasRef} style={{ display: "none" }} />
        {!running && (
          <div className="video-overlay">
            <p>Position camera at <strong>side view</strong>, floor level</p>
          </div>
        )}
      </div>

      <div className="stats-panel">
        <div className="rep-box">
          <span className="rep-label">REPS</span>
          <span className="rep-count">{poseData?.reps ?? 0}</span>
        </div>

        <div className={`state-box ${poseData?.state === "UP" ? "up" : "down"}`}>
          <span className="state-label">STATE</span>
          <span className="state-value">{poseData?.state ?? "---"}</span>
        </div>

        {checks && (
          <div className="form-checks">
            <h3>Form</h3>
            <div className="check-row">
              <span>Elbow angle</span>
              <span className="check-value">{checks.elbow_angle}°</span>
            </div>
            <FormRow label="Hip alignment" ok={checks.hip_ok} />
            <FormRow label="Head position" ok={checks.head_ok} />
            <FormRow label="Wrist stack" ok={checks.stack_ok} />
          </div>
        )}

        {poseData?.feedback && (
          <div className={`feedback ${poseData.feedback.includes("Good") ? "good" : "bad"}`}>
            {poseData.feedback}
          </div>
        )}

        {error && <div className="error">{error}</div>}

        <button className={`btn ${running ? "btn-stop" : "btn-start"}`}
          onClick={running ? stopSession : startSession}>
          {running ? "Stop Session" : "Start Session"}
        </button>
      </div>
    </div>
  );
}

function FormRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="check-row">
      <span>{label}</span>
      <span className={`check-badge ${ok ? "ok" : "fix"}`}>{ok ? "✓ OK" : "✗ FIX"}</span>
    </div>
  );
}
