import { useEffect, useState } from "react";

const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const EXERCISE_ORDER = ["pushup", "squat", "plank", "abs", "wallsit"];

const EXERCISE_LABELS: Record<string, string> = {
  pushup:  "Push-ups",
  squat:   "Squats",
  plank:   "Plank",
  abs:     "Abs",
  wallsit: "Wall Sit",
};

const TIMED = new Set(["plank", "wallsit"]);

interface Session {
  Date: string;
  Exercise: string;
  Total_Reps: string;
  Duration_Seconds: string;
}

export default function History() {
  const [sessions, setSessions]   = useState<Session[]>([]);
  const [loading, setLoading]     = useState(true);
  const [expanded, setExpanded]   = useState<Record<string, boolean>>({});

  const fetchHistory = () => {
    fetch(`${BACKEND}/history`)
      .then((r) => r.json())
      .then((data: Session[]) => {
        setSessions(data);
        setLoading(false);
        // default all groups open
        const init: Record<string, boolean> = {};
        data.forEach((s) => { init[s.Exercise || "pushup"] = true; });
        setExpanded(init);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => { fetchHistory(); }, []);

  const deleteSession = (date: string) => {
    if (!window.confirm(`Delete session from ${date}?`)) return;
    fetch(`${BACKEND}/history/${encodeURIComponent(date)}`, { method: "DELETE" })
      .then(() => fetchHistory());
  };

  const toggle = (ex: string) =>
    setExpanded((prev) => ({ ...prev, [ex]: !prev[ex] }));

  if (loading) return <p className="loading">Loading history...</p>;
  if (!sessions.length) return <p className="loading">No sessions yet. Start a workout!</p>;

  // Group sessions by exercise
  const grouped: Record<string, Session[]> = {};
  sessions.forEach((s) => {
    const ex = s.Exercise || "pushup";
    if (!grouped[ex]) grouped[ex] = [];
    grouped[ex].push(s);
  });

  // Sort groups by predefined order
  const groupKeys = EXERCISE_ORDER.filter((ex) => grouped[ex]);

  return (
    <div className="history-page">

      {/* Top summary row */}
      <div className="history-summary">
        <div className="summary-card">
          <span>{sessions.length}</span>
          <label>Total Sessions</label>
        </div>
        {groupKeys.map((ex) => {
          const rows   = grouped[ex];
          const isTimed = TIMED.has(ex);
          const total  = rows.reduce((s, r) => s + parseInt(r.Total_Reps), 0);
          return (
            <div key={ex} className="summary-card">
              <span>{total}{isTimed ? "s" : ""}</span>
              <label>{EXERCISE_LABELS[ex]}</label>
            </div>
          );
        })}
      </div>

      {/* One collapsible section per exercise */}
      {groupKeys.map((ex) => {
        const rows    = grouped[ex];
        const isTimed = TIMED.has(ex);
        const isOpen  = expanded[ex] ?? true;
        const total   = rows.reduce((s, r) => s + parseInt(r.Total_Reps), 0);
        const best    = Math.max(...rows.map((r) => parseInt(r.Total_Reps)));

        return (
          <div key={ex} className="history-group">
            <button className="group-header" onClick={() => toggle(ex)}>
              <div className="group-header-left">
                <span className="group-title">{EXERCISE_LABELS[ex]}</span>
                <span className="group-meta">
                  {rows.length} session{rows.length !== 1 ? "s" : ""}
                  &nbsp;&nbsp;
                  Total: {total}{isTimed ? "s" : " reps"}
                  &nbsp;&nbsp;
                  Best: {best}{isTimed ? "s" : " reps"}
                </span>
              </div>
              <span className="group-chevron">{isOpen ? "▲" : "▼"}</span>
            </button>

            {isOpen && (
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>{isTimed ? "Hold" : "Reps"}</th>
                    <th>Duration</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows].reverse().map((s, i) => (
                    <tr key={i}>
                      <td>{s.Date}</td>
                      <td>{s.Total_Reps}{isTimed ? "s" : ""}</td>
                      <td>{parseFloat(s.Duration_Seconds).toFixed(1)}s</td>
                      <td>
                        <button
                          className="delete-btn"
                          onClick={() => deleteSession(s.Date)}
                        >
                          delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        );
      })}
    </div>
  );
}
