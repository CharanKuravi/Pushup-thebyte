import { useState } from "react";
import ExerciseCounter from "./components/ExerciseCounter";
import History from "./components/History";
import "./App.css";

type Tab = "workout" | "history";

export type ExerciseType = "pushup" | "squat" | "plank" | "abs" | "wallsit";

export const EXERCISES: { id: ExerciseType; label: string; timed: boolean; cameraHint: string }[] = [
  { id: "pushup",  label: "Push-ups",  timed: false, cameraHint: "Side view, floor level" },
  { id: "squat",   label: "Squats",    timed: false, cameraHint: "Front view, full body visible" },
  { id: "plank",   label: "Plank",     timed: true,  cameraHint: "Side view, floor level" },
  { id: "abs",     label: "Abs",       timed: false, cameraHint: "Side view, lying on floor" },
  { id: "wallsit", label: "Wall Sit",  timed: true,  cameraHint: "Front view, full body visible" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("workout");
  const [exercise, setExercise] = useState<ExerciseType>("pushup");

  return (
    <div className="app">
      <header className="header">
        <h1>💪 AI Workout Tracker</h1>
        <nav>
          <button className={tab === "workout" ? "active" : ""} onClick={() => setTab("workout")}>
            Workout
          </button>
          <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}>
            History
          </button>
        </nav>
      </header>

      {tab === "workout" && (
        <div className="exercise-selector">
          {EXERCISES.map((ex) => (
            <button
              key={ex.id}
              className={`ex-btn ${exercise === ex.id ? "active" : ""}`}
              onClick={() => setExercise(ex.id)}
            >
              <span className="ex-label">{ex.label}</span>
            </button>
          ))}
        </div>
      )}

      <main>
        {tab === "workout"
          ? <ExerciseCounter key={exercise} exercise={exercise} />
          : <History />}
      </main>
    </div>
  );
}
