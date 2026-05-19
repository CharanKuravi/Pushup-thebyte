# AI Workout Tracker

Real-time exercise tracking using MediaPipe pose estimation and OpenCV. Tracks reps and hold time for five exercises through a webcam feed. Powered by Hindsight for persistent agent memory and cascadeflow for intelligent LLM routing.

---

## Exercises

| Exercise  | Type       | Camera Position                        |
|-----------|------------|----------------------------------------|
| Push-ups  | Rep count  | Side view, hip height, 1-1.5m away     |
| Squats    | Rep count  | Front view, full body visible          |
| Plank     | Timed hold | Side view, floor level                 |
| Abs       | Rep count  | Side view, lying on floor              |
| Wall Sit  | Timed hold | Front view, full body visible          |

---

## How It Works

The frontend captures webcam frames at ~16fps, encodes them as base64 JPEG, and sends them over a WebSocket to the backend. The backend runs MediaPipe pose landmarker on each frame, computes joint angles using the arccos dot product formula, and streams back rep count, state, form checks, and feedback in real time.

The core angle formula used across all exercises:

```python
def angle_arccos(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
```

`b` is always the vertex joint. Three points, one angle. Every exercise uses this same function with different landmark combinations.

---

## Exercise Math

### Push-ups

Camera: side view, hip height, 1-1.5m away.

| Joint | Points | Threshold |
|-------|--------|-----------|
| Elbow (primary) | Shoulder -> Elbow -> Wrist | down below 95°, up above 155° |
| Body line | Shoulder -> Hip -> Ankle | 165°-185° |
| Shoulder flare | Elbow -> Shoulder -> Hip | below 75° |
| Wrist alignment | Elbow -> Wrist -> Index finger | above 160° |

State machine: `up -> down -> up = 1 rep`. 5-frame smoothing on elbow angle. 10-frame lockout after each rep. Visibility gate on elbow and hip — if the camera cannot see the body, the state machine is disabled.

### Squats

Camera: front view, full body visible.

| Joint | Points | Threshold |
|-------|--------|-----------|
| Knee (primary) | Hip -> Knee -> Ankle | down below 95°, up above 155° |
| Hip depth | Shoulder -> Hip -> Knee | must drop below 95° during descent |
| Torso lean | Vertical -> Hip -> Shoulder | above 60° |
| Ankle dorsiflexion | Knee -> Ankle -> Foot index | below 88° |
| Valgus check | Knee width vs hip width | knee width >= 85% of hip width |

State machine: `up -> down -> up = 1 rep`. 5-frame rolling average on knee angle. Hip depth tracked across the entire down phase, not just at the transition. Form issues shown as feedback alongside the rep count but never block it.

### Plank

Camera: side view, floor level.

| Angle | Points | Good range |
|-------|--------|------------|
| Body line (primary) | Shoulder -> Hip -> Ankle | 160°-195° |
| Neck line | Ear -> Shoulder -> Hip | 160°-195° |
| Elbow angle | Shoulder -> Elbow -> Wrist | 75°-105° |

Timer accumulates only when all three angles are in range simultaneously. Pauses the instant any joint goes out of range and resumes when form is corrected. 3-frame smoothing on body line. Visibility gate on hip and ankle.

### Abs / Crunches

Camera: side view, lying on floor.

| Angle | Points | Flat | Crunched |
|-------|--------|------|----------|
| Hip flexion (primary) | Shoulder -> Hip -> Knee | above 150° | below 85° |
| Neck | Ear -> Shoulder -> Hip | above 140° | above 140° |

State machine: `down (flat) -> up (crunched) -> down = 1 rep`. 3-frame smoothing. 6-frame lockout. Minimum 45° swing required between flat baseline and crunched peak — prevents head bobs from counting. Visibility gate on hip and knee — if the camera only sees the head, the state machine is disabled entirely.

### Wall Sit

Camera: front view, full body visible.

| Angle | Points | Good range |
|-------|--------|------------|
| Knee (primary) | Hip -> Knee -> Ankle | 80°-100° |
| Hip | Shoulder -> Hip -> Knee | 80°-100° |
| Torso vs vertical | Vertical -> Hip -> Shoulder | above 80° |

All three joints must be in range simultaneously. One joint going out of range pauses the clock instantly. Best hold tracked separately. 5-frame smoothing on knee angle.

---

## Hindsight — Persistent Agent Memory

Without memory, the coaching is generic every session. With Hindsight, the agent remembers every session — reps, duration, form issues — and uses that history to give specific coaching.

Every session end stores a memory:

```python
hs.retain(
    bank_id="workout-tracker",
    content="User completed squats on 2026-05-19. Reps: 19. Duration: 195s. Form issues: knees caving in.",
    metadata={"user_id": "default", "exercise": "squat"},
    tags=["default", "squat"],
)
```

Every session start recalls past sessions:

```python
result = hs.recall(
    bank_id="workout-tracker",
    query="workout history for user default doing squat",
    tags=["default", "squat"],
    tags_match="all",
    budget="mid",
)
```

That recalled context gets injected into the coaching prompt. The agent goes from:

Session 1 — no history:
```
"Drive through your heels on the way up."
```

Session 5 — Hindsight recalled knees caving in across 3 sessions:
```
"Your knees have been caving in consistently — focus on pushing them out at the bottom of the squat."
```

That delta is the entire point. Session 1 is generic. Session 5 is personal.

Get your Hindsight API key at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io). Use promo code `MEMHACK515` for $50 free credits.

---

## cascadeflow — Runtime Intelligence

Every coaching tip requires an LLM call. Without cascadeflow, every call goes to the same model regardless of complexity. That wastes money on simple responses and under-serves complex ones.

cascadeflow sits between the app and the LLMs and routes each call to the right model:

```
Your code
    |
    v
CascadeAgent  <-- cascadeflow
    |
    |-- simple task --> llama-3.1-8b-instant   (fast, $0.0001/1K tokens)
    |
    |-- complex task -> llama-3.3-70b-versatile (better, $0.0008/1K tokens)
```

The routing decision is driven by `complexity_hint`:

```python
# No history, no recurring issues — stays on cheap model
complexity_hint = "simple"   # -> llama-3.1-8b-instant

# Recurring form issues OR Hindsight found past sessions — escalates
complexity_hint = "complex"  # -> llama-3.3-70b-versatile
```

The escalation trigger in code:

```python
recurring = [issue for issue, count in self.issues.items() if count >= 3]
complex_analysis = bool(recurring or self.history_context)
```

If the same form issue appears 3 or more times in a session, or if Hindsight found past history, cascadeflow escalates to the better model automatically.

Session 1 — no history, no issues:
```
[cascadeflow] model=llama-3.1-8b-instant cost=$0.00001 latency=380ms complexity=simple
```

Session 5 — Hindsight recalled knees caving in:
```
[cascadeflow] model=llama-3.3-70b-versatile cost=$0.00008 latency=1100ms complexity=complex
```

Budget is hard-capped at $0.05 per session:

```python
cascadeflow.init(mode="enforce", budget=0.05)
```

If the budget is hit, cascadeflow stops escalating and stays on the cheap model. It does not crash — it degrades gracefully. Every routing decision is logged automatically with model used, cost, and latency. That is the full audit trail.

The cost difference between simple and complex calls is 8x. Over 50 sessions, that adds up.

cascadeflow is open source and free. Install: `pip install cascadeflow`. Docs at [docs.cascadeflow.ai](https://docs.cascadeflow.ai).

---

## Stack

**Backend**
- Python 3.10+
- FastAPI — WebSocket server and REST history API
- MediaPipe — pose landmarker (full model, float16)
- OpenCV — frame decoding
- NumPy — angle math
- Hindsight (`hindsight-client`) — persistent agent memory across sessions
- cascadeflow — LLM routing, budget enforcement, audit trail
- Groq — LLM provider (free tier, `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`)

**Frontend**
- React 19 + TypeScript
- Vite
- Native WebSocket API

---

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app, all 5 WebSocket endpoints, coaching REST endpoint
│   ├── coach.py             # Hindsight memory + cascadeflow routing layer
│   ├── requirements.txt
│   ├── .env                 # API keys (not committed)
│   └── .env.example         # API key template
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Exercise selector, tab navigation
│   │   ├── App.css
│   │   └── components/
│   │       ├── ExerciseCounter.tsx   # Webcam capture, WS client, form display, coaching panel
│   │       └── History.tsx           # Session history grouped by exercise
│   └── package.json
├── workout_history.csv      # Auto-created on first session save
└── README.md
```

---

## Running Locally

**1. Set up API keys**

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```
HINDSIGHT_API_KEY=your_key_here
HINDSIGHT_PIPELINE_ID=workout-tracker
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
GROQ_API_KEY=your_key_here
```

Get Hindsight key at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) — use promo code `MEMHACK515` for $50 free.

Get Groq key at [groq.com](https://groq.com) — free tier, no credit card needed.

The app works without keys — coaching falls back to rule-based tips. Hindsight and cascadeflow activate automatically when keys are present.

**2. Backend**

```bash
cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The pose model (~30 MB) downloads automatically on first run.

**3. Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

**4. Verify integrations are live**

Do one exercise session and stop it. Check the terminal:

```
[Hindsight] connected.
[cascadeflow] CascadeAgent ready. Tier 1: llama-3.1-8b-instant, Tier 2: llama-3.3-70b-versatile
[Hindsight] retained session for default/squat
[cascadeflow] model=llama-3.1-8b-instant cost=$0.00001 latency=528ms complexity=simple
```

If you see those lines, both integrations are live.

---

## API

### WebSocket endpoints

| Endpoint | Exercise |
|----------|----------|
| `ws://localhost:8000/ws/pushup` | Push-ups |
| `ws://localhost:8000/ws/squat` | Squats |
| `ws://localhost:8000/ws/plank` | Plank |
| `ws://localhost:8000/ws/abs` | Abs |
| `ws://localhost:8000/ws/wallsit` | Wall Sit |

**Send:** base64-encoded JPEG frame as a text message.

**Receive:** JSON

```json
{
  "reps": 5,
  "state": "UP",
  "feedback": "Rep 5",
  "pose_detected": true,
  "checks": {
    "elbow_angle": 162.3,
    "body_line": 174.1,
    "hip_ok": true,
    "flare_ok": true,
    "wrist_ok": true
  },
  "landmarks": [{ "x": 0.52, "y": 0.34 }]
}
```

For timed exercises (plank, wall sit), `reps` contains the hold time in seconds.

### REST endpoints

```
GET    /history                            returns all sessions grouped by exercise
DELETE /history/{date}                     deletes a session by date string
GET    /coach/{user_id}/{exercise}?reps=0  returns personalized coaching message
```

The `/coach` endpoint triggers a Hindsight recall and cascadeflow routing call. Returns a coaching tip personalized to the user's history.

---

## History

Sessions are saved to `workout_history.csv` on WebSocket disconnect. The history page groups sessions by exercise and shows total reps/hold time and best single session per exercise.

---

## Deployment

The `Procfile` is configured for Heroku-style deployments:

```
web: cd backend && pip install -r requirements.txt && python -m uvicorn main:app --host 0.0.0.0 --port $PORT
```

For the frontend, build and serve statically:

```bash
cd frontend
npm run build
```

Set `VITE_BACKEND_URL` in `frontend/.env.production` to point to your deployed backend.

Set `HINDSIGHT_API_KEY`, `HINDSIGHT_PIPELINE_ID`, `HINDSIGHT_BASE_URL`, and `GROQ_API_KEY` as environment variables on your hosting platform.
