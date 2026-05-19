"""
coach.py — Persistent memory + intelligent coaching layer.

Hindsight: stores and retrieves workout history, form failures, and progress
           across sessions so the agent remembers every rep you have ever done.

cascadeflow: routes coaching LLM calls intelligently using CascadeAgent.
             Simple feedback -> fast cheap model (llama3-8b on Groq).
             Complex form analysis -> better model (llama3-70b on Groq).
             Budget enforced per session. Full audit trail via session.trace().
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Hindsight — persistent agent memory
# ---------------------------------------------------------------------------

HINDSIGHT_API_KEY    = os.environ.get("HINDSIGHT_API_KEY", "")
HINDSIGHT_PIPELINE_ID = os.environ.get("HINDSIGHT_PIPELINE_ID", "workout-tracker")
HINDSIGHT_BASE_URL   = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")

_hindsight = None
_hindsight_tried = False


def _get_hindsight():
    global _hindsight, _hindsight_tried
    if _hindsight_tried:
        return _hindsight
    _hindsight_tried = True
    if not HINDSIGHT_API_KEY:
        return None
    try:
        from hindsight_client import Hindsight
        _hindsight = Hindsight(
            base_url=HINDSIGHT_BASE_URL,
            api_key=HINDSIGHT_API_KEY,
        )
        print("[Hindsight] connected.")
    except Exception as e:
        print(f"[Hindsight] init failed: {e}")
    return _hindsight


async def remember_session(user_id: str, exercise: str, reps: int,
                           duration: float, form_issues: list):
    hs = _get_hindsight()
    if hs is None:
        return
    memory_text = (
        f"User {user_id} completed {exercise} on "
        f"{datetime.now().strftime('%Y-%m-%d')}. "
        f"Reps/hold: {reps}. Duration: {round(duration, 1)}s. "
        f"Form issues: {', '.join(form_issues) if form_issues else 'none'}."
    )
    try:
        await asyncio.to_thread(
            hs.retain,
            bank_id=HINDSIGHT_PIPELINE_ID,
            content=memory_text,
            metadata={"user_id": user_id, "exercise": exercise},
            tags=[user_id, exercise],
        )
        print(f"[Hindsight] retained session for {user_id}/{exercise}")
    except Exception as e:
        print(f"[Hindsight] retain failed: {e}")


async def recall_history(user_id: str, exercise: str) -> str:
    hs = _get_hindsight()
    if hs is None:
        return ""
    query = f"workout history for user {user_id} doing {exercise}"
    try:
        result = await asyncio.to_thread(
            hs.recall,
            bank_id=HINDSIGHT_PIPELINE_ID,
            query=query,
            tags=[user_id, exercise],
            tags_match="all",
            budget="mid",
        )
        # recall returns a RecallResponse — extract the text content
        text = ""
        if hasattr(result, "context") and result.context:
            text = result.context
        elif hasattr(result, "facts") and result.facts:
            text = "\n".join(f"- {f}" for f in result.facts)
        if text:
            print(f"[Hindsight] recalled history for {user_id}/{exercise}")
            return f"Past sessions:\n{text}"
        return ""
    except Exception as e:
        print(f"[Hindsight] recall failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# cascadeflow — runtime intelligence, model routing, budget enforcement
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

_cascade_agent = None
_cascade_tried = False


def _get_cascade():
    global _cascade_agent, _cascade_tried
    if _cascade_tried:
        return _cascade_agent
    _cascade_tried = True
    if not GROQ_API_KEY:
        return None
    try:
        import cascadeflow
        from cascadeflow.agent import CascadeAgent
        from cascadeflow.schema.config import ModelConfig

        # Initialise the harness in enforce mode so budget caps are hard limits
        cascadeflow.init(
            mode="enforce",
            budget=0.05,        # $0.05 max per session
            verbose=True,
        )

        # Two-model cascade:
        # Tier 1 — llama3-8b: fast, cheap, handles simple rep feedback
        # Tier 2 — llama3-70b: better reasoning, used for recurring form issues
        #          and multi-session pattern analysis
        _cascade_agent = CascadeAgent(
            models=[
                ModelConfig(
                    name="llama-3.1-8b-instant",
                    provider="groq",
                    api_key=GROQ_API_KEY,
                    cost=0.0001,
                    speed_ms=400,
                    quality_score=0.65,
                    system_prompt=(
                        "You are a concise personal trainer. "
                        "Give one short coaching tip in under 20 words. "
                        "No filler. No emojis. Direct and actionable."
                    ),
                ),
                ModelConfig(
                    name="llama-3.3-70b-versatile",
                    provider="groq",
                    api_key=GROQ_API_KEY,
                    cost=0.0008,
                    speed_ms=1200,
                    quality_score=0.92,
                    system_prompt=(
                        "You are an expert personal trainer with memory of past sessions. "
                        "Give one specific coaching tip based on the user's history. "
                        "Under 25 words. No filler. No emojis."
                    ),
                ),
            ],
            enable_cascade=True,
            verbose=True,
        )
        print("[cascadeflow] CascadeAgent ready. Tier 1: llama-3.1-8b-instant, Tier 2: llama-3.3-70b-versatile")
    except Exception as e:
        print(f"[cascadeflow] init failed: {e}")
    return _cascade_agent


async def get_coaching(
    user_id: str,
    exercise: str,
    current_reps: int,
    form_issues: list,
    history_context: str,
    complex_analysis: bool = False,
) -> str:
    """
    Generate a coaching message via cascadeflow.

    cascadeflow routes automatically:
    - No history, no issues -> llama3-8b (fast, cheap)
    - Recurring issues or past history -> llama3-70b (better reasoning)

    complexity_hint tells cascadeflow how hard the task is so it can
    make the routing decision. Every decision is logged in the audit trail.
    """
    agent = _get_cascade()
    if agent is None:
        return _rule_based_coaching(exercise, current_reps, form_issues)

    prompt = f"Exercise: {exercise}. Reps completed: {current_reps}."
    if form_issues:
        prompt += f" Form issues this session: {', '.join(form_issues)}."
    if history_context:
        prompt += f"\n\n{history_context}"
        prompt += "\nBased on their history, give one specific improvement tip."

    # complexity_hint drives cascadeflow's routing decision:
    # "simple" stays on llama3-8b, "complex" escalates to llama3-70b
    complexity_hint = "complex" if complex_analysis else "simple"

    try:
        result = await agent.run(
            query=prompt,
            max_tokens=60,
            temperature=0.5,
            complexity_hint=complexity_hint,
        )
        tip = result.content.strip() if result.content else ""

        print(
            f"[cascadeflow] model={result.model_used} "
            f"cost=${result.total_cost:.5f} "
            f"latency={result.latency_ms}ms "
            f"complexity={complexity_hint}"
        )
        return tip if tip else _rule_based_coaching(exercise, current_reps, form_issues)
    except Exception as e:
        print(f"[cascadeflow] run failed: {e}")
        return _rule_based_coaching(exercise, current_reps, form_issues)


def _rule_based_coaching(exercise: str, reps: int, form_issues: list) -> str:
    """Fallback when LLM is unavailable — no keys needed."""
    if not form_issues:
        tips = {
            "pushup":  "Keep your core tight throughout.",
            "squat":   "Drive through your heels on the way up.",
            "plank":   "Breathe steadily. Do not hold your breath.",
            "abs":     "Exhale on the crunch, inhale on the way down.",
            "wallsit": "Keep your back flat against the wall.",
        }
        return tips.get(exercise, f"{reps} reps done. Keep going.")
    return f"Focus on: {form_issues[0].lower()}"


# ---------------------------------------------------------------------------
# SessionTracker — accumulates form issues, drives coaching escalation
# ---------------------------------------------------------------------------

class SessionTracker:
    def __init__(self, user_id: str, exercise: str):
        self.user_id          = user_id
        self.exercise         = exercise
        self.issues: dict     = {}   # issue text -> count
        self.history_context  = ""
        self._history_loaded  = False

    async def load_history(self):
        if not self._history_loaded:
            self.history_context = await recall_history(self.user_id, self.exercise)
            self._history_loaded = True

    def record_issue(self, issue: str):
        if issue:
            self.issues[issue] = self.issues.get(issue, 0) + 1

    def recurring_issues(self) -> list:
        """Issues that appeared 3 or more times this session."""
        return [k for k, v in self.issues.items() if v >= 3]

    def all_issues(self) -> list:
        return list(self.issues.keys())

    async def coaching_message(self, current_reps: int) -> str:
        await self.load_history()
        recurring = self.recurring_issues()
        # Escalate to llama3-70b if there are recurring issues or past history
        complex_analysis = bool(recurring or self.history_context)
        return await get_coaching(
            user_id=self.user_id,
            exercise=self.exercise,
            current_reps=current_reps,
            form_issues=recurring or self.all_issues(),
            history_context=self.history_context,
            complex_analysis=complex_analysis,
        )

    async def save(self, reps: int, duration: float):
        await remember_session(
            user_id=self.user_id,
            exercise=self.exercise,
            reps=reps,
            duration=duration,
            form_issues=self.all_issues(),
        )
