"""
Server-side environment wrapper.
Wraps IndiaITREnvironment for the OpenEnv FastAPI server.
Handles session state and serialization.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from env.environment import IndiaITREnvironment
from env.models import (
    ComputeCapGainsAction,
    ExtractFieldAction,
    FlagDeductionAction,
    ITRObservation,
    ITRState,
    ReadSectionAction,
    SelectRegimeAction,
    SubmitReturnAction,
)

# Map action_type string → Pydantic model
ACTION_MAP = {
    "read_section": ReadSectionAction,
    "extract_field": ExtractFieldAction,
    "flag_deduction": FlagDeductionAction,
    "select_regime": SelectRegimeAction,
    "compute_capgains": ComputeCapGainsAction,
    "submit": SubmitReturnAction,
}

TASK_IDS = ["task1_parse", "task2_deduct", "task3_capgains"]


class IndiaITRServerEnvironment:
    """
    Multi-session server environment.
    Each reset() creates a fresh IndiaITREnvironment instance.
    Thread-safe for read; designed for single-threaded async FastAPI.
    """

    def __init__(self):
        # task_id and seed can be passed at reset time
        self._envs: Dict[str, IndiaITREnvironment] = {}
        self._current_session: Optional[str] = None
        self._default_task = "task1_parse"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> ITRObservation:
        tid = task_id or self._default_task
        if tid not in TASK_IDS:
            tid = self._default_task
        sid = episode_id or str(uuid.uuid4())
        env = IndiaITREnvironment(task_id=tid, seed=seed or 42)
        obs = env.reset(seed=seed, episode_id=sid)
        self._envs[sid] = env
        self._current_session = sid
        return obs

    def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs) -> ITRObservation:
        env = self._get_current_env()
        # If action is a dict, deserialize it
        if isinstance(action, dict):
            action = self._deserialize_action(action)
        return env.step(action)

    @property
    def state(self) -> ITRState:
        env = self._get_current_env()
        return env.state

    def close(self) -> None:
        if self._current_session and self._current_session in self._envs:
            del self._envs[self._current_session]

    def final_grade(self) -> float:
        env = self._get_current_env()
        return env.final_grade()

    def _get_current_env(self) -> IndiaITREnvironment:
        if not self._current_session or self._current_session not in self._envs:
            raise RuntimeError("No active session. Call reset() first.")
        return self._envs[self._current_session]

    @staticmethod
    def _deserialize_action(data: dict) -> Any:
        action_type = data.get("action_type")
        cls = ACTION_MAP.get(action_type)
        if cls is None:
            raise ValueError(f"Unknown action_type: {action_type!r}. Must be one of {list(ACTION_MAP)}")
        return cls.model_validate(data)
