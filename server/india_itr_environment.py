"""
IndiaITR OpenEnv server environment.

Implements openenv.core.env_server.interfaces.Environment so create_app()
can wire it up automatically. Wraps the pure IndiaITREnvironment core.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.environment import IndiaITREnvironment
from env.models import (
    ComputeCapGainsAction,
    ExtractFieldAction,
    FlagDeductionAction,
    ReadSectionAction,
    SelectRegimeAction,
    SubmitReturnAction,
)

try:
    from ..models import ITRAction, ITRObservation
except (ImportError, ModuleNotFoundError):
    from models import ITRAction, ITRObservation

TASK_IDS = ["task1_parse", "task2_deduct", "task3_capgains"]

# Dispatch table: action_type str → concrete model class
_ACTION_MAP = {
    "read_section": ReadSectionAction,
    "extract_field": ExtractFieldAction,
    "flag_deduction": FlagDeductionAction,
    "select_regime": SelectRegimeAction,
    "compute_capgains": ComputeCapGainsAction,
    "submit": SubmitReturnAction,
}


def _to_concrete(action: ITRAction) -> Any:
    """Convert ITRAction (flat wrapper) to the concrete typed action model."""
    atype = action.action_type
    cls = _ACTION_MAP.get(atype)
    if cls is None:
        raise ValueError(
            f"Unknown action_type: {atype!r}. Must be one of {list(_ACTION_MAP)}"
        )
    return cls.model_validate(action.model_dump(exclude_none=False))


def _to_itr_obs(inner_obs) -> ITRObservation:
    """Convert internal ITRObservation to the root-level ITRObservation."""
    return ITRObservation(
        done=inner_obs.done,
        reward=inner_obs.reward,
        metadata=inner_obs.metadata,
        turn=inner_obs.turn,
        task_id=inner_obs.task_id,
        max_turns=inner_obs.max_turns,
        sections_available=inner_obs.sections_available,
        sections_read=list(inner_obs.sections_read),
        visible_data=dict(inner_obs.visible_data),
        extracted_fields=dict(inner_obs.extracted_fields),
        flagged_deductions=dict(inner_obs.flagged_deductions),
        capital_gains_computed=dict(inner_obs.capital_gains_computed),
        current_draft=inner_obs.current_draft.model_copy(),
        last_action_result=inner_obs.last_action_result,
        validation_errors=list(inner_obs.validation_errors),
        cumulative_reward=inner_obs.cumulative_reward,
    )


class IndiaITROpenEnvEnvironment(Environment):
    """
    OpenEnv-compliant wrapper around IndiaITREnvironment.

    A new instance is created per /reset call (HTTP mode: single-session).
    In WebSocket mode create_app() creates one per WS connection.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._task_id: str = "task1_parse"
        self._core: Optional[IndiaITREnvironment] = None
        self._state_obj: State = State(episode_id=None, step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ITRObservation:
        tid = task_id or kwargs.get("task_id") or self._task_id
        if tid not in TASK_IDS:
            tid = "task1_parse"
        self._task_id = tid

        eid = episode_id or str(uuid.uuid4())
        effective_seed = seed if seed is not None else 42

        self._core = IndiaITREnvironment(task_id=tid, seed=effective_seed)
        inner_obs = self._core.reset(seed=effective_seed, episode_id=eid)

        self._state_obj = State(episode_id=eid, step_count=0)
        return _to_itr_obs(inner_obs)

    def step(
        self,
        action: ITRAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ITRObservation:
        if self._core is None:
            raise RuntimeError("Call reset() before step()")
        concrete = _to_concrete(action)
        inner_obs = self._core.step(concrete)
        self._state_obj.step_count = self._core.state.step_count
        return _to_itr_obs(inner_obs)

    @property
    def state(self) -> State:
        if self._core is not None:
            s = self._core.state
            return State(
                episode_id=s.episode_id,
                step_count=s.step_count,
            )
        return self._state_obj

    def close(self) -> None:
        if self._core is not None:
            self._core.close()

    # ------------------------------------------------------------------
    # Extra helpers (not required by openenv interface)
    # ------------------------------------------------------------------

    def final_grade(self) -> float:
        if self._core is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._core.final_grade()
