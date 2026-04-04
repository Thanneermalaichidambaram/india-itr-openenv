"""
IndiaITREnvironment — OpenEnv-compliant environment core.
Pure state machine; zero I/O. Implements reset() / step() / state property.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .case_generator import CaseGenerator
from .models import (
    CaseFile,
    ComputeCapGainsAction,
    ExtractFieldAction,
    FlagDeductionAction,
    ITRDraft,
    ITRObservation,
    ITRState,
    ReadSectionAction,
    SelectRegimeAction,
    SubmitReturnAction,
)
from . import reward as reward_module
from . import tax_engine

TASK_MAX_TURNS = {
    "task1_parse": 20,
    "task2_deduct": 20,
    "task3_capgains": 25,
}


def _case_section_data(case: CaseFile, section: str) -> Dict[str, Any]:
    """Return the visible data dict for a document section."""
    if section == "form16_part_a":
        return case.form16_part_a.model_dump()
    if section == "form16_part_b":
        return case.form16_part_b.model_dump()
    if section == "investments":
        return case.investments.model_dump()
    if section == "assets":
        return {"assets": [a.model_dump() for a in case.assets]}
    return {}


def _lookup_true_field(case: CaseFile, field_path: str) -> Any:
    parts = field_path.split(".")
    obj = case
    for p in parts:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


class IndiaITREnvironment:
    """OpenEnv environment for Indian ITR filing (3 tasks)."""

    TASK_IDS = list(TASK_MAX_TURNS.keys())

    def __init__(self, task_id: str = "task1_parse", seed: int = 42):
        if task_id not in self.TASK_IDS:
            raise ValueError(f"task_id must be one of {self.TASK_IDS}")
        self._task_id = task_id
        self._seed = seed
        self._rng: Optional[np.random.Generator] = None
        self._case: Optional[CaseFile] = None
        self._obs: Optional[ITRObservation] = None
        self._state_obj: ITRState = ITRState()
        self._true_deductions: Dict[str, float] = {}
        self._submitted = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ITRObservation:
        effective_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(effective_seed)
        self._case = CaseGenerator(self._rng).generate(self._task_id)
        self._submitted = False
        self._true_deductions = tax_engine.compute_all_deductions(self._case)

        self._state_obj = ITRState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task_id,
            seed=effective_seed,
            done=False,
            cumulative_reward=0.0,
            case_file=self._case.model_dump(mode="json"),
        )

        self._obs = ITRObservation(
            done=False,
            reward=0.0,
            turn=0,
            task_id=self._task_id,
            max_turns=TASK_MAX_TURNS[self._task_id],
            sections_read=[],
            visible_data={},
            extracted_fields={},
            flagged_deductions={},
            capital_gains_computed={},
            current_draft=ITRDraft(),
            last_action_result="Episode started. Read a section to begin.",
            validation_errors=[],
            cumulative_reward=0.0,
        )
        return self._obs

    def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs) -> ITRObservation:
        assert self._case is not None, "Call reset() before step()"
        assert not self._submitted, "Episode already done. Call reset()."

        self._state_obj.step_count += 1
        turn = self._state_obj.step_count
        max_turns = TASK_MAX_TURNS[self._task_id]

        step_reward, result_msg, errors = self._execute_action(action)

        # Loop penalty: if extracting nothing new after turn 8
        if turn > 8 and isinstance(action, ReadSectionAction):
            if action.section in (self._obs.sections_read or []):
                step_reward += reward_module.REWARDS["loop_penalty"]

        self._state_obj.cumulative_reward += step_reward

        done = self._submitted or turn >= max_turns

        self._obs = ITRObservation(
            done=done,
            reward=round(step_reward, 4),
            turn=turn,
            task_id=self._task_id,
            max_turns=max_turns,
            sections_read=list(self._obs.sections_read),
            visible_data=dict(self._obs.visible_data),
            extracted_fields=dict(self._obs.extracted_fields),
            flagged_deductions=dict(self._obs.flagged_deductions),
            capital_gains_computed=dict(self._obs.capital_gains_computed),
            current_draft=self._obs.current_draft.model_copy(),
            last_action_result=result_msg,
            validation_errors=errors,
            cumulative_reward=round(self._state_obj.cumulative_reward, 4),
        )

        self._state_obj.done = done
        return self._obs

    @property
    def state(self) -> ITRState:
        return self._state_obj

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Episode summary for graders
    # ------------------------------------------------------------------

    def final_grade(self) -> float:
        """Compute final task score via the appropriate grader."""
        if self._case is None or self._obs is None:
            return 0.0

        if self._task_id == "task1_parse":
            from graders.task1_grader import Task1Grader
            return Task1Grader().grade(self._obs.extracted_fields, self._case)

        if self._task_id == "task2_deduct":
            from graders.task2_grader import Task2Grader
            # Build a pseudo SubmitReturnAction from obs state
            sub = SubmitReturnAction(
                total_income=self._obs.current_draft.total_income,
                total_deductions=sum(self._obs.flagged_deductions.values()),
                taxable_income=self._obs.current_draft.total_income - sum(self._obs.flagged_deductions.values()),
                tax_payable=self._obs.current_draft.tax_payable,
                regime_selected=self._obs.current_draft.regime_selected or "new",
                deductions_claimed=dict(self._obs.flagged_deductions),
            )
            return Task2Grader().grade(sub, self._case)

        if self._task_id == "task3_capgains":
            from graders.task3_grader import Task3Grader
            from env.models import ComputeCapGainsAction
            agent_assets = [
                ComputeCapGainsAction(
                    asset_id=aid,
                    indexed_cost=float(data.get("indexed_cost", 0)),
                    gain=float(data.get("gain", 0)),
                    tax_rule_applied=str(data.get("tax_rule_applied", "")),
                )
                for aid, data in self._obs.capital_gains_computed.items()
            ]
            return Task3Grader().grade(agent_assets, self._case)

        return 0.0

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: Any):
        """Dispatch action, return (reward, result_msg, errors)."""
        obs = self._obs
        case = self._case

        if isinstance(action, ReadSectionAction):
            return self._do_read_section(action, obs, case)

        if isinstance(action, ExtractFieldAction):
            return self._do_extract_field(action, obs, case)

        if isinstance(action, FlagDeductionAction):
            return self._do_flag_deduction(action, obs, case)

        if isinstance(action, SelectRegimeAction):
            return self._do_select_regime(action, obs, case)

        if isinstance(action, ComputeCapGainsAction):
            return self._do_compute_capgains(action, obs, case)

        if isinstance(action, SubmitReturnAction):
            return self._do_submit(action, obs, case)

        return -0.05, f"Unknown action type: {type(action).__name__}", ["Unknown action"]

    def _do_read_section(self, action: ReadSectionAction, obs: ITRObservation, case: CaseFile):
        r = reward_module.reward_read_section(action, obs.sections_read)
        if action.section not in obs.sections_read:
            obs.sections_read.append(action.section)
        obs.visible_data[action.section] = _case_section_data(case, action.section)
        return r, f"Read section: {action.section}", []

    def _do_extract_field(self, action: ExtractFieldAction, obs: ITRObservation, case: CaseFile):
        true_val = _lookup_true_field(case, action.field_name)
        r = reward_module.reward_extract_field(action, true_val)
        obs.extracted_fields[action.field_name] = action.extracted_value
        status = "correct" if r >= 0.05 else "incorrect"
        return r, f"Extracted {action.field_name} = {action.extracted_value} [{status}]", []

    def _do_flag_deduction(self, action: FlagDeductionAction, obs: ITRObservation, case: CaseFile):
        r = reward_module.reward_flag_deduction(action, self._true_deductions)
        obs.flagged_deductions[action.section] = action.amount
        obs.current_draft.deductions_claimed[action.section] = action.amount
        status = "correct" if r > 0 else "incorrect"
        return r, f"Flagged {action.section} = ₹{action.amount:,.0f} [{status}]", []

    def _do_select_regime(self, action: SelectRegimeAction, obs: ITRObservation, case: CaseFile):
        true_regime = tax_engine.optimal_regime(case)
        r = reward_module.reward_select_regime(action, true_regime)
        obs.current_draft.regime_selected = action.regime
        true_tax = tax_engine.compute_tax(case, action.regime)
        obs.current_draft.tax_payable = true_tax  # use oracle tax for draft
        status = "correct" if action.regime == true_regime else f"wrong (optimal={true_regime})"
        return r, f"Selected regime={action.regime} [{status}]", []

    def _do_compute_capgains(self, action: ComputeCapGainsAction, obs: ITRObservation, case: CaseFile):
        asset = next((a for a in case.assets if a.asset_id == action.asset_id), None)
        if asset is None:
            return -0.05, f"Asset {action.asset_id} not found", [f"Unknown asset_id: {action.asset_id}"]
        true_cost = tax_engine.indexed_cost(asset)
        true_gain = tax_engine.capital_gain(asset)
        true_rule = tax_engine.applicable_rule(asset)
        r = reward_module.reward_capgains(action, true_cost, true_gain, true_rule)
        obs.capital_gains_computed[action.asset_id] = {
            "indexed_cost": action.indexed_cost,
            "gain": action.gain,
            "tax_rule_applied": action.tax_rule_applied,
        }
        obs.current_draft.capital_gains[action.asset_id] = action.gain
        status = "correct" if r > 0 else f"incorrect (rule={true_rule})"
        return r, f"Computed capgains for {action.asset_id}: gain=₹{action.gain:,.0f} rule={action.tax_rule_applied} [{status}]", []

    def _do_submit(self, action: SubmitReturnAction, obs: ITRObservation, case: CaseFile):
        self._submitted = True
        errors: List[str] = []

        if action.total_income <= 0:
            errors.append("total_income must be positive")
        if action.tax_payable < 0:
            errors.append("tax_payable cannot be negative")

        # Score the submission
        grade = self.final_grade()
        r = reward_module.REWARDS["submit_pass"] * grade

        obs.current_draft.total_income = action.total_income
        obs.current_draft.tax_payable = action.tax_payable
        obs.current_draft.regime_selected = action.regime_selected

        return r, f"Submitted ITR. Final grade: {grade:.3f}", errors
