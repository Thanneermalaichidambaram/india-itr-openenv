"""
Dense reward calculator — called by the environment on every step.
"""
from __future__ import annotations

from .models import (
    ComputeCapGainsAction,
    ExtractFieldAction,
    FlagDeductionAction,
    ReadSectionAction,
    SelectRegimeAction,
    SubmitReturnAction,
)

# Per-action reward table
REWARDS = {
    "read_new_section": 0.05,
    "read_already_seen": 0.00,
    "field_correct": 0.10,
    "field_wrong_value": -0.05,
    "field_unknown": -0.02,
    "deduction_correct": 0.15,
    "deduction_wrong_section": -0.10,
    "deduction_overclaim": -0.05,
    "regime_correct": 0.15,
    "regime_wrong": -0.20,
    "capgains_correct": 0.12,
    "capgains_wrong_rule": -0.15,
    "capgains_wrong_value": -0.08,
    "submit_pass": 0.30,
    "submit_partial": 0.10,
    "loop_penalty": -0.05,
}


def reward_read_section(action: ReadSectionAction, sections_read: list[str]) -> float:
    if action.section not in sections_read:
        return REWARDS["read_new_section"]
    return REWARDS["read_already_seen"]


def reward_extract_field(
    action: ExtractFieldAction,
    true_value: float | str | None,
    tolerance: float = 0.01,
) -> float:
    if true_value is None:
        return REWARDS["field_unknown"]
    try:
        true_f = float(true_value)
        agent_f = float(action.extracted_value)
        if true_f == 0:
            return REWARDS["field_correct"] if agent_f == 0 else REWARDS["field_wrong_value"]
        if abs(agent_f - true_f) / abs(true_f) <= tolerance:
            return REWARDS["field_correct"]
        return REWARDS["field_wrong_value"]
    except (TypeError, ValueError):
        return REWARDS["field_correct"] if str(action.extracted_value) == str(true_value) else REWARDS["field_wrong_value"]


def reward_flag_deduction(
    action: FlagDeductionAction,
    true_deductions: dict[str, float],
) -> float:
    true_amount = true_deductions.get(action.section, 0.0)
    if true_amount == 0 and action.amount == 0:
        return 0.0
    if true_amount == 0 and action.amount > 0:
        return REWARDS["deduction_wrong_section"]
    error_ratio = abs(action.amount - true_amount) / max(true_amount, 1)
    if error_ratio <= 0.02:
        return REWARDS["deduction_correct"]
    if error_ratio <= 0.10:
        return REWARDS["deduction_correct"] * 0.5
    return REWARDS["deduction_wrong_section"]


def reward_select_regime(
    action: SelectRegimeAction,
    true_regime: str,
) -> float:
    if action.regime == true_regime:
        return REWARDS["regime_correct"]
    return REWARDS["regime_wrong"]


def reward_capgains(
    action: ComputeCapGainsAction,
    true_indexed_cost: float,
    true_gain: float,
    true_rule: str,
) -> float:
    rule_correct = action.tax_rule_applied == true_rule
    cost_err = abs(action.indexed_cost - true_indexed_cost) / max(abs(true_indexed_cost), 1)
    gain_err = abs(action.gain - true_gain) / max(abs(true_gain), 1)

    if not rule_correct:
        return REWARDS["capgains_wrong_rule"]
    if cost_err > 0.05 or gain_err > 0.05:
        return REWARDS["capgains_wrong_value"]
    return REWARDS["capgains_correct"]
