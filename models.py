"""
Root-level models for IndiaITR-OpenEnv.

Follows the official OpenEnv pattern: Action and Observation subclass
openenv.core.env_server.types base classes.

The ITR environment has 6 action types (discriminated union).
ITRActionWrapper wraps the union so create_app() can deserialize any of them.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

from env.models import (
    ITRDraft,
    ITRObservation as _ITRObservation,
)


# ---------------------------------------------------------------------------
# Action wrapper — a single Action subclass that accepts all 6 action types
# ---------------------------------------------------------------------------

class ITRAction(Action):
    """
    Discriminated-union action for Indian ITR filing.

    The agent sends exactly one action_type per step:
      read_section, extract_field, flag_deduction,
      select_regime, compute_capgains, submit
    """

    # Required by base Action; keep extra='allow' so any ITR action passes
    model_config = Action.model_config.copy()
    model_config["extra"] = "allow"

    action_type: str = Field(
        ...,
        description=(
            "One of: read_section | extract_field | flag_deduction | "
            "select_regime | compute_capgains | submit"
        ),
    )

    # Optional fields — present depending on action_type
    # read_section
    section: Optional[str] = Field(None, description="Section name for read_section")

    # extract_field
    field_name: Optional[str] = Field(None, description="Dot-path field name")
    extracted_value: Optional[Any] = Field(None, description="Agent-extracted value")

    # flag_deduction
    amount: Optional[float] = Field(None, description="Deduction amount in INR")

    # select_regime
    regime: Optional[str] = Field(None, description="'old' or 'new'")
    computed_tax_old: Optional[float] = Field(None)
    computed_tax_new: Optional[float] = Field(None)

    # compute_capgains
    asset_id: Optional[str] = Field(None, description="Asset identifier")
    indexed_cost: Optional[float] = Field(None)
    gain: Optional[float] = Field(None)
    tax_rule_applied: Optional[str] = Field(None)

    # submit
    total_income: Optional[float] = Field(None)
    total_deductions: Optional[float] = Field(None)
    taxable_income: Optional[float] = Field(None)
    tax_payable: Optional[float] = Field(None)
    regime_selected: Optional[str] = Field(None)
    deductions_claimed: Dict[str, float] = Field(default_factory=dict)
    capital_gains_computed: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation — thin wrapper re-exposing the internal ITRObservation fields
# ---------------------------------------------------------------------------

class ITRObservation(Observation):
    """Observation returned by the IndiaITR environment each step."""

    # Task context
    turn: int = Field(default=0)
    task_id: str = Field(default="")
    max_turns: int = Field(default=20)

    # Visible document sections
    sections_available: List[str] = Field(
        default_factory=lambda: ["form16_part_a", "form16_part_b", "investments", "assets"]
    )
    sections_read: List[str] = Field(default_factory=list)
    visible_data: Dict[str, Any] = Field(default_factory=dict)

    # Agent's accumulated work
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    flagged_deductions: Dict[str, float] = Field(default_factory=dict)
    capital_gains_computed: Dict[str, Any] = Field(default_factory=dict)

    # ITR draft
    current_draft: ITRDraft = Field(default_factory=ITRDraft)

    # Feedback
    last_action_result: str = Field(default="")
    validation_errors: List[str] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
