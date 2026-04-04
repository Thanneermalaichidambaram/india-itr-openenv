"""
Pydantic models for IndiaITR-OpenEnv.
All domain types, action types, observation, and state.
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Domain document models
# ---------------------------------------------------------------------------

class Form16PartA(BaseModel):
    """Employer-issued TDS certificate summary."""
    employer_name: str
    employee_pan: str
    gross_salary: float
    tds_q1: float
    tds_q2: float
    tds_q3: float
    tds_q4: float
    total_tds: float  # may differ from sum of quarters (mismatch scenario)


class Form16PartB(BaseModel):
    """Detailed salary breakup from employer."""
    basic_salary: float
    hra_received: float
    special_allowance: float
    lta: float
    gross_salary: float  # may differ from Part A gross (mismatch scenario)
    standard_deduction: float = 50000.0
    professional_tax: float
    net_taxable_salary: float
    city: str = "mumbai"  # for HRA metro/non-metro classification


class InvestmentStatement(BaseModel):
    """80C and other investment proofs."""
    ppf: float = 0.0
    elss: float = 0.0
    life_insurance: float = 0.0
    home_loan_principal: float = 0.0
    school_fees: float = 0.0
    nsc: float = 0.0
    # 80D
    health_insurance_self: float = 0.0
    health_insurance_parents: float = 0.0
    parents_senior_citizen: bool = False
    # 80TTA
    savings_account_interest: float = 0.0
    # 80G
    donations: float = 0.0
    # HRA
    rent_paid_monthly: float = 0.0


class Asset(BaseModel):
    """Capital asset for gains computation."""
    asset_id: str
    asset_type: Literal[
        "equity_listed", "equity_unlisted", "debt_mf", "gold_etf", "property"
    ]
    purchase_date: date
    purchase_amount: float
    sale_date: date
    sale_amount: float
    stt_paid: bool = False
    # For equity grandfathering (pre-Jan 31 2018)
    jan31_2018_nav: Optional[float] = None


class CaseFile(BaseModel):
    """Complete synthetic tax case for one episode."""
    case_id: str
    tax_year: str = "FY2024-25"
    form16_part_a: Form16PartA
    form16_part_b: Form16PartB
    investments: InvestmentStatement
    assets: List[Asset] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Action models — agent sends one of these per step
# ---------------------------------------------------------------------------

class ReadSectionAction(BaseModel):
    action_type: Literal["read_section"] = "read_section"
    section: Literal["form16_part_a", "form16_part_b", "investments", "assets"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractFieldAction(BaseModel):
    action_type: Literal["extract_field"] = "extract_field"
    field_name: str = Field(..., description="Dot-path field, e.g. 'form16_part_a.gross_salary'")
    extracted_value: Union[float, str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlagDeductionAction(BaseModel):
    action_type: Literal["flag_deduction"] = "flag_deduction"
    section: Literal["80C", "24b", "80D", "HRA", "80TTA", "80G"]
    amount: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SelectRegimeAction(BaseModel):
    action_type: Literal["select_regime"] = "select_regime"
    regime: Literal["old", "new"]
    computed_tax_old: float
    computed_tax_new: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComputeCapGainsAction(BaseModel):
    action_type: Literal["compute_capgains"] = "compute_capgains"
    asset_id: str
    indexed_cost: float
    gain: float
    tax_rule_applied: str  # e.g. 'ltcg_equity_112a', 'ltcg_debt_indexed', 'stcg_slab'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubmitReturnAction(BaseModel):
    action_type: Literal["submit"] = "submit"
    total_income: float
    total_deductions: float
    taxable_income: float
    tax_payable: float
    regime_selected: Literal["old", "new"]
    deductions_claimed: Dict[str, float] = Field(default_factory=dict)
    capital_gains_computed: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


ITRAction = Annotated[
    Union[
        ReadSectionAction,
        ExtractFieldAction,
        FlagDeductionAction,
        SelectRegimeAction,
        ComputeCapGainsAction,
        SubmitReturnAction,
    ],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# ITR draft — accumulated work in progress
# ---------------------------------------------------------------------------

class ITRDraft(BaseModel):
    total_income: float = 0.0
    deductions_claimed: Dict[str, float] = Field(default_factory=dict)
    regime_selected: Optional[str] = None
    tax_payable: float = 0.0
    capital_gains: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation — what the agent sees each turn
# ---------------------------------------------------------------------------

class ITRObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # OpenEnv required fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Task context
    turn: int = 0
    task_id: str = ""
    max_turns: int = 20

    # Visible document sections (only after agent reads them)
    sections_available: List[str] = Field(
        default_factory=lambda: ["form16_part_a", "form16_part_b", "investments", "assets"]
    )
    sections_read: List[str] = Field(default_factory=list)
    visible_data: Dict[str, Any] = Field(default_factory=dict)

    # Agent's accumulated work
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    flagged_deductions: Dict[str, float] = Field(default_factory=dict)
    capital_gains_computed: Dict[str, Any] = Field(default_factory=dict)

    # Current ITR draft
    current_draft: ITRDraft = Field(default_factory=ITRDraft)

    # Feedback
    last_action_result: str = ""
    validation_errors: List[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0


# ---------------------------------------------------------------------------
# State — full internal state (not shown to agent)
# ---------------------------------------------------------------------------

class ITRState(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = ""
    seed: int = 42
    done: bool = False
    cumulative_reward: float = 0.0
    # Internal: the true case file (oracle, not shown to agent)
    case_file: Optional[Dict[str, Any]] = None
