"""
Task 1 Grader — Form 16 document parsing.
Score = 0.60 × field_accuracy + 0.40 × reconciliation_score
"""
from __future__ import annotations

from typing import Any, Dict

from env.models import CaseFile


# The 12 fields the agent is expected to extract
REQUIRED_FIELDS = [
    "form16_part_a.employer_name",
    "form16_part_a.employee_pan",
    "form16_part_a.gross_salary",
    "form16_part_a.tds_q1",
    "form16_part_a.tds_q2",
    "form16_part_a.tds_q3",
    "form16_part_a.tds_q4",
    "form16_part_a.total_tds",
    "form16_part_b.basic_salary",
    "form16_part_b.hra_received",
    "form16_part_b.gross_salary",
    "form16_part_b.net_taxable_salary",
]

FIELD_WEIGHT = 0.60 / len(REQUIRED_FIELDS)  # per field
RECONCILE_WEIGHT = 0.40


def _get_true_value(case: CaseFile, field_path: str) -> Any:
    parts = field_path.split(".")
    obj = case
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _value_correct(agent_val: Any, true_val: Any, tol: float = 0.01) -> bool:
    if isinstance(true_val, str):
        return str(agent_val).strip().lower() == str(true_val).strip().lower()
    try:
        af, tf = float(agent_val), float(true_val)
        if tf == 0:
            return af == 0
        return abs(af - tf) / abs(tf) <= tol
    except (TypeError, ValueError):
        return False


class Task1Grader:
    def grade(
        self,
        extracted_fields: Dict[str, Any],
        case: CaseFile,
    ) -> float:
        # --- Field accuracy (0.60) ---
        field_score = 0.0
        for field in REQUIRED_FIELDS:
            true_val = _get_true_value(case, field)
            agent_val = extracted_fields.get(field)
            if agent_val is not None and _value_correct(agent_val, true_val):
                field_score += FIELD_WEIGHT

        # --- Reconciliation score (0.40) ---
        reconcile_score = 0.0

        # +0.20: did agent detect Part A/B gross mismatch correctly?
        part_a_gross = case.form16_part_a.gross_salary
        part_b_gross = case.form16_part_b.gross_salary
        true_mismatch = abs(part_a_gross - part_b_gross) > 1.0

        agent_a_gross = extracted_fields.get("form16_part_a.gross_salary")
        agent_b_gross = extracted_fields.get("form16_part_b.gross_salary")
        if agent_a_gross is not None and agent_b_gross is not None:
            agent_detected_mismatch = abs(float(agent_a_gross) - float(agent_b_gross)) > 1.0
            if agent_detected_mismatch == true_mismatch:
                reconcile_score += 0.20

        # +0.20: did agent detect TDS quarterly sum mismatch?
        true_tds_sum = round(
            case.form16_part_a.tds_q1 + case.form16_part_a.tds_q2 +
            case.form16_part_a.tds_q3 + case.form16_part_a.tds_q4, 2
        )
        true_total_tds = case.form16_part_a.total_tds
        true_tds_mismatch = abs(true_tds_sum - true_total_tds) > 1.0

        agent_q1 = extracted_fields.get("form16_part_a.tds_q1")
        agent_q2 = extracted_fields.get("form16_part_a.tds_q2")
        agent_q3 = extracted_fields.get("form16_part_a.tds_q3")
        agent_q4 = extracted_fields.get("form16_part_a.tds_q4")
        agent_total = extracted_fields.get("form16_part_a.total_tds")

        if all(v is not None for v in [agent_q1, agent_q2, agent_q3, agent_q4, agent_total]):
            agent_sum = float(agent_q1) + float(agent_q2) + float(agent_q3) + float(agent_q4)
            agent_detected_tds_mismatch = abs(agent_sum - float(agent_total)) > 1.0
            if agent_detected_tds_mismatch == true_tds_mismatch:
                reconcile_score += 0.20

        return round(field_score + reconcile_score, 3)
