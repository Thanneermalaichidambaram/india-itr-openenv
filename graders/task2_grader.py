"""
Task 2 Grader — Deductions + Regime selection.
Score = 0.40 × deduction_accuracy + 0.35 × regime_correct + 0.25 × tax_accuracy
"""
from __future__ import annotations

from typing import Dict

from env import tax_engine
from env.models import CaseFile, SubmitReturnAction


class Task2Grader:
    def grade(
        self,
        submission: SubmitReturnAction,
        case: CaseFile,
    ) -> float:
        true_deductions = tax_engine.compute_all_deductions(case)
        true_regime = tax_engine.optimal_regime(case)
        true_tax = tax_engine.compute_tax(case, true_regime)

        # --- Deduction accuracy (0.40) ---
        deduct_errors = []
        for section, true_amt in true_deductions.items():
            agent_amt = submission.deductions_claimed.get(section, 0.0)
            denom = max(abs(true_amt), 1.0)
            deduct_errors.append(abs(agent_amt - true_amt) / denom)

        avg_error = sum(deduct_errors) / max(len(deduct_errors), 1)
        deduct_score = max(0.0, 1.0 - avg_error) * 0.40

        # --- Regime selection (0.35) — binary ---
        regime_score = 0.35 if submission.regime_selected == true_regime else 0.0

        # --- Tax accuracy (0.25) ---
        tax_error = abs(submission.tax_payable - true_tax) / max(abs(true_tax), 1.0)
        tax_score = max(0.0, 1.0 - tax_error) * 0.25

        return round(deduct_score + regime_score + tax_score, 3)
