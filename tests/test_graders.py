"""
Tests for all three graders: perfect agent, zero agent, common mistakes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from datetime import date

from env.models import (
    Asset, CaseFile, ComputeCapGainsAction, Form16PartA, Form16PartB,
    InvestmentStatement, SubmitReturnAction,
)
from env import tax_engine
from graders.task1_grader import Task1Grader
from graders.task2_grader import Task2Grader
from graders.task3_grader import Task3Grader


def make_case():
    rng = np.random.default_rng(42)
    from env.case_generator import CaseGenerator
    return CaseGenerator(rng).generate("task1_parse")


def make_case_task2():
    rng = np.random.default_rng(42)
    from env.case_generator import CaseGenerator
    return CaseGenerator(rng).generate("task2_deduct")


def make_case_task3():
    rng = np.random.default_rng(42)
    from env.case_generator import CaseGenerator
    return CaseGenerator(rng).generate("task3_capgains")


# --- Task 1 Grader ---

def test_task1_perfect_agent():
    case = make_case()
    # Build perfect extractions
    fields = {
        "form16_part_a.employer_name": case.form16_part_a.employer_name,
        "form16_part_a.employee_pan": case.form16_part_a.employee_pan,
        "form16_part_a.gross_salary": case.form16_part_a.gross_salary,
        "form16_part_a.tds_q1": case.form16_part_a.tds_q1,
        "form16_part_a.tds_q2": case.form16_part_a.tds_q2,
        "form16_part_a.tds_q3": case.form16_part_a.tds_q3,
        "form16_part_a.tds_q4": case.form16_part_a.tds_q4,
        "form16_part_a.total_tds": case.form16_part_a.total_tds,
        "form16_part_b.basic_salary": case.form16_part_b.basic_salary,
        "form16_part_b.hra_received": case.form16_part_b.hra_received,
        "form16_part_b.gross_salary": case.form16_part_b.gross_salary,
        "form16_part_b.net_taxable_salary": case.form16_part_b.net_taxable_salary,
    }
    score = Task1Grader().grade(fields, case)
    assert score >= 0.95  # perfect field extraction → high score


def test_task1_zero_agent():
    case = make_case()
    score = Task1Grader().grade({}, case)
    assert 0.0 < score < 1.0


def test_task1_score_in_range():
    case = make_case()
    partial = {"form16_part_a.gross_salary": case.form16_part_a.gross_salary}
    score = Task1Grader().grade(partial, case)
    assert 0.0 < score < 1.0


# --- Task 2 Grader ---

def test_task2_perfect_agent():
    case = make_case_task2()
    true_deductions = tax_engine.compute_all_deductions(case)
    true_regime = tax_engine.optimal_regime(case)
    true_tax = tax_engine.compute_tax(case, true_regime)
    sub = SubmitReturnAction(
        total_income=case.form16_part_b.gross_salary,
        total_deductions=sum(true_deductions.values()),
        taxable_income=case.form16_part_b.gross_salary - sum(true_deductions.values()),
        tax_payable=true_tax,
        regime_selected=true_regime,
        deductions_claimed=true_deductions,
    )
    score = Task2Grader().grade(sub, case)
    assert score >= 0.90


def test_task2_wrong_regime():
    case = make_case_task2()
    true_regime = tax_engine.optimal_regime(case)
    wrong_regime = "new" if true_regime == "old" else "old"
    true_deductions = tax_engine.compute_all_deductions(case)
    true_tax = tax_engine.compute_tax(case, wrong_regime)
    sub = SubmitReturnAction(
        total_income=case.form16_part_b.gross_salary,
        total_deductions=sum(true_deductions.values()),
        taxable_income=case.form16_part_b.gross_salary - sum(true_deductions.values()),
        tax_payable=true_tax,
        regime_selected=wrong_regime,
        deductions_claimed=true_deductions,
    )
    score = Task2Grader().grade(sub, case)
    # regime_score = 0, max possible = 0.65
    assert score <= 0.66


def test_task2_score_in_range():
    case = make_case_task2()
    sub = SubmitReturnAction(
        total_income=0, total_deductions=0, taxable_income=0,
        tax_payable=0, regime_selected="new",
    )
    score = Task2Grader().grade(sub, case)
    assert 0.0 < score < 1.0


# --- Task 3 Grader ---

def test_task3_perfect_agent():
    case = make_case_task3()
    agent_assets = []
    for asset in case.assets:
        agent_assets.append(ComputeCapGainsAction(
            asset_id=asset.asset_id,
            indexed_cost=tax_engine.indexed_cost(asset),
            gain=tax_engine.capital_gain(asset),
            tax_rule_applied=tax_engine.applicable_rule(asset),
        ))
    score = Task3Grader().grade(agent_assets, case)
    assert score >= 0.95


def test_task3_zero_agent():
    case = make_case_task3()
    score = Task3Grader().grade([], case)
    assert 0.0 < score < 1.0


def test_task3_wrong_rule_finance_act():
    """Agent ignores Finance Act 2023 → wrong rule for debt MF post-2023."""
    case = make_case_task3()
    agent_assets = []
    for asset in case.assets:
        true_rule = tax_engine.applicable_rule(asset)
        # Deliberately use wrong rule for debt_mf
        if asset.asset_type == "debt_mf":
            wrong_rule = "ltcg_debt_indexed"  # ignores Finance Act 2023
        else:
            wrong_rule = true_rule
        agent_assets.append(ComputeCapGainsAction(
            asset_id=asset.asset_id,
            indexed_cost=tax_engine.indexed_cost(asset),
            gain=tax_engine.capital_gain(asset),
            tax_rule_applied=wrong_rule,
        ))
    score = Task3Grader().grade(agent_assets, case)
    perfect_score = Task3Grader().grade([
        ComputeCapGainsAction(
            asset_id=a.asset_id,
            indexed_cost=tax_engine.indexed_cost(a),
            gain=tax_engine.capital_gain(a),
            tax_rule_applied=tax_engine.applicable_rule(a),
        ) for a in case.assets
    ], case)
    assert score < perfect_score  # wrong rule must reduce score


def test_task3_score_in_range():
    case = make_case_task3()
    agent_assets = [
        ComputeCapGainsAction(
            asset_id=asset.asset_id,
            indexed_cost=0.0, gain=0.0, tax_rule_applied="stcg_slab",
        )
        for asset in case.assets
    ]
    score = Task3Grader().grade(agent_assets, case)
    assert 0.0 < score < 1.0
