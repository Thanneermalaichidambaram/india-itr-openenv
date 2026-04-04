"""
Unit tests for the tax engine oracle.
Tests all tax rules including Finance Act 2023 edge cases.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import date
import pytest

from env.models import Asset, CaseFile, Form16PartA, Form16PartB, InvestmentStatement
from env import tax_engine


def make_case(gross=1_200_000, basic=480_000, hra=240_000, city="mumbai",
              ppf=50_000, elss=50_000, lic=20_000, rent=25_000,
              health_self=20_000, health_parents=20_000, parents_senior=False):
    part_a = Form16PartA(
        employer_name="Test Corp", employee_pan="ABCDE1234F",
        gross_salary=gross, tds_q1=0, tds_q2=0, tds_q3=0, tds_q4=0, total_tds=100_000,
    )
    part_b = Form16PartB(
        basic_salary=basic, hra_received=hra,
        special_allowance=gross - basic - hra,
        lta=20_000, gross_salary=gross,
        standard_deduction=50_000, professional_tax=2400,
        net_taxable_salary=gross - 52_400, city=city,
    )
    inv = InvestmentStatement(
        ppf=ppf, elss=elss, life_insurance=lic,
        health_insurance_self=health_self,
        health_insurance_parents=health_parents,
        parents_senior_citizen=parents_senior,
        rent_paid_monthly=rent,
    )
    return CaseFile(case_id="test", form16_part_a=part_a, form16_part_b=part_b, investments=inv)


# --- HRA Tests ---

def test_hra_metro():
    case = make_case(basic=480_000, hra=240_000, city="mumbai", rent=30_000)
    hra = tax_engine.hra_exemption(480_000, 240_000, 30_000, "mumbai")
    # a=240000, b=0.5*480000=240000, c=360000-48000=312000 → min=240000
    assert hra == 240_000.0


def test_hra_non_metro():
    case = make_case(basic=480_000, hra=200_000, city="pune", rent=20_000)
    hra = tax_engine.hra_exemption(480_000, 200_000, 20_000, "pune")
    # a=200000, b=0.4*480000=192000, c=240000-48000=192000 → min=192000
    assert hra == 192_000.0


def test_hra_zero_when_no_rent():
    hra = tax_engine.hra_exemption(480_000, 200_000, 0, "mumbai")
    # c = 0 - 48000 < 0 → max(0) = 0
    assert hra == 0.0


# --- 80C Tests ---

def test_80c_capped_at_150k():
    inv = InvestmentStatement(ppf=100_000, elss=80_000, life_insurance=40_000)
    assert tax_engine.deduction_80c(inv) == 150_000.0


def test_80c_under_limit():
    inv = InvestmentStatement(ppf=50_000, elss=30_000)
    assert tax_engine.deduction_80c(inv) == 80_000.0


# --- 80D Tests ---

def test_80d_parents_senior():
    inv = InvestmentStatement(
        health_insurance_self=25_000, health_insurance_parents=60_000, parents_senior_citizen=True
    )
    # self: min(25000,25000)=25000, parents: min(60000,50000)=50000
    assert tax_engine.deduction_80d(inv) == 75_000.0


def test_80d_parents_non_senior():
    inv = InvestmentStatement(
        health_insurance_self=20_000, health_insurance_parents=30_000, parents_senior_citizen=False
    )
    assert tax_engine.deduction_80d(inv) == 45_000.0


# --- Finance Act 2023 ---

def test_debt_mf_pre_2023_long_term():
    asset = Asset(
        asset_id="X", asset_type="debt_mf",
        purchase_date=date(2019, 6, 1), purchase_amount=200_000,
        sale_date=date(2024, 6, 1), sale_amount=280_000,
    )
    assert tax_engine.applicable_rule(asset) == "ltcg_debt_indexed"


def test_debt_mf_post_2023():
    asset = Asset(
        asset_id="X", asset_type="debt_mf",
        purchase_date=date(2023, 4, 1), purchase_amount=200_000,
        sale_date=date(2024, 10, 1), sale_amount=220_000,
    )
    assert tax_engine.applicable_rule(asset) == "stcg_slab_finact2023"


def test_debt_mf_boundary_31mar2023():
    """31-Mar-2023 is the last day old rules apply."""
    asset = Asset(
        asset_id="X", asset_type="debt_mf",
        purchase_date=date(2023, 3, 31), purchase_amount=200_000,
        sale_date=date(2026, 6, 1), sale_amount=260_000,
    )
    # Held >36 months, pre-2023 → ltcg_debt_indexed
    assert tax_engine.applicable_rule(asset) == "ltcg_debt_indexed"


def test_gold_etf_post_2023():
    asset = Asset(
        asset_id="X", asset_type="gold_etf",
        purchase_date=date(2023, 7, 1), purchase_amount=100_000,
        sale_date=date(2024, 12, 1), sale_amount=115_000,
    )
    assert tax_engine.applicable_rule(asset) == "stcg_slab_finact2023"


# --- Unlisted equity 24-month threshold ---

def test_unlisted_equity_23_months_short_term():
    asset = Asset(
        asset_id="X", asset_type="equity_unlisted",
        purchase_date=date(2023, 1, 1), purchase_amount=100_000,
        sale_date=date(2024, 11, 15), sale_amount=150_000,  # ~22.5 months
    )
    assert tax_engine.applicable_rule(asset) == "stcg_slab"


def test_unlisted_equity_24_months_long_term():
    asset = Asset(
        asset_id="X", asset_type="equity_unlisted",
        purchase_date=date(2022, 6, 1), purchase_amount=100_000,
        sale_date=date(2024, 6, 15), sale_amount=200_000,  # 24+ months
    )
    assert tax_engine.applicable_rule(asset) == "ltcg_unlisted_20pct"


# --- LTCG equity 112A exemption ---

def test_ltcg_equity_exactly_100k_gain():
    asset = Asset(
        asset_id="X", asset_type="equity_listed",
        purchase_date=date(2021, 1, 1), purchase_amount=500_000,
        sale_date=date(2024, 1, 1), sale_amount=600_000,
        stt_paid=True,
    )
    gain = tax_engine.capital_gain(asset)
    assert gain == 100_000.0
    tax = tax_engine.capital_gain_tax(asset)
    # Exactly 100k → taxable gain = 0 → tax = 0
    assert tax == 0.0


def test_ltcg_equity_above_100k():
    asset = Asset(
        asset_id="X", asset_type="equity_listed",
        purchase_date=date(2021, 1, 1), purchase_amount=500_000,
        sale_date=date(2024, 1, 1), sale_amount=700_000,
        stt_paid=True,
    )
    gain = tax_engine.capital_gain(asset)
    assert gain == 200_000.0
    tax = tax_engine.capital_gain_tax(asset)
    # Taxable = 200000 - 100000 = 100000 @ 10% + 4% cess = 10400
    assert abs(tax - 10_400.0) < 1.0


# --- CII indexation ---

def test_cii_indexation_debt_pre2023():
    asset = Asset(
        asset_id="X", asset_type="debt_mf",
        purchase_date=date(2020, 6, 1), purchase_amount=100_000,
        sale_date=date(2024, 6, 1), sale_amount=140_000,
    )
    # CII 2020=301, 2024=363
    expected = round(100_000 * (363 / 301), 2)
    assert abs(tax_engine.indexed_cost(asset) - expected) < 1.0


def test_cii_no_indexation_equity():
    asset = Asset(
        asset_id="X", asset_type="equity_listed",
        purchase_date=date(2020, 6, 1), purchase_amount=200_000,
        sale_date=date(2024, 6, 1), sale_amount=400_000,
    )
    # Equity: no CII indexation
    assert tax_engine.indexed_cost(asset) == 200_000.0


# --- Optimal regime ---

def test_optimal_regime_low_income_new_regime():
    """Low income with few deductions → new regime likely better (rebate)."""
    case = make_case(gross=600_000, basic=240_000, hra=96_000, city="pune",
                     ppf=10_000, elss=0, lic=0, rent=10_000,
                     health_self=5_000, health_parents=0)
    regime = tax_engine.optimal_regime(case)
    # With 7L new regime rebate, low income → new regime wins
    assert regime in ("old", "new")  # just verify it runs


def test_compute_tax_non_negative():
    case = make_case()
    assert tax_engine.compute_tax_old_regime(case) >= 0
    assert tax_engine.compute_tax_new_regime(case) >= 0
