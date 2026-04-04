"""
Tax Engine — deterministic oracle for Indian ITR FY2024-25.
Never exposed to the agent. Only graders call this.
All amounts in INR.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Literal, Tuple

from .models import Asset, CaseFile, InvestmentStatement

# ---------------------------------------------------------------------------
# Constants — FY2024-25
# ---------------------------------------------------------------------------

# Cost Inflation Index (official IT dept data)
CII: Dict[int, int] = {
    2001: 100, 2002: 105, 2003: 109, 2004: 113, 2005: 117,
    2006: 122, 2007: 129, 2008: 137, 2009: 148, 2010: 167,
    2011: 184, 2012: 200, 2013: 220, 2014: 240, 2015: 254,
    2016: 264, 2017: 272, 2018: 280, 2019: 289, 2020: 301,
    2021: 317, 2022: 331, 2023: 348, 2024: 363, 2025: 376,
}

# Metro cities for HRA (50% of basic); non-metro = 40%
METRO_CITIES = {
    "mumbai", "delhi", "kolkata", "chennai",
    "bangalore", "bengaluru", "hyderabad",
}

# New regime tax slabs FY2024-25 (post-July 2024 Budget)
NEW_REGIME_SLABS = [
    (300_000, 0.00),
    (700_000, 0.05),
    (1_000_000, 0.10),
    (1_200_000, 0.15),
    (1_500_000, 0.20),
    (float("inf"), 0.30),
]

# Old regime tax slabs (below 60 years)
OLD_REGIME_SLABS = [
    (250_000, 0.00),
    (500_000, 0.05),
    (1_000_000, 0.20),
    (float("inf"), 0.30),
]

# Finance Act 2023: debt MF / gold ETF bought on/after this date → no indexation
FINANCE_ACT_2023_DATE = date(2023, 4, 1)

# LTCG equity exemption threshold
LTCG_EQUITY_EXEMPTION = 100_000


def _cii_for_year(d: date) -> int:
    """Return CII for the financial year containing date d."""
    fy_start = d.year if d.month >= 4 else d.year - 1
    return CII.get(fy_start, CII[2024])


def _slab_tax(income: float, slabs: list) -> float:
    tax = 0.0
    prev = 0.0
    for ceiling, rate in slabs:
        if income <= prev:
            break
        taxable = min(income, ceiling) - prev
        tax += taxable * rate
        prev = ceiling
    return tax


# ---------------------------------------------------------------------------
# HRA Exemption
# ---------------------------------------------------------------------------

def hra_exemption(
    basic: float,
    hra_received: float,
    rent_paid_monthly: float,
    city: str,
) -> float:
    """Compute HRA exemption: minimum of three conditions."""
    annual_rent = rent_paid_monthly * 12
    metro = city.lower() in METRO_CITIES
    basic_pct = 0.50 if metro else 0.40
    a = hra_received
    b = basic_pct * basic
    c = max(annual_rent - 0.10 * basic, 0)
    return round(min(a, b, c), 2)


# ---------------------------------------------------------------------------
# Deductions
# ---------------------------------------------------------------------------

def deduction_80c(inv: InvestmentStatement) -> float:
    total = (
        inv.ppf + inv.elss + inv.life_insurance +
        inv.home_loan_principal + inv.school_fees + inv.nsc
    )
    return round(min(total, 150_000), 2)


def deduction_80d(inv: InvestmentStatement) -> float:
    self_limit = 25_000
    parents_limit = 50_000 if inv.parents_senior_citizen else 25_000
    return round(
        min(inv.health_insurance_self, self_limit) +
        min(inv.health_insurance_parents, parents_limit),
        2,
    )


def deduction_80tta(inv: InvestmentStatement) -> float:
    return round(min(inv.savings_account_interest, 10_000), 2)


def deduction_24b(case: CaseFile) -> float:
    """Home loan interest — capped at 2L for self-occupied."""
    # We store it in metadata if present; default 0
    return round(min(case.investments.home_loan_principal * 0, 200_000), 2)


def compute_all_deductions(case: CaseFile) -> Dict[str, float]:
    inv = case.investments
    fb = case.form16_part_b
    deductions: Dict[str, float] = {}

    deductions["80C"] = deduction_80c(inv)
    deductions["80D"] = deduction_80d(inv)
    deductions["80TTA"] = deduction_80tta(inv)
    deductions["standard_deduction"] = fb.standard_deduction

    hra = hra_exemption(
        basic=fb.basic_salary,
        hra_received=fb.hra_received,
        rent_paid_monthly=inv.rent_paid_monthly,
        city=fb.city,
    )
    deductions["HRA"] = hra

    return deductions


# ---------------------------------------------------------------------------
# Regime comparison
# ---------------------------------------------------------------------------

def compute_tax_old_regime(case: CaseFile) -> float:
    deductions = compute_all_deductions(case)
    gross = case.form16_part_b.net_taxable_salary
    total_ded = sum(deductions.values())
    taxable = max(gross - total_ded + deductions["standard_deduction"], 0)
    # standard_deduction already in net_taxable_salary, so add back before subtracting
    taxable = max(case.form16_part_b.gross_salary - sum(
        v for k, v in deductions.items() if k != "standard_deduction"
    ) - deductions["standard_deduction"], 0)

    tax = _slab_tax(taxable, OLD_REGIME_SLABS)
    # 87A rebate: if taxable ≤ 5L, rebate up to 12500
    if taxable <= 500_000:
        tax = max(tax - 12_500, 0)
    cess = tax * 0.04
    return round(tax + cess, 2)


def compute_tax_new_regime(case: CaseFile) -> float:
    # New regime: only standard deduction allowed
    taxable = max(
        case.form16_part_b.gross_salary - case.form16_part_b.standard_deduction, 0
    )
    tax = _slab_tax(taxable, NEW_REGIME_SLABS)
    # 87A rebate: if taxable ≤ 7L, full rebate
    if taxable <= 700_000:
        tax = 0.0
    cess = tax * 0.04
    return round(tax + cess, 2)


def optimal_regime(case: CaseFile) -> Literal["old", "new"]:
    old = compute_tax_old_regime(case)
    new = compute_tax_new_regime(case)
    return "old" if old < new else "new"


def compute_tax(case: CaseFile, regime: Literal["old", "new"]) -> float:
    if regime == "old":
        return compute_tax_old_regime(case)
    return compute_tax_new_regime(case)


# ---------------------------------------------------------------------------
# Capital gains
# ---------------------------------------------------------------------------

def _holding_months(asset: Asset) -> int:
    delta = asset.sale_date - asset.purchase_date
    return int(delta.days / 30.44)


def _is_long_term(asset: Asset) -> bool:
    months = _holding_months(asset)
    if asset.asset_type in ("equity_listed", "equity_unlisted"):
        threshold = 12 if asset.asset_type == "equity_listed" else 24
        return months >= threshold
    if asset.asset_type in ("debt_mf", "gold_etf"):
        # Finance Act 2023: always STCG if purchased on/after 1 Apr 2023
        if asset.purchase_date >= FINANCE_ACT_2023_DATE:
            return False
        return months >= 36
    if asset.asset_type == "property":
        return months >= 24
    return months >= 36


def indexed_cost(asset: Asset) -> float:
    """
    Compute indexed cost of acquisition.
    Finance Act 2023: debt_mf / gold_etf bought on/after 1-Apr-2023 → no indexation.
    Equity: no indexation (112A).
    """
    if asset.asset_type in ("equity_listed", "equity_unlisted"):
        # Equity: no CII indexation; grandfathering for pre-31-Jan-2018
        if asset.purchase_date < date(2018, 1, 31) and asset.jan31_2018_nav:
            return round(max(asset.purchase_amount, asset.jan31_2018_nav), 2)
        return round(asset.purchase_amount, 2)

    if asset.asset_type in ("debt_mf", "gold_etf"):
        if asset.purchase_date >= FINANCE_ACT_2023_DATE:
            return round(asset.purchase_amount, 2)  # no indexation
        # Old rules: CII indexation allowed
        cii_buy = _cii_for_year(asset.purchase_date)
        cii_sell = _cii_for_year(asset.sale_date)
        return round(asset.purchase_amount * (cii_sell / cii_buy), 2)

    # property: CII indexation
    cii_buy = _cii_for_year(asset.purchase_date)
    cii_sell = _cii_for_year(asset.sale_date)
    return round(asset.purchase_amount * (cii_sell / cii_buy), 2)


def capital_gain(asset: Asset) -> float:
    cost = indexed_cost(asset)
    return round(asset.sale_amount - cost, 2)


def applicable_rule(asset: Asset) -> str:
    """Return the tax rule string that applies to this asset."""
    lt = _is_long_term(asset)

    if asset.asset_type == "equity_listed":
        return "ltcg_equity_112a" if lt else "stcg_equity_111a"

    if asset.asset_type == "equity_unlisted":
        return "ltcg_unlisted_20pct" if lt else "stcg_slab"

    if asset.asset_type in ("debt_mf", "gold_etf"):
        if asset.purchase_date >= FINANCE_ACT_2023_DATE:
            return "stcg_slab_finact2023"
        return "ltcg_debt_indexed" if lt else "stcg_slab"

    if asset.asset_type == "property":
        return "ltcg_property_indexed" if lt else "stcg_slab"

    return "stcg_slab"


def capital_gain_tax(asset: Asset, slab_rate: float = 0.30) -> float:
    """
    Compute capital gains tax.
    slab_rate used for STCG where income tax slab applies (default 30% for high income).
    """
    rule = applicable_rule(asset)
    gain = capital_gain(asset)
    if gain <= 0:
        return 0.0

    if rule == "ltcg_equity_112a":
        taxable = max(gain - LTCG_EQUITY_EXEMPTION, 0)
        tax = taxable * 0.10
    elif rule == "stcg_equity_111a":
        tax = gain * 0.15
    elif rule == "ltcg_unlisted_20pct":
        tax = gain * 0.20
    elif rule == "ltcg_debt_indexed":
        tax = gain * 0.20
    elif rule == "ltcg_property_indexed":
        tax = gain * 0.20
    else:
        # stcg_slab or stcg_slab_finact2023
        tax = gain * slab_rate

    cess = tax * 0.04
    return round(tax + cess, 2)
