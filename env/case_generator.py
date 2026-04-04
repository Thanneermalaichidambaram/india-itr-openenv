"""
Seeded synthetic case generator.
Same seed always produces the same CaseFile — fully deterministic.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import List, Literal

import numpy as np

from .models import Asset, CaseFile, Form16PartA, Form16PartB, InvestmentStatement

METRO_CITIES = ["mumbai", "delhi", "bengaluru", "hyderabad", "chennai", "kolkata"]
NON_METRO_CITIES = ["pune", "ahmedabad", "jaipur", "lucknow", "bhopal", "nagpur", "surat"]

EMPLOYER_NAMES = [
    "Infosys Ltd", "TCS Ltd", "Wipro Ltd", "HCL Technologies",
    "Tech Mahindra", "HDFC Bank", "ICICI Bank", "Reliance Industries",
    "Bajaj Finance", "Larsen & Toubro",
]


def _random_pan(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = "".join(rng.choice(list(letters)) for _ in range(5))
    digits = "".join(str(d) for d in rng.integers(0, 9, size=4))
    last = rng.choice(list(letters))
    return p + digits + last


def _random_date_between(rng: np.random.Generator, start: date, end: date) -> date:
    delta = (end - start).days
    offset = int(rng.integers(0, max(delta, 1)))
    return start + timedelta(days=offset)


class CaseGenerator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate(self, task_id: str) -> CaseFile:
        if task_id == "task1_parse":
            return self._generate_task1()
        if task_id == "task2_deduct":
            return self._generate_task2()
        if task_id == "task3_capgains":
            return self._generate_task3()
        raise ValueError(f"Unknown task_id: {task_id}")

    # ------------------------------------------------------------------
    # Task 1 — Form 16 parsing (mismatch scenarios)
    # ------------------------------------------------------------------

    def _generate_task1(self) -> CaseFile:
        scenario = self.rng.choice(["clean", "tds_mismatch", "gross_mismatch"], p=[0.60, 0.20, 0.20])
        gross = round(float(self.rng.uniform(600_000, 2_500_000)), 2)
        basic = round(gross * float(self.rng.uniform(0.35, 0.45)), 2)
        hra = round(gross * float(self.rng.uniform(0.20, 0.30)), 2)
        special = round(gross - basic - hra, 2)
        prof_tax = 2400.0
        std_ded = 50_000.0
        net_taxable = round(gross - std_ded - prof_tax, 2)

        # Quarterly TDS — roughly equal splits with slight variation
        tds_annual = round(gross * float(self.rng.uniform(0.10, 0.25)), 2)
        splits = self.rng.dirichlet([1, 1, 1, 1])
        tds_q = [round(tds_annual * float(s), 2) for s in splits]
        tds_q[3] = round(tds_annual - sum(tds_q[:3]), 2)  # ensure exact sum

        # Mismatch scenarios
        part_a_gross = gross
        total_tds_in_a = tds_annual
        if scenario == "tds_mismatch":
            total_tds_in_a = round(tds_annual + float(self.rng.choice([-500, -1000, 500, 1000])), 2)
        elif scenario == "gross_mismatch":
            part_a_gross = round(gross + float(self.rng.choice([-2000, -5000, 2000, 5000])), 2)

        city = str(self.rng.choice(METRO_CITIES + NON_METRO_CITIES))

        part_a = Form16PartA(
            employer_name=str(self.rng.choice(EMPLOYER_NAMES)),
            employee_pan=_random_pan(self.rng),
            gross_salary=part_a_gross,
            tds_q1=tds_q[0], tds_q2=tds_q[1],
            tds_q3=tds_q[2], tds_q4=tds_q[3],
            total_tds=total_tds_in_a,
        )
        part_b = Form16PartB(
            basic_salary=basic,
            hra_received=hra,
            special_allowance=special,
            lta=round(float(self.rng.uniform(10_000, 50_000)), 2),
            gross_salary=gross,
            standard_deduction=std_ded,
            professional_tax=prof_tax,
            net_taxable_salary=net_taxable,
            city=city,
        )
        inv = InvestmentStatement()
        return CaseFile(
            case_id=str(uuid.uuid4()),
            form16_part_a=part_a,
            form16_part_b=part_b,
            investments=inv,
        )

    # ------------------------------------------------------------------
    # Task 2 — Deductions + regime selection
    # ------------------------------------------------------------------

    def _generate_task2(self) -> CaseFile:
        gross = round(float(self.rng.uniform(800_000, 3_000_000)), 2)
        basic = round(gross * float(self.rng.uniform(0.40, 0.50)), 2)
        hra = round(gross * float(self.rng.uniform(0.20, 0.30)), 2)
        special = round(gross - basic - hra, 2)
        city = str(self.rng.choice(METRO_CITIES + NON_METRO_CITIES))

        part_a = Form16PartA(
            employer_name=str(self.rng.choice(EMPLOYER_NAMES)),
            employee_pan=_random_pan(self.rng),
            gross_salary=gross,
            tds_q1=0, tds_q2=0, tds_q3=0, tds_q4=0,
            total_tds=round(gross * 0.15, 2),
        )
        part_b = Form16PartB(
            basic_salary=basic,
            hra_received=hra,
            special_allowance=special,
            lta=round(float(self.rng.uniform(10_000, 40_000)), 2),
            gross_salary=gross,
            standard_deduction=50_000.0,
            professional_tax=2400.0,
            net_taxable_salary=round(gross - 52_400, 2),
            city=city,
        )

        # Investments with realistic amounts
        ppf = round(float(self.rng.uniform(0, 80_000)), 2)
        elss = round(float(self.rng.uniform(0, 60_000)), 2)
        lic = round(float(self.rng.uniform(0, 40_000)), 2)
        health_self = round(float(self.rng.uniform(10_000, 25_000)), 2)
        parents_senior = bool(self.rng.choice([True, False]))
        health_parents = round(float(self.rng.uniform(15_000, 50_000)), 2)
        rent_monthly = round(float(self.rng.uniform(10_000, 60_000)), 2)
        savings_interest = round(float(self.rng.uniform(0, 15_000)), 2)

        inv = InvestmentStatement(
            ppf=ppf, elss=elss, life_insurance=lic,
            health_insurance_self=health_self,
            health_insurance_parents=health_parents,
            parents_senior_citizen=parents_senior,
            rent_paid_monthly=rent_monthly,
            savings_account_interest=savings_interest,
        )

        return CaseFile(
            case_id=str(uuid.uuid4()),
            form16_part_a=part_a,
            form16_part_b=part_b,
            investments=inv,
        )

    # ------------------------------------------------------------------
    # Task 3 — Capital gains
    # ------------------------------------------------------------------

    def _generate_task3(self) -> CaseFile:
        gross = round(float(self.rng.uniform(1_200_000, 4_000_000)), 2)
        basic = round(gross * 0.40, 2)
        city = str(self.rng.choice(METRO_CITIES))

        part_a = Form16PartA(
            employer_name=str(self.rng.choice(EMPLOYER_NAMES)),
            employee_pan=_random_pan(self.rng),
            gross_salary=gross,
            tds_q1=0, tds_q2=0, tds_q3=0, tds_q4=0,
            total_tds=round(gross * 0.20, 2),
        )
        part_b = Form16PartB(
            basic_salary=basic,
            hra_received=round(basic * 0.40, 2),
            special_allowance=round(gross - basic - basic * 0.40, 2),
            lta=20_000.0,
            gross_salary=gross,
            standard_deduction=50_000.0,
            professional_tax=2400.0,
            net_taxable_salary=round(gross - 52_400, 2),
            city=city,
        )

        assets = self._generate_assets()
        inv = InvestmentStatement(
            ppf=50_000, elss=50_000, life_insurance=20_000,
            health_insurance_self=25_000,
        )

        return CaseFile(
            case_id=str(uuid.uuid4()),
            form16_part_a=part_a,
            form16_part_b=part_b,
            investments=inv,
            assets=assets,
        )

    def _generate_assets(self) -> List[Asset]:
        assets = []
        sale_base = date(2024, 12, 1)

        # 1. Listed equity (LTCG 112A)
        buy_date = _random_date_between(self.rng, date(2020, 4, 1), date(2023, 3, 31))
        buy_price = round(float(self.rng.uniform(200_000, 800_000)), 2)
        jan31nav = round(buy_price * float(self.rng.uniform(1.1, 1.4)), 2) if buy_date < date(2018, 1, 31) else None
        assets.append(Asset(
            asset_id="A001",
            asset_type="equity_listed",
            purchase_date=buy_date,
            purchase_amount=buy_price,
            sale_date=sale_base,
            sale_amount=round(buy_price * float(self.rng.uniform(1.3, 2.5)), 2),
            stt_paid=True,
            jan31_2018_nav=jan31nav,
        ))

        # 2. Debt MF — 40% pre-2023, 20% post-2023 (Finance Act trap), 40% old short term
        scenario = self.rng.choice(["pre2023_lt", "post2023", "pre2023_st"], p=[0.40, 0.20, 0.40])
        if scenario == "pre2023_lt":
            buy_date = _random_date_between(self.rng, date(2019, 4, 1), date(2022, 12, 31))
        elif scenario == "post2023":
            buy_date = _random_date_between(self.rng, date(2023, 4, 1), date(2024, 3, 31))
        else:
            buy_date = _random_date_between(self.rng, date(2024, 1, 1), date(2024, 9, 30))
        buy_price = round(float(self.rng.uniform(100_000, 500_000)), 2)
        assets.append(Asset(
            asset_id="A002",
            asset_type="debt_mf",
            purchase_date=buy_date,
            purchase_amount=buy_price,
            sale_date=sale_base,
            sale_amount=round(buy_price * float(self.rng.uniform(1.05, 1.40)), 2),
        ))

        # 3. Unlisted equity (ESOP) — 30% hit 24-month threshold, 70% below
        months = int(self.rng.choice([20, 21, 22, 23, 24, 25, 30, 36], p=[0.10, 0.10, 0.15, 0.15, 0.10, 0.15, 0.15, 0.10]))
        buy_date = sale_base - timedelta(days=int(months * 30.44))
        buy_price = round(float(self.rng.uniform(50_000, 300_000)), 2)
        assets.append(Asset(
            asset_id="A003",
            asset_type="equity_unlisted",
            purchase_date=buy_date,
            purchase_amount=buy_price,
            sale_date=sale_base,
            sale_amount=round(buy_price * float(self.rng.uniform(1.2, 3.0)), 2),
        ))

        # 4. Gold ETF — always post-2023 (to test Finance Act rule)
        buy_date = _random_date_between(self.rng, date(2023, 6, 1), date(2024, 6, 30))
        buy_price = round(float(self.rng.uniform(80_000, 300_000)), 2)
        assets.append(Asset(
            asset_id="A004",
            asset_type="gold_etf",
            purchase_date=buy_date,
            purchase_amount=buy_price,
            sale_date=sale_base,
            sale_amount=round(buy_price * float(self.rng.uniform(1.05, 1.25)), 2),
        ))

        return assets
