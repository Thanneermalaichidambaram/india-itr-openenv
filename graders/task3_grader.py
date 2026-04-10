"""
Task 3 Grader — Capital gains with CII indexation + Finance Act 2023.
Score = 0.70 × mean(per_asset_score) + 0.30 × rule_application_score
"""
from __future__ import annotations

from typing import Dict, List

from env import tax_engine
from env.models import CaseFile, ComputeCapGainsAction


class Task3Grader:
    def grade(
        self,
        agent_assets: List[ComputeCapGainsAction],
        case: CaseFile,
    ) -> float:
        if not case.assets:
            return 0.001

        per_asset_scores = []
        rule_correct_count = 0

        for asset in case.assets:
            agent = next(
                (a for a in agent_assets if a.asset_id == asset.asset_id), None
            )
            if agent is None:
                per_asset_scores.append(0.0)
                continue

            true_cost = tax_engine.indexed_cost(asset)
            true_gain = tax_engine.capital_gain(asset)
            true_rule = tax_engine.applicable_rule(asset)

            # Cost accuracy (35% of asset score)
            cost_err = abs(agent.indexed_cost - true_cost) / max(abs(true_cost), 1.0)
            cost_score = max(0.0, 1.0 - cost_err) * 0.35

            # Gain accuracy (35% of asset score)
            gain_err = abs(agent.gain - true_gain) / max(abs(true_gain), 1.0)
            gain_score = max(0.0, 1.0 - gain_err) * 0.35

            # Rule correct (30% of asset score)
            rule_ok = agent.tax_rule_applied == true_rule
            rule_score = 0.30 if rule_ok else 0.0
            if rule_ok:
                rule_correct_count += 1

            asset_score = cost_score + gain_score + rule_score
            per_asset_scores.append(round(asset_score, 4))

        mean_asset_score = sum(per_asset_scores) / len(per_asset_scores)
        rule_application_score = rule_correct_count / len(case.assets)

        final = 0.70 * mean_asset_score + 0.30 * rule_application_score
        return round(min(max(final, 0.001), 0.999), 3)
