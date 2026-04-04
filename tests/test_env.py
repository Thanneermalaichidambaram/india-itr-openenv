"""
Tests for IndiaITREnvironment: reset, step, state, episode boundaries.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from env.environment import IndiaITREnvironment
from env.models import ReadSectionAction, ExtractFieldAction, SubmitReturnAction


@pytest.mark.parametrize("task_id", ["task1_parse", "task2_deduct", "task3_capgains"])
def test_reset_produces_clean_state(task_id):
    env = IndiaITREnvironment(task_id=task_id, seed=42)
    obs = env.reset()
    assert obs.turn == 0
    assert obs.done is False
    assert obs.cumulative_reward == 0.0
    assert obs.sections_read == []
    assert obs.extracted_fields == {}


def test_step_advances_turn():
    env = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env.reset()
    obs = env.step(ReadSectionAction(section="form16_part_a"))
    assert obs.turn == 1


def test_read_section_reveals_data():
    env = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env.reset()
    obs = env.step(ReadSectionAction(section="form16_part_a"))
    assert "form16_part_a" in obs.sections_read
    assert "form16_part_a" in obs.visible_data
    assert "gross_salary" in obs.visible_data["form16_part_a"]


def test_extract_field_reward():
    env = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env.reset()
    env.step(ReadSectionAction(section="form16_part_a"))
    # Get true value from case
    true_val = env._case.form16_part_a.gross_salary
    obs = env.step(ExtractFieldAction(
        field_name="form16_part_a.gross_salary",
        extracted_value=true_val,
    ))
    assert obs.reward >= 0.05  # at least correct field reward


def test_max_turns_terminates_episode():
    env = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env.reset()
    obs = None
    for _ in range(20):
        obs = env.step(ReadSectionAction(section="form16_part_a"))
        if obs.done:
            break
    assert obs.done is True
    assert obs.turn == 20


def test_submit_terminates_episode():
    env = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env.reset()
    obs = env.step(SubmitReturnAction(
        total_income=1_000_000, total_deductions=50_000,
        taxable_income=950_000, tax_payable=100_000,
        regime_selected="new",
    ))
    assert obs.done is True


def test_state_is_idempotent():
    env = IndiaITREnvironment(task_id="task2_deduct", seed=42)
    env.reset()
    env.step(ReadSectionAction(section="form16_part_b"))
    s1 = env.state
    s2 = env.state
    assert s1.step_count == s2.step_count


def test_same_seed_same_case():
    env1 = IndiaITREnvironment(task_id="task1_parse", seed=42)
    env2 = IndiaITREnvironment(task_id="task1_parse", seed=42)
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert env1._case.form16_part_a.gross_salary == env2._case.form16_part_a.gross_salary


def test_different_seeds_different_cases():
    env1 = IndiaITREnvironment(task_id="task1_parse", seed=1)
    env2 = IndiaITREnvironment(task_id="task1_parse", seed=99)
    env1.reset()
    env2.reset()
    # With overwhelming probability, different seeds → different gross salaries
    assert env1._case.form16_part_a.gross_salary != env2._case.form16_part_a.gross_salary


def test_task3_has_4_assets():
    env = IndiaITREnvironment(task_id="task3_capgains", seed=42)
    env.reset()
    assert len(env._case.assets) == 4
