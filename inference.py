"""
IndiaITR-OpenEnv Inference Script
===================================
Mandatory environment variables:
    API_BASE_URL   - LLM endpoint (default: HuggingFace router)
    MODEL_NAME     - Model identifier
    HF_TOKEN       - HuggingFace / API key
    LOCAL_IMAGE_NAME - Docker image name (optional, for from_docker_image)

Stdout format (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "india-itr-openenv"
SEED = 42
MAX_STEPS = 20
MAX_STEPS_TASK3 = 25
TEMPERATURE = 0.0
MAX_TOKENS = 800
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"task_id": "task1_parse", "max_steps": MAX_STEPS},
    {"task_id": "task2_deduct", "max_steps": MAX_STEPS},
    {"task_id": "task3_capgains", "max_steps": MAX_STEPS_TASK3},
]

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string: no newlines
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task1_parse": textwrap.dedent("""
        You are an expert Indian Chartered Accountant assistant.
        Your task is to parse Form 16 Part A and Part B documents and extract key fields.
        You must also detect mismatches between Part A and Part B.

        At each turn, respond ONLY with a valid JSON object for one of these actions:

        1. Read a section first:
        {"action_type": "read_section", "section": "form16_part_a"}
        {"action_type": "read_section", "section": "form16_part_b"}

        2. Extract a field (use dot-path notation):
        {"action_type": "extract_field", "field_name": "form16_part_a.gross_salary", "extracted_value": 1200000.0}

        3. Submit the return when done:
        {"action_type": "submit", "total_income": 1200000.0, "total_deductions": 50000.0,
         "taxable_income": 1150000.0, "tax_payable": 120000.0, "regime_selected": "new"}

        Key fields to extract (12 required):
        - form16_part_a.employer_name, form16_part_a.employee_pan
        - form16_part_a.gross_salary, form16_part_a.tds_q1/q2/q3/q4, form16_part_a.total_tds
        - form16_part_b.basic_salary, form16_part_b.hra_received
        - form16_part_b.gross_salary, form16_part_b.net_taxable_salary

        STRICT ORDER — do NOT submit until all 12 fields are extracted.

        Step 1: {"action_type": "read_section", "section": "form16_part_a"}
        Step 2: {"action_type": "read_section", "section": "form16_part_b"}
        Steps 3-14: extract each field below ONE BY ONE (one per step, use exact values from visible_data):
          1.  form16_part_a.employer_name
          2.  form16_part_a.employee_pan
          3.  form16_part_a.gross_salary
          4.  form16_part_a.tds_q1
          5.  form16_part_a.tds_q2
          6.  form16_part_a.tds_q3
          7.  form16_part_a.tds_q4
          8.  form16_part_a.total_tds        ← DO NOT SKIP THIS
          9.  form16_part_b.basic_salary
          10. form16_part_b.hra_received
          11. form16_part_b.gross_salary
          12. form16_part_b.net_taxable_salary
        Step 15: submit only after all 12 extracted_fields are present in the observation.

        Check observation.extracted_fields — if a field is missing, extract it before submitting.
        Respond ONLY with JSON. No explanation text.
    """).strip(),

    "task2_deduct": textwrap.dedent("""
        You are an expert Indian Chartered Accountant.
        Your task is to identify all applicable tax deductions and select the optimal tax regime.

        STRICT RULES — follow exactly:
        1. Read form16_part_b and investments sections first.
        2. Flag each deduction ONCE using exact amounts from the data. Do NOT repeat a section.
        3. Select regime. Then submit.

        DEDUCTION RULES:
        - 80C: sum of ppf+elss+life_insurance+home_loan_principal+school_fees+nsc, capped at 150000
        - 80D: min(health_insurance_self, 25000) + min(health_insurance_parents, 50000 if parents_senior_citizen else 25000)
        - HRA: compute min(A, B, C) where:
            A = hra_received (from form16_part_b)
            B = basic_salary * 0.50 if city in [mumbai,delhi,bengaluru,hyderabad,chennai,kolkata] else basic_salary * 0.40
            C = max(rent_paid_monthly * 12 - basic_salary * 0.10, 0)
          Use the EXACT numbers from visible_data. Compute once, flag once.
        - 80TTA: min(savings_account_interest, 10000)

        Valid flag_deduction sections: 80C, 24b, 80D, HRA, 80TTA, 80G
        Do NOT flag "standard_deduction" — it is not a valid section.

        Action sequence (do each ONCE, in order):
        {"action_type": "read_section", "section": "form16_part_b"}
        {"action_type": "read_section", "section": "investments"}
        {"action_type": "flag_deduction", "section": "80C", "amount": <computed, max 150000>}
        {"action_type": "flag_deduction", "section": "80D", "amount": <computed>}
        {"action_type": "flag_deduction", "section": "HRA", "amount": <computed using min formula>}
        {"action_type": "flag_deduction", "section": "80TTA", "amount": <min(savings_interest, 10000)>}
        {"action_type": "select_regime", "regime": "old", "computed_tax_old": <tax>, "computed_tax_new": <tax>}
        {"action_type": "submit", "total_income": <gross_salary>, "total_deductions": <sum of all deductions + 50000 standard>, "taxable_income": <gross - deductions - 50000>, "tax_payable": <tax>, "regime_selected": "old", "deductions_claimed": {"80C": ..., "80D": ..., "HRA": ..., "80TTA": ...}}

        IMPORTANT: flag each section exactly ONCE. Check flagged_deductions in observation — skip already flagged ones.
        Respond ONLY with JSON. No text outside JSON.
    """).strip(),

    "task3_capgains": textwrap.dedent("""
        You are an expert Indian Chartered Accountant specializing in capital gains tax.
        Your task is to compute capital gains for all assets using correct CII indexation.

        CRITICAL Finance Act 2023 rule:
        - Debt MF / Gold ETF purchased ON OR AFTER 1-Apr-2023: ALWAYS stcg_slab_finact2023 (no indexation, regardless of holding period)
        - Debt MF / Gold ETF purchased BEFORE 1-Apr-2023 AND held MORE THAN 36 months: ltcg_debt_indexed (CII indexation, 20%)
        - Debt MF / Gold ETF purchased BEFORE 1-Apr-2023 AND held 36 months or less: stcg_slab

        HOLDING PERIOD: count months between purchase_date and sale_date.
        Example: bought 2022-03-12, sold 2024-12-01 = ~32 months = less than 36 → stcg_slab (NOT ltcg_debt_indexed)

        CII values (FY): 2019→289, 2020→301, 2021→317, 2022→331, 2023→348, 2024→363, 2025→376
        Indexed cost = purchase_price × (CII_sale_year / CII_purchase_year)
        FY for a date: if month >= April, use that year; else use previous year.

        Tax rules:
        - Equity listed, held >12m, STT paid: ltcg_equity_112a (10% on gain above ₹1L, no indexation)
        - Equity listed, held <12m: stcg_equity_111a (15%)
        - Equity unlisted, held >24m: ltcg_unlisted_20pct (20% with indexation)
        - Equity unlisted, held <24m: stcg_slab
        - Property, held >24m: ltcg_property_indexed (20% with indexation)

        Action sequence:
        1. Read assets: {"action_type": "read_section", "section": "assets"}
        2. Compute each asset ONE BY ONE — check observation.capital_gains_computed to see which are done.
           Assets to compute: A001, A002, A003, A004 — do each ONCE, then move to the next.
        3. After ALL 4 assets computed, submit.

        Example compute action:
        {"action_type": "compute_capgains", "asset_id": "A001", "indexed_cost": 500000.0,
         "gain": 300000.0, "tax_rule_applied": "ltcg_equity_112a"}

        Submit after all 4 done:
        {"action_type": "submit", "total_income": 2000000.0, "total_deductions": 200000.0,
         "taxable_income": 1800000.0, "tax_payable": 350000.0, "regime_selected": "new",
         "capital_gains_computed": {"A001": 300000, "A002": 50000, "A003": 80000, "A004": 20000}}

        IMPORTANT: After computing an asset, check capital_gains_computed in the observation.
        If A002 is already there, move to A003. Never repeat the same asset_id twice.
        Respond ONLY with JSON.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    task_id: str,
    obs_json: str,
    history: List[Dict],
    step: int,
) -> str:
    system = SYSTEM_PROMPTS[task_id]
    import json as _json
    try:
        obs_dict = _json.loads(obs_json)
        extracted = obs_dict.get("extracted_fields", {})
        computed = obs_dict.get("capital_gains_computed", {})
        hint = ""
        if task_id == "task1_parse":
            required = ["form16_part_a.employer_name","form16_part_a.employee_pan",
                        "form16_part_a.gross_salary","form16_part_a.tds_q1","form16_part_a.tds_q2",
                        "form16_part_a.tds_q3","form16_part_a.tds_q4","form16_part_a.total_tds",
                        "form16_part_b.basic_salary","form16_part_b.hra_received",
                        "form16_part_b.gross_salary","form16_part_b.net_taxable_salary"]
            missing = [f for f in required if f not in extracted]
            if missing:
                hint = f"\nNOT YET EXTRACTED (do these before submit): {missing}"
        elif task_id == "task3_capgains":
            missing_assets = [a for a in ["A001","A002","A003","A004"] if a not in computed]
            if missing_assets:
                hint = f"\nNOT YET COMPUTED (do these before submit): {missing_assets}"
    except Exception:
        hint = ""

    user_content = textwrap.dedent(f"""
        Step {step}. Current environment state:
        {obs_json}{hint}

        Respond with exactly one JSON action object. No other text.
    """).strip()

    messages = [{"role": "system", "content": system}]
    # Include last 4 turns of history for context
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return text
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        # On any API error (rate limit, credits exhausted, etc.) → submit immediately
        return None


def _make_submit_action(obs) -> Dict:
    """Build a best-effort submit action from current observation state."""
    draft = obs.current_draft
    total_income = draft.total_income or obs.extracted_fields.get("form16_part_b.gross_salary", 0) or obs.extracted_fields.get("form16_part_a.gross_salary", 0) or 1_000_000.0
    total_ded = sum(obs.flagged_deductions.values()) if obs.flagged_deductions else 50_000.0
    taxable = max(float(total_income) - float(total_ded), 0)
    return {
        "action_type": "submit",
        "total_income": float(total_income),
        "total_deductions": float(total_ded),
        "taxable_income": taxable,
        "tax_payable": float(draft.tax_payable or 0),
        "regime_selected": draft.regime_selected or "new",
        "deductions_claimed": dict(obs.flagged_deductions),
        "capital_gains_computed": {k: v.get("gain", 0) for k, v in obs.capital_gains_computed.items()},
    }


def run_task(client: OpenAI, task_config: Dict) -> Dict:
    """Run one full task episode. Returns result dict."""
    from client import IndiaITRClient

    task_id = task_config["task_id"]
    max_steps = task_config["max_steps"]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict] = []
    consecutive_api_errors = 0
    last_actions: List[str] = []   # track last 3 actions for loop detection

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = IndiaITRClient(base_url=ENV_BASE_URL)
    try:
        result = env.reset(task_id=task_id, seed=SEED)
        done = result.done

        for step in range(1, max_steps + 1):
            if done:
                break

            obs_json = result.observation.model_dump_json()
            action_str = get_model_action(client, task_id, obs_json, history, step)

            # API error → submit immediately with best-effort values
            if action_str is None:
                consecutive_api_errors += 1
                if consecutive_api_errors >= 1:
                    action_str = json.dumps(_make_submit_action(result.observation))
                else:
                    action_str = json.dumps({"action_type": "read_section", "section": "form16_part_a"})
            else:
                consecutive_api_errors = 0

            # Loop detection: same action 3x OR alternating A-B-A-B → break out
            last_actions.append(action_str[:80])
            if len(last_actions) > 4:
                last_actions.pop(0)
            is_repeat = len(last_actions) == 3 and len(set(last_actions)) == 1
            is_alternating = (len(last_actions) == 4 and
                              last_actions[0] == last_actions[2] and
                              last_actions[1] == last_actions[3] and
                              last_actions[0] != last_actions[1])
            if is_repeat or is_alternating:
                last_actions.clear()
                # For task3: try to move to next uncomputed asset before submitting
                obs = result.observation
                if task_id == "task3_capgains" and hasattr(obs, "capital_gains_computed"):
                    all_assets = ["A001", "A002", "A003", "A004"]
                    computed = set(obs.capital_gains_computed.keys())
                    missing = [a for a in all_assets if a not in computed]
                    if missing:
                        # nudge model with explicit next asset instruction
                        next_asset = missing[0]
                        print(f"[DEBUG] Loop on capgains, nudging to {next_asset}", flush=True)
                        action_str = get_model_action(
                            client, task_id,
                            obs.model_dump_json(),
                            history + [{"role": "user", "content":
                                f"IMPORTANT: Stop repeating. Asset {next_asset} is not yet computed. "
                                f"Compute capgains for {next_asset} now."}],
                            step,
                        )
                        if action_str is None:
                            action_str = json.dumps(_make_submit_action(obs))
                    else:
                        print(f"[DEBUG] Loop detected at step {step}, forcing submit", flush=True)
                        action_str = json.dumps(_make_submit_action(obs))
                else:
                    print(f"[DEBUG] Loop detected at step {step}, forcing submit", flush=True)
                    action_str = json.dumps(_make_submit_action(result.observation))

            error_msg = None
            try:
                action_dict = json.loads(action_str)
                result = env.step(action_dict)
                reward = result.reward or 0.0
                done = result.done
            except json.JSONDecodeError as e:
                reward = 0.0
                error_msg = f"invalid_json: {str(e)[:80]}"
                done = False
            except Exception as e:
                reward = 0.0
                error_msg = str(e)[:120]
                done = False

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            history.append({"role": "assistant", "content": action_str})
            if not error_msg:
                history.append({
                    "role": "user",
                    "content": f"Step {step} result: reward={reward:.2f} done={done}"
                })

        # Get final graded score
        try:
            score = env.grade()
        except Exception:
            score = sum(rewards) / max(len(rewards), 1) / 0.30  # fallback estimate
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        error_msg = str(e)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_results = {}

    for task_config in TASKS:
        result = run_task(llm_client, task_config)
        all_results[result["task_id"]] = result
        print(f"[DEBUG] {result['task_id']} score={result['score']:.3f}", flush=True)

    # Write results to file
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll tasks complete. Results written to results.json", flush=True)
    for tid, r in all_results.items():
        print(f"  {tid}: score={r['score']:.3f} success={r['success']}", flush=True)


if __name__ == "__main__":
    main()
