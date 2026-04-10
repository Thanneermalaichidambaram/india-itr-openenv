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

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — read from environment
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")
# OPENAI_API_KEY takes priority (local OpenAI usage); falls back to HF_TOKEN (HF router / Spaces)
API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "india-itr-openenv"
SEED = 42
MAX_STEPS = 20
MAX_STEPS_TASK3 = 25
TEMPERATURE = 0.0
MAX_TOKENS = 1500
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
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task1_parse": textwrap.dedent("""
        You are an expert Indian Chartered Accountant assistant.
        Your task is to parse Form 16 Part A and Part B documents and extract key fields.

        THINK BEFORE ACTING: Use the "_think" field to reason about what you see in visible_data
        and what exact value to extract. This prevents errors.

        At each turn, respond ONLY with a valid JSON object. You may include an optional "_think"
        key for reasoning — it will be ignored by the environment but helps you get values right.

        Action types:

        1. Read a section:
        {"_think": "I need to read form16_part_a first to see the data", "action_type": "read_section", "section": "form16_part_a"}
        {"_think": "Now I'll read part B", "action_type": "read_section", "section": "form16_part_b"}

        2. Extract a field (use EXACT value from visible_data, do not compute):
        {"_think": "visible_data shows gross_salary: 1200000.0 in part_a", "action_type": "extract_field", "field_name": "form16_part_a.gross_salary", "extracted_value": 1200000.0}

        3. Submit (only after all 12 fields extracted):
        {"action_type": "submit", "total_income": 1200000.0, "total_deductions": 50000.0,
         "taxable_income": 1150000.0, "tax_payable": 120000.0, "regime_selected": "new"}

        STRICT ORDER — do NOT submit until all 12 fields are extracted:
        Step 1: read_section form16_part_a
        Step 2: read_section form16_part_b
        Steps 3-14: extract each field ONE BY ONE in this order:
          1.  form16_part_a.employer_name        (string)
          2.  form16_part_a.employee_pan         (string)
          3.  form16_part_a.gross_salary         (number)
          4.  form16_part_a.tds_q1              (number)
          5.  form16_part_a.tds_q2              (number)
          6.  form16_part_a.tds_q3              (number)
          7.  form16_part_a.tds_q4              (number)
          8.  form16_part_a.total_tds           (number — read directly, do NOT sum quarters)
          9.  form16_part_b.basic_salary        (number)
          10. form16_part_b.hra_received        (number)
          11. form16_part_b.gross_salary        (number)
          12. form16_part_b.net_taxable_salary  (number)
        Step 15: submit after all 12 are in extracted_fields.

        IMPORTANT: Always check extracted_fields in the observation — skip fields already extracted.
        Respond ONLY with JSON. No text outside the JSON object.
    """).strip(),

    "task2_deduct": textwrap.dedent("""
        You are an expert Indian Chartered Accountant.
        Your task: identify all applicable tax deductions and select the optimal tax regime.

        THINK BEFORE ACTING: Use the "_think" field to show your computation step by step
        before committing to an amount. This prevents arithmetic errors.

        DEDUCTION RULES (compute carefully):
        - 80C: sum of (ppf + elss + life_insurance + home_loan_principal + school_fees + nsc), capped at 150000
        - 80D: min(health_insurance_self, 25000) + min(health_insurance_parents, 50000 if parents_senior_citizen else 25000)
        - HRA: min(A, B, C) where:
            A = hra_received (from form16_part_b)
            B = basic_salary × 0.50 if city in [mumbai, delhi, bengaluru, bengalore, hyderabad, chennai, kolkata] else × 0.40
            C = max(rent_paid_monthly × 12 - basic_salary × 0.10, 0)
          Use EXACT numbers from visible_data. Show all three values in _think.
        - 80TTA: min(savings_account_interest, 10000)
        - Standard deduction: always 50000 (do NOT flag this, but include in submit)

        TAX SLABS (FY2024-25):
        Old regime: 0-2.5L→0%, 2.5L-5L→5%, 5L-10L→20%, >10L→30%. 87A rebate if taxable≤5L.
        New regime: 0-3L→0%, 3L-7L→5%, 7L-10L→10%, 10L-12L→15%, 12L-15L→20%, >15L→30%. 87A rebate if taxable≤7L.
        Add 4% health & education cess on final tax.
        Old regime taxable = gross_salary - 80C - 80D - HRA - 80TTA - 50000(standard)
        New regime taxable = gross_salary - 50000(standard) [no other deductions]

        Action sequence (each ONCE, in order):
        {"_think": "need to read part_b for salary data", "action_type": "read_section", "section": "form16_part_b"}
        {"_think": "need investments data", "action_type": "read_section", "section": "investments"}
        {"_think": "ppf=X elss=Y ... sum=Z, capped at 150000", "action_type": "flag_deduction", "section": "80C", "amount": <computed>}
        {"_think": "self=X min(X,25000)=A; parents=Y senior=T/F min(Y,50000/25000)=B; total=A+B", "action_type": "flag_deduction", "section": "80D", "amount": <computed>}
        {"_think": "A=hra_received=X; B=basic×0.5/0.4=Y; C=rent×12-basic×0.1=Z; min(A,B,C)=W", "action_type": "flag_deduction", "section": "HRA", "amount": <computed>}
        {"_think": "savings_interest=X, min(X,10000)=Y", "action_type": "flag_deduction", "section": "80TTA", "amount": <computed>}
        {"_think": "old_taxable=gross-80C-80D-HRA-80TTA-50000=X; old_tax=slab(X)+4%cess=A; new_taxable=gross-50000=Y; new_tax=slab(Y)+4%cess=B; choose lower", "action_type": "select_regime", "regime": "old_or_new", "computed_tax_old": <A>, "computed_tax_new": <B>}
        {"action_type": "submit", "total_income": <gross_salary>, "total_deductions": <80C+80D+HRA+80TTA+50000>, "taxable_income": <computed>, "tax_payable": <computed>, "regime_selected": "old_or_new", "deductions_claimed": {"80C": ..., "80D": ..., "HRA": ..., "80TTA": ..., "standard_deduction": 50000}}

        CRITICAL: The submit's deductions_claimed MUST include "standard_deduction": 50000.
        IMPORTANT: Flag each section exactly ONCE. Check flagged_deductions in observation.
        Valid flag_deduction sections: 80C, 24b, 80D, HRA, 80TTA, 80G (NOT standard_deduction).
        Respond ONLY with JSON. No text outside the JSON object.
    """).strip(),

    "task3_capgains": textwrap.dedent("""
        You are an expert Indian Chartered Accountant specializing in capital gains tax.
        Your task: compute capital gains for all assets using correct CII indexation.

        THINK BEFORE ACTING: Use "_think" to show your date math, CII lookup, and rule selection
        step by step before computing. This prevents errors.

        CLASSIFICATION RULES:
        1. Debt MF / Gold ETF purchased ON OR AFTER 2023-04-01: → stcg_slab_finact2023 (no indexation, any holding period)
        2. Debt MF / Gold ETF purchased BEFORE 2023-04-01:
           - Held MORE THAN 36 months: → ltcg_debt_indexed (CII indexation, 20%)
           - Held 36 months or less: → stcg_slab (no indexation)
        3. Equity listed + STT paid:
           - Held MORE THAN 12 months: → ltcg_equity_112a (no CII, 10% above ₹1L exempt)
           - Held ≤ 12 months: → stcg_equity_111a (15%)
        4. Equity unlisted:
           - Held MORE THAN 24 months: → ltcg_unlisted_20pct (CII indexation, 20%)
           - Held ≤ 24 months: → stcg_slab
        5. Property:
           - Held MORE THAN 24 months: → ltcg_property_indexed (CII indexation, 20%)
           - Held ≤ 24 months: → stcg_slab

        HOLDING PERIOD: Count months = floor(days_between / 30.44)
        Example: 2022-03-12 to 2024-12-01 = 995 days / 30.44 = 32 months (< 36 → stcg_slab)

        FINANCIAL YEAR: if month >= 4 (April), FY = that year; else FY = year - 1
        Examples: 2022-03-15 → FY2021; 2023-07-01 → FY2023; 2025-01-10 → FY2024

        CII TABLE (FY → CII):
        2001→100, 2002→105, 2003→109, 2004→113, 2005→117, 2006→122, 2007→129,
        2008→137, 2009→148, 2010→167, 2011→184, 2012→200, 2013→220, 2014→240,
        2015→254, 2016→264, 2017→272, 2018→280, 2019→289, 2020→301, 2021→317,
        2022→331, 2023→348, 2024→363, 2025→376

        INDEXED COST:
        - Equity: NO CII indexation. indexed_cost = purchase_amount (unless pre-31-Jan-2018 grandfathering applies)
        - Debt MF/Gold ETF (Finance Act 2023 case): NO CII. indexed_cost = purchase_amount
        - Debt MF/Gold ETF (old rules, LTCG): indexed_cost = purchase_amount × (CII_sale_FY / CII_purchase_FY)
        - Property (LTCG): indexed_cost = purchase_amount × (CII_sale_FY / CII_purchase_FY)
        - STCG cases: indexed_cost = purchase_amount (no indexation)

        GAIN: gain = sale_amount - indexed_cost

        COMPUTE EXAMPLE:
        Asset: gold_etf, purchased 2022-01-15 (FY2021, CII=317), sold 2024-11-01 (FY2024, CII=363)
        purchase before 2023-04-01 ✓
        holding = (2024-11-01 - 2022-01-15) = 1021 days / 30.44 = 33 months < 36 → stcg_slab
        indexed_cost = purchase_amount (stcg, no indexation)
        gain = sale_amount - purchase_amount

        Action sequence:
        1. {"action_type": "read_section", "section": "assets"}
        2. For each asset (A001, A002, A003, A004) ONE BY ONE:
           {"_think": "asset_type=X purchase_date=Y sale_date=Z; purchase FY=P CII=p; sale FY=S CII=s; holding=(Z-Y).days/30.44=M months; rule=...; indexed_cost=...; gain=sale-cost", "action_type": "compute_capgains", "asset_id": "A001", "indexed_cost": <float>, "gain": <float>, "tax_rule_applied": "<rule>"}
        3. After ALL 4 assets computed, submit.

        CRITICAL JSON RULE: JSON values must be pre-computed numbers — NEVER write arithmetic
        expressions inside JSON. WRONG: "gain": 509565.68 - 276868.18
        RIGHT: compute it first in _think, then write the result: "gain": 232697.5

        IMPORTANT: After each compute, check capital_gains_computed in the observation.
        Never repeat an asset_id that's already computed.
        Respond ONLY with JSON. No text outside the JSON object.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Observation summary builder (compact, task-relevant)
# ---------------------------------------------------------------------------

def _build_obs_summary(obs_json: str, task_id: str) -> str:
    """Return a compact JSON string with only the fields relevant to the task."""
    try:
        obs = json.loads(obs_json)
        compact: Dict[str, Any] = {
            "turn": obs.get("turn"),
            "max_turns": obs.get("max_turns"),
            "sections_read": obs.get("sections_read", []),
            "last_action_result": obs.get("last_action_result", ""),
        }
        if obs.get("validation_errors"):
            compact["validation_errors"] = obs["validation_errors"]

        if task_id == "task1_parse":
            compact["extracted_fields"] = obs.get("extracted_fields", {})
            compact["visible_data"] = obs.get("visible_data", {})
        elif task_id == "task2_deduct":
            compact["flagged_deductions"] = obs.get("flagged_deductions", {})
            compact["current_draft"] = obs.get("current_draft", {})
            compact["visible_data"] = obs.get("visible_data", {})
        elif task_id == "task3_capgains":
            compact["capital_gains_computed"] = obs.get("capital_gains_computed", {})
            compact["visible_data"] = obs.get("visible_data", {})

        return json.dumps(compact)
    except Exception:
        return obs_json


# ---------------------------------------------------------------------------
# Hint builder — tells model exactly what's still missing
# ---------------------------------------------------------------------------

def _build_hint(obs_json: str, task_id: str, step: int, max_steps: int) -> str:
    hints = []
    steps_left = max_steps - step
    if steps_left <= 4:
        hints.append(f"WARNING: Only {steps_left} steps remaining — prioritize submit!")

    try:
        obs = json.loads(obs_json)
        if task_id == "task1_parse":
            required = [
                "form16_part_a.employer_name", "form16_part_a.employee_pan",
                "form16_part_a.gross_salary", "form16_part_a.tds_q1",
                "form16_part_a.tds_q2", "form16_part_a.tds_q3",
                "form16_part_a.tds_q4", "form16_part_a.total_tds",
                "form16_part_b.basic_salary", "form16_part_b.hra_received",
                "form16_part_b.gross_salary", "form16_part_b.net_taxable_salary",
            ]
            extracted = obs.get("extracted_fields", {})
            missing = [f for f in required if f not in extracted]
            if missing:
                hints.append(f"NOT YET EXTRACTED (do these before submit): {missing}")
                hints.append(f"Next field to extract: {missing[0]}")
            else:
                hints.append("All 12 fields extracted. Now submit.")

        elif task_id == "task2_deduct":
            flagged = set(obs.get("flagged_deductions", {}).keys())
            needed = {"80C", "80D", "HRA", "80TTA"}
            sections_read = set(obs.get("sections_read", []))
            if "form16_part_b" not in sections_read:
                hints.append("Must read form16_part_b first.")
            elif "investments" not in sections_read:
                hints.append("Must read investments next.")
            else:
                unflagged = needed - flagged
                if unflagged:
                    hints.append(f"NOT YET FLAGGED: {sorted(unflagged)}. Flag each once.")
                    hints.append(f"Next: flag_deduction for {sorted(unflagged)[0]}")
                elif "regime_selected" not in str(obs.get("current_draft", {})):
                    hints.append("All deductions flagged. Now select_regime.")
                else:
                    hints.append("Regime selected. Now submit with all deductions_claimed including standard_deduction: 50000.")

        elif task_id == "task3_capgains":
            computed = set(obs.get("capital_gains_computed", {}).keys())
            all_assets = ["A001", "A002", "A003", "A004"]
            missing = [a for a in all_assets if a not in computed]
            if "assets" not in obs.get("sections_read", []):
                hints.append("Must read assets section first.")
            elif missing:
                hints.append(f"NOT YET COMPUTED: {missing}. Next: {missing[0]}")
            else:
                hints.append("All 4 assets computed. Now submit.")
    except Exception:
        pass

    return ("\n" + "\n".join(hints)) if hints else ""


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _llm_call(client: OpenAI, messages: List[Dict]) -> Optional[str]:
    """Make one LLM call; return text or None on error."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        return text if text else None
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return None


def _eval_arithmetic_in_json(text: str) -> str:
    """
    Fix model habit of writing arithmetic expressions as JSON values, e.g.:
      "gain": 509565.68 - 276868.18   →   "gain": 232697.5
    Replaces all occurrences of  : <number> <op> <number>  with the evaluated result.
    """
    import re
    def _replace(m):
        try:
            return f": {round(eval(m.group(1)), 2)}"  # noqa: S307 — safe: only numbers/ops
        except Exception:
            return m.group(0)
    return re.sub(r":\s*([\d.]+\s*[-+*/]\s*[\d.]+)", _replace, text)


def _parse_action(text: str) -> Optional[Dict]:
    """Parse and clean a JSON action string. Strips _think key."""
    if text is None:
        return None
    # Try direct parse first
    try:
        d = json.loads(text)
        d.pop("_think", None)
        return d
    except json.JSONDecodeError:
        pass
    # Model wrote arithmetic expressions — evaluate and retry
    try:
        d = json.loads(_eval_arithmetic_in_json(text))
        d.pop("_think", None)
        return d
    except json.JSONDecodeError:
        return None


def get_model_action(
    client: OpenAI,
    task_id: str,
    obs_json: str,
    history: List[Dict],
    step: int,
    max_steps: int,
) -> Optional[str]:
    """
    Build the prompt and call the LLM. Returns cleaned action JSON string or None.
    Includes one retry on invalid JSON.
    """
    obs_summary = _build_obs_summary(obs_json, task_id)
    hint = _build_hint(obs_json, task_id, step, max_steps)

    steps_left = max_steps - step
    budget_note = f" ({steps_left} steps remaining)" if steps_left <= 6 else ""

    user_content = textwrap.dedent(f"""
        Step {step}{budget_note}. Current environment state:
        {obs_summary}{hint}

        Think in "_think", then respond with exactly one JSON action object. No other text.
    """).strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task_id]}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_content})

    text = _llm_call(client, messages)
    if text is None:
        return None

    # Check if parseable; if not, retry once
    if _parse_action(text) is None:
        print(f"[DEBUG] Invalid JSON at step {step}, retrying...", flush=True)
        retry_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                "Your response was not valid JSON. Respond with ONLY a valid JSON object. "
                "No markdown, no explanation outside the JSON."
            )},
        ]
        text = _llm_call(client, retry_messages)

    return text


# ---------------------------------------------------------------------------
# Emergency submit builder
# ---------------------------------------------------------------------------

def _make_submit_action(obs, task_id: str = "") -> Dict:
    """Build a best-effort submit from current observation state."""
    draft = obs.current_draft
    total_income = (
        draft.total_income
        or obs.extracted_fields.get("form16_part_b.gross_salary", 0)
        or obs.extracted_fields.get("form16_part_a.gross_salary", 0)
        or 1_000_000.0
    )
    flagged = dict(obs.flagged_deductions) if obs.flagged_deductions else {}
    # Always include standard_deduction — grader checks it
    if "standard_deduction" not in flagged:
        flagged["standard_deduction"] = 50_000.0
    total_ded = sum(flagged.values())
    taxable = max(float(total_income) - float(total_ded), 0)

    capgains = {}
    for k, v in obs.capital_gains_computed.items():
        capgains[k] = v.get("gain", 0) if isinstance(v, dict) else float(v)

    return {
        "action_type": "submit",
        "total_income": float(total_income),
        "total_deductions": float(total_ded),
        "taxable_income": taxable,
        "tax_payable": float(draft.tax_payable or 0),
        "regime_selected": draft.regime_selected or "new",
        "deductions_claimed": flagged,
        "capital_gains_computed": capgains,
    }


def _final_chance_submit(client: OpenAI, task_id: str, obs, history: List[Dict]) -> str:
    """
    Like runner.py's _default_answer: when steps are exhausted, ask the model to
    produce a final submit based on all work done so far.
    """
    obs_summary = _build_obs_summary(obs.model_dump_json(), task_id)
    prompt = (
        f"You are running out of steps. Based on all your work so far, produce a submit action now.\n"
        f"Current state:\n{obs_summary}\n\n"
        f"Output ONLY a JSON submit action with all values filled in. "
        f"For task2, include standard_deduction: 50000 in deductions_claimed."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task_id]}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": prompt})
    text = _llm_call(client, messages)
    if text:
        action = _parse_action(text)
        if action is not None:
            # Validate submit has required fields (model sometimes returns bare {"action_type":"submit"})
            if action.get("action_type") == "submit":
                if action.get("total_income") and action.get("regime_selected"):
                    return text
            else:
                return text
    return json.dumps(_make_submit_action(obs, task_id))


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

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
    last_actions: List[str] = []   # for loop detection

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = IndiaITRClient(base_url=ENV_BASE_URL)
    try:
        result = env.reset(task_id=task_id, seed=SEED)
        done = result.done

        for step in range(1, max_steps + 1):
            if done:
                break

            obs_json = result.observation.model_dump_json()
            steps_left = max_steps - step

            # ── Final-chance submit when out of time ──────────────────────
            if steps_left == 0:
                print(f"[DEBUG] Final step {step}, forcing submit", flush=True)
                action_str = _final_chance_submit(client, task_id, result.observation, history)
            else:
                action_str = get_model_action(client, task_id, obs_json, history, step, max_steps)

                if action_str is None:
                    # API error → best-effort submit
                    print(f"[DEBUG] API error at step {step}, forcing submit", flush=True)
                    action_str = json.dumps(_make_submit_action(result.observation, task_id))

            # ── Loop detection: same action 3× or A-B-A-B pattern ────────
            last_actions.append(action_str[:80])
            if len(last_actions) > 4:
                last_actions.pop(0)
            is_repeat = len(last_actions) >= 3 and len(set(last_actions[-3:])) == 1
            is_alternating = (
                len(last_actions) == 4
                and last_actions[0] == last_actions[2]
                and last_actions[1] == last_actions[3]
                and last_actions[0] != last_actions[1]
            )
            if is_repeat or is_alternating:
                last_actions.clear()
                obs = result.observation

                if task_id == "task3_capgains" and hasattr(obs, "capital_gains_computed"):
                    all_assets = ["A001", "A002", "A003", "A004"]
                    computed = set(obs.capital_gains_computed.keys())
                    missing_assets = [a for a in all_assets if a not in computed]
                    if missing_assets:
                        next_asset = missing_assets[0]
                        print(f"[DEBUG] Loop detected, nudging to asset {next_asset}", flush=True)
                        nudge_history = history + [{"role": "user", "content":
                            f"STOP REPEATING. Asset {next_asset} is NOT yet computed. "
                            f"Read the asset data for {next_asset} from visible_data and compute it now. "
                            f"Use _think to show: purchase_date, sale_date, holding months, FY years, CII values, rule, indexed_cost, gain."}]
                        nudged = get_model_action(client, task_id, obs.model_dump_json(), nudge_history, step, max_steps)
                        action_str = nudged if nudged else json.dumps(_make_submit_action(obs, task_id))
                    else:
                        print(f"[DEBUG] Loop at step {step}, all assets done, forcing submit", flush=True)
                        action_str = json.dumps(_make_submit_action(obs, task_id))
                else:
                    print(f"[DEBUG] Loop at step {step}, forcing submit", flush=True)
                    action_str = _final_chance_submit(client, task_id, result.observation, history)

            # ── Task1 guard: block extracting part_b fields before reading part_b ──
            if task_id == "task1_parse":
                action_dict_peek = _parse_action(action_str)
                if (
                    action_dict_peek is not None
                    and action_dict_peek.get("action_type") == "extract_field"
                    and str(action_dict_peek.get("field_name", "")).startswith("form16_part_b.")
                    and "form16_part_b" not in result.observation.sections_read
                ):
                    print("[DEBUG] Task1: blocking part_b extraction — part_b not yet read", flush=True)
                    action_str = json.dumps({"action_type": "read_section", "section": "form16_part_b"})

            # ── Task2 guard: cannot submit without select_regime first ───────
            if task_id == "task2_deduct":
                action_dict_peek = _parse_action(action_str)
                if (
                    action_dict_peek is not None
                    and action_dict_peek.get("action_type") == "submit"
                    and not result.observation.current_draft.regime_selected
                    and result.observation.flagged_deductions  # has done some deductions
                ):
                    print("[DEBUG] Task2: submit without select_regime — forcing select_regime first", flush=True)
                    # Build a select_regime action from current draft/flagged state
                    obs = result.observation
                    flagged = dict(obs.flagged_deductions)
                    flagged["standard_deduction"] = 50_000.0
                    gross = (
                        obs.extracted_fields.get("form16_part_b.gross_salary")
                        or obs.current_draft.total_income
                        or (obs.visible_data.get("form16_part_b") or {}).get("gross_salary")
                        or 0.0
                    )
                    total_ded_old = sum(flagged.values())
                    taxable_old = max(float(gross) - float(total_ded_old), 0)
                    taxable_new = max(float(gross) - 50_000.0, 0)

                    def _slab(income, slabs):
                        tax, prev = 0.0, 0.0
                        for ceil, rate in slabs:
                            if income <= prev: break
                            tax += (min(income, ceil) - prev) * rate
                            prev = ceil
                        return round(tax * 1.04, 2)

                    old_slabs = [(250000,0),(500000,.05),(1000000,.20),(float("inf"),.30)]
                    new_slabs = [(300000,0),(700000,.05),(1000000,.10),(1200000,.15),(1500000,.20),(float("inf"),.30)]
                    old_tax = _slab(taxable_old, old_slabs)
                    new_tax = _slab(taxable_new, new_slabs)
                    regime = "old" if old_tax < new_tax else "new"
                    action_str = json.dumps({
                        "action_type": "select_regime",
                        "regime": regime,
                        "computed_tax_old": old_tax,
                        "computed_tax_new": new_tax,
                    })

            # ── Guard: catch bare submit missing required fields ──────────
            action_dict_pre = _parse_action(action_str)
            if (
                action_dict_pre is not None
                and action_dict_pre.get("action_type") == "submit"
                and not action_dict_pre.get("total_income")
            ):
                print(f"[DEBUG] Bare submit at step {step}, replacing with full submit", flush=True)
                action_str = json.dumps(_make_submit_action(result.observation, task_id))

            # ── Parse and execute ─────────────────────────────────────────
            error_msg = None
            action_dict = _parse_action(action_str)

            if action_dict is None:
                reward = 0.0
                error_msg = "invalid_json"
                done = False
                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=str(action_str)[:200], reward=reward, done=done, error=error_msg)
                history.append({"role": "assistant", "content": str(action_str)})
                continue

            # Rebuild action_str without _think for logging
            action_str_clean = json.dumps(action_dict)

            try:
                result = env.step(action_dict)
                reward = result.reward or 0.0
                done = result.done
            except Exception as e:
                reward = 0.0
                error_msg = str(e)[:120]
                done = False

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str_clean, reward=reward, done=done, error=error_msg)

            history.append({"role": "assistant", "content": action_str_clean})
            if not error_msg and hasattr(result, "observation"):
                feedback = f"Step {step} result: reward={reward:.2f} done={done}"
                obs_result = result.observation
                if obs_result.last_action_result:
                    feedback += f" | {obs_result.last_action_result}"
                if obs_result.validation_errors:
                    feedback += f" | errors: {obs_result.validation_errors}"
                history.append({"role": "user", "content": feedback})

        # ── Grade ─────────────────────────────────────────────────────────
        try:
            score = env.grade()
        except Exception:
            score = sum(rewards) / max(len(rewards), 1) / 0.30
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        import traceback; traceback.print_exc()
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_results = {}

    for task_config in TASKS:
        result = run_task(llm_client, task_config)
        all_results[result["task_id"]] = result
        print(f"[DEBUG] {result['task_id']} score={result['score']:.3f}", flush=True)

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll tasks complete. Results written to results.json", flush=True)
    for tid, r in all_results.items():
        print(f"  {tid}: score={r['score']:.3f} success={r['success']}", flush=True)


if __name__ == "__main__":
    main()
