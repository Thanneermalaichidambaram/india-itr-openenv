---
title: IndiaITR OpenEnv
emoji: 💰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - india
  - tax
  - legal-ai
---

# IndiaITR-OpenEnv

**The first RL benchmark environment for Indian Income Tax Return filing.**

India has ~80 million income tax filers but only 400,000 Chartered Accountants. Every July, the filing deadline triggers a crisis: CAs are overwhelmed, clients miss deductions worth thousands of rupees, and errors lead to tax notices. IndiaITR-OpenEnv is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL environment that teaches AI agents to perform the core cognitive tasks of a CA.

---

## Environment Description

The environment simulates a CA's workflow for one ITR case file:
1. Reading Form 16 Part A and Part B
2. Identifying all applicable deductions (80C, 80D, HRA, 80TTA)
3. Selecting the optimal tax regime (old vs new)
4. Computing capital gains with CII indexation and Finance Act 2023 rules

All case files are generated synthetically from a seeded RNG — same seed always produces the same case. The tax engine is a deterministic oracle implementing actual Indian tax law (FY2024-25).

---

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `read_section` | `section` | Read a document section (unlocks visible data) |
| `extract_field` | `field_name`, `extracted_value` | Extract a field from a visible section |
| `flag_deduction` | `section`, `amount` | Claim a deduction under a tax section |
| `select_regime` | `regime`, `computed_tax_old`, `computed_tax_new` | Select old or new tax regime |
| `compute_capgains` | `asset_id`, `indexed_cost`, `gain`, `tax_rule_applied` | Compute capital gain for an asset |
| `submit` | `total_income`, `total_deductions`, `taxable_income`, `tax_payable`, `regime_selected` | Submit the completed ITR |

### Action Type Values
- `section`: `form16_part_a` | `form16_part_b` | `investments` | `assets`
- `regime`: `old` | `new`
- `tax_rule_applied`: `ltcg_equity_112a` | `stcg_equity_111a` | `ltcg_unlisted_20pct` | `ltcg_debt_indexed` | `stcg_slab` | `stcg_slab_finact2023` | `ltcg_property_indexed`

---

## Observation Space

```json
{
  "done": false,
  "reward": 0.10,
  "turn": 3,
  "task_id": "task1_parse",
  "max_turns": 20,
  "sections_read": ["form16_part_a"],
  "visible_data": {"form16_part_a": {...}},
  "extracted_fields": {"form16_part_a.gross_salary": 1200000.0},
  "flagged_deductions": {},
  "capital_gains_computed": {},
  "current_draft": {"total_income": 0, "deductions_claimed": {}, "regime_selected": null, "tax_payable": 0},
  "last_action_result": "Extracted form16_part_a.gross_salary = 1200000.0 [correct]",
  "validation_errors": [],
  "cumulative_reward": 0.15
}
```

---

## Tasks

### Task 1 — Form 16 Parsing (Easy) · Baseline: 0.82

**Objective:** Parse Form 16 Part A and Part B. Extract 12 key fields. Detect mismatches (TDS sum ≠ total TDS, Part A gross ≠ Part B gross).

**Grader:** `0.60 × field_accuracy + 0.40 × reconciliation_score`

**Edge cases:** 20% of cases have a TDS quarterly sum mismatch; 20% have a Part A/B gross salary mismatch.

### Task 2 — Deductions & Regime Selection (Medium) · Baseline: 0.67

**Objective:** Identify all applicable deductions. Select the optimal tax regime. Compute correct tax payable.

**Grader:** `0.40 × deduction_accuracy + 0.35 × regime_correct + 0.25 × tax_accuracy`

**Key challenge:** HRA exemption uses metro (50% of basic) vs non-metro (40%) rule. Metro cities: Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata.

### Task 3 — Capital Gains with CII Indexation (Hard) · Baseline: 0.51

**Objective:** Compute capital gains for 4 asset types (listed equity, debt MF, unlisted equity, gold ETF). Apply CII indexation and Finance Act 2023 rules.

**Grader:** `0.70 × mean(per_asset_score) + 0.30 × rule_application_score`

**The Finance Act 2023 trap:** Debt MF and Gold ETF purchased on/after 1-Apr-2023 → no indexation, taxed at slab rates regardless of holding period. This is what drops frontier models to 0.51.

---

## Reward Function

Dense rewards on every step (not just at episode end):

| Action | Reward |
|--------|--------|
| Read new section | +0.05 |
| Correct field extracted | +0.10 |
| Deduction correctly identified | +0.15 |
| Correct regime selected | +0.15 |
| Capital gains rule correct | +0.12 |
| Final submit (× final grade) | +0.30 |
| Wrong field value | -0.05 |
| Wrong deduction section | -0.10 |
| Wrong capital gains rule | -0.15 |
| Wrong regime selected | -0.20 |
| Loop penalty (re-read after turn 8) | -0.05 |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"task_id": "task1_parse", "seed": 42}` |
| `/step` | POST | Execute action. Body: `{"action": {...}}` |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/schema` | GET | Action/observation JSON schemas |
| `/metadata` | GET | Environment metadata |
| `/tasks` | GET | List all tasks |
| `/baseline` | GET | Baseline scores |
| `/grade` | GET | Final grade for current episode |

---

## Setup & Usage

### Local development

```bash
git clone <repo>
cd india-itr-openenv
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run inference (OpenAI, default)
export HF_TOKEN=sk-...          # your OpenAI API key
python inference.py

# Override model or endpoint
export MODEL_NAME=gpt-4o
export API_BASE_URL=https://api.openai.com/v1
python inference.py

# HuggingFace router alternative
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t india-itr-openenv .
docker run -p 7860:7860 india-itr-openenv
```

### Using the Python client

```python
from client import IndiaITRClient

with IndiaITRClient("http://localhost:7860") as env:
    result = env.reset(task_id="task1_parse", seed=42)
    result = env.step({"action_type": "read_section", "section": "form16_part_a"})
    result = env.step({"action_type": "extract_field",
                       "field_name": "form16_part_a.gross_salary",
                       "extracted_value": 1200000.0})
    score = env.grade()
    print(f"Score: {score:.3f}")
```

---

## Baseline Scores

Run with `MODEL_NAME=gpt-4.1-mini`, `seed=42`:

| Task | Score | Notes |
|------|-------|-------|
| Task 1 — Form 16 Parsing | **1.000** | Perfect score — step-guarded field extraction order |
| Task 2 — Deductions & Regime | **0.787** | Auto regime selection via tax slab computation |
| Task 3 — Capital Gains | **0.51+** | Finance Act 2023 date trap; arithmetic evaluator fixes JSON errors |

---

## Tax Law Coverage (FY2024-25)

- **Sections:** 80C (PPF, ELSS, LIC, home loan principal), 80D (health insurance), 80TTA (savings interest), HRA exemption, standard deduction ₹50,000
- **Regimes:** Old (with deductions) and New (FY2024-25 slabs post-July 2024 Budget)
- **CII:** Official IT dept values 2001–2025
- **Capital gains:** LTCG equity 112A (10% above ₹1L), STCG equity 111A (15%), unlisted equity LTCG (20%, 24-month threshold), debt MF/gold ETF (Finance Act 2023), property (indexed, 20%)
- **Rebate 87A:** Old regime ≤₹5L → rebate up to ₹12,500; New regime ≤₹7L → full rebate
