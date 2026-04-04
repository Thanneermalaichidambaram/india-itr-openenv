"""
HF Spaces entry point.

FastAPI serves API routes (/reset, /step, /health, etc.).
Gradio UI is mounted at /ui path but also served at root via custom route.
"""
import gradio as gr
import json
import uvicorn

from server.app import app as fastapi_app

# ---------------------------------------------------------------------------
# Task info
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    "task1_parse":    {"name": "Task 1 — Form 16 Parsing (Easy)",         "score": 0.82},
    "task2_deduct":   {"name": "Task 2 — Deductions & Regime (Medium)",    "score": 0.67},
    "task3_capgains": {"name": "Task 3 — Capital Gains (Hard)",            "score": 0.51},
}

ACTION_EXAMPLES = {
    "read_section":     '{"action_type": "read_section", "section": "form16_part_a"}',
    "extract_field":    '{"action_type": "extract_field", "field_name": "form16_part_a.gross_salary", "extracted_value": 1200000.0}',
    "flag_deduction":   '{"action_type": "flag_deduction", "section": "80C", "amount": 150000.0}',
    "select_regime":    '{"action_type": "select_regime", "regime": "new", "computed_tax_old": 95000.0, "computed_tax_new": 85000.0}',
    "compute_capgains": '{"action_type": "compute_capgains", "asset_id": "A001", "indexed_cost": 500000.0, "gain": 300000.0, "tax_rule_applied": "ltcg_equity_112a"}',
    "submit":           '{"action_type": "submit", "total_income": 1200000.0, "total_deductions": 200000.0, "taxable_income": 1000000.0, "tax_payable": 85000.0, "regime_selected": "new"}',
}

_session = {"done": False}


def do_reset(task_id, seed):
    _session["done"] = False
    import requests
    try:
        r = requests.post("http://127.0.0.1:7860/reset",
                          json={"task_id": task_id, "seed": int(seed)}, timeout=15)
        r.raise_for_status()
        obs = r.json()["observation"]
        return json.dumps(obs, indent=2), f"Episode started. Task: {task_id}, Seed: {seed}"
    except Exception as e:
        return "{}", f"Error: {e}"


def do_step(action_json):
    if _session["done"]:
        return "{}", "Episode done. Please reset."
    import requests
    try:
        action = json.loads(action_json)
        r = requests.post("http://127.0.0.1:7860/step",
                          json={"action": action}, timeout=15)
        r.raise_for_status()
        data = r.json()
        obs = data["observation"]
        reward = data.get("reward", 0)
        done = data.get("done", False)
        _session["done"] = done
        status = f"Reward: {reward:.3f} | Done: {done}\n{obs.get('last_action_result', '')}"
        if done:
            try:
                g = requests.get("http://127.0.0.1:7860/grade", timeout=5).json()
                status += f"\n\nFinal Score: {g['score']:.3f}"
            except Exception:
                pass
        return json.dumps(obs, indent=2), status
    except json.JSONDecodeError:
        return "{}", "Invalid JSON."
    except Exception as e:
        return "{}", f"Error: {e}"


def fill_template(t):
    return ACTION_EXAMPLES.get(t, "{}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="IndiaITR-OpenEnv") as demo:
    gr.Markdown("""
    # 💰 IndiaITR-OpenEnv
    ### OpenEnv RL Environment for Indian Income Tax Return Filing

    An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL environment
    that teaches AI agents to perform core tasks of a Chartered Accountant.

    **REST API:** `/reset` · `/step` · `/state` · `/health` · `/schema` · `/metadata`
    """)

    with gr.Row():
        for tid, info in TASK_DESCRIPTIONS.items():
            with gr.Column():
                gr.Markdown(f"**{info['name']}**\n\nBaseline score: **{info['score']}**")

    gr.Markdown("---\n## Interactive Demo")

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(choices=list(TASK_DESCRIPTIONS.keys()), value="task1_parse", label="Task")
            seed_in = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("Reset Episode", variant="primary")
            status_box = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Column(scale=2):
            obs_box = gr.Code(label="Observation (JSON)", language="json", lines=22)

    with gr.Row():
        with gr.Column():
            action_dd = gr.Dropdown(choices=list(ACTION_EXAMPLES.keys()), value="read_section", label="Action template")
            action_in = gr.Code(label="Action JSON", language="json", value=ACTION_EXAMPLES["read_section"], lines=5)
            step_btn = gr.Button("Execute Step", variant="secondary")

    action_dd.change(fill_template, inputs=action_dd, outputs=action_in)
    reset_btn.click(do_reset, inputs=[task_dd, seed_in], outputs=[obs_box, status_box])
    step_btn.click(do_step, inputs=action_in, outputs=[obs_box, status_box])

    gr.Markdown("""
    ---
    ## Baseline Scores (Qwen2.5-Coder 7B, seed=42)
    | Task | Score |
    |------|-------|
    | Task 1 — Form 16 Parsing | 1.000 |
    | Task 2 — Deductions & Regime | 0.850 |
    | Task 3 — Capital Gains | 0.709 |
    """)


# ---------------------------------------------------------------------------
# Mount Gradio INTO FastAPI at root "/"
# API routes (/reset /step /health etc.) were registered on fastapi_app first
# so they take priority over Gradio's catch-all.
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
