"""
FastAPI application for IndiaITR-OpenEnv.

Uses a single shared environment instance for HTTP sessions (/reset, /step, /state).
Also registers openenv-core required endpoints (/health, /schema, /metadata, /openapi.json).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.models import ITRObservation, ITRState
from server.india_itr_environment import IndiaITROpenEnvEnvironment, TASK_IDS

try:
    from ..models import ITRAction, ITRObservation as RootITRObservation
except (ImportError, ModuleNotFoundError):
    from models import ITRAction, ITRObservation as RootITRObservation

app = FastAPI(
    title="IndiaITR-OpenEnv",
    description="RL environment for Indian Income Tax Return filing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance for HTTP sessions
_env = IndiaITROpenEnvEnvironment()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    episode_id: Optional[str] = None
    task_id: Optional[str] = "task1_parse"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    timeout_s: Optional[float] = None


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


# ---------------------------------------------------------------------------
# OpenEnv required endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "env": "india-itr-openenv"}


@app.get("/schema")
def schema():
    return {
        "action": ITRAction.model_json_schema(),
        "observation": RootITRObservation.model_json_schema(),
        "state": {"type": "object", "description": "Environment state"},
    }


@app.get("/metadata")
def metadata():
    return {
        "name": "india-itr-openenv",
        "description": "RL environment for Indian ITR filing",
        "version": "1.0.0",
        "tasks": TASK_IDS,
        "reward_range": [-1.0, 1.0],
        "max_turns": {"task1_parse": 20, "task2_deduct": 20, "task3_capgains": 25},
        "baseline_scores": {"task1_parse": 0.82, "task2_deduct": 0.67, "task3_capgains": 0.51},
    }


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    try:
        obs = _env.reset(
            seed=req.seed,
            episode_id=req.episode_id,
            task_id=req.task_id,
        )
        return ResetResponse(
            observation=obs.model_dump(mode="json"),
            reward=obs.reward,
            done=obs.done,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        action = ITRAction.model_validate(req.action)
        obs = _env.step(action, timeout_s=req.timeout_s)
        return StepResponse(
            observation=obs.model_dump(mode="json"),
            reward=obs.reward,
            done=obs.done,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        s = _env.state
        return s.model_dump(mode="json")
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task1_parse", "name": "Form 16 Document Parsing", "difficulty": "easy", "baseline_score": 0.82},
            {"id": "task2_deduct", "name": "Deduction Identification & Regime Selection", "difficulty": "medium", "baseline_score": 0.67},
            {"id": "task3_capgains", "name": "Capital Gains with CII Indexation", "difficulty": "hard", "baseline_score": 0.51},
        ]
    }


@app.get("/baseline")
def baseline():
    return {"task1_parse": 0.82, "task2_deduct": 0.67, "task3_capgains": 0.51}


@app.get("/grade")
def grade():
    try:
        score = _env.final_grade()
        return {"score": score}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
