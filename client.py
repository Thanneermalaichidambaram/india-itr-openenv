"""
IndiaITR OpenEnv client.
Thin HTTP client wrapping the FastAPI server endpoints.
Supports both sync and async usage patterns.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from env.models import ITRObservation, ITRState


@dataclass
class StepResult:
    observation: ITRObservation
    reward: Optional[float]
    done: bool


class IndiaITRClient:
    """
    Synchronous HTTP client for IndiaITR-OpenEnv server.

    Usage:
        client = IndiaITRClient(base_url="http://localhost:7860")
        result = client.reset(task_id="task1_parse", seed=42)
        result = client.step({"action_type": "read_section", "section": "form16_part_a"})
        client.close()
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(
        self,
        task_id: str = "task1_parse",
        seed: int = 42,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed, "episode_id": episode_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = ITRObservation.model_validate(data["observation"])
        return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

    def step(self, action: Dict[str, Any]) -> StepResult:
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = ITRObservation.model_validate(data["observation"])
        return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

    def state(self) -> ITRState:
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return ITRState.model_validate(resp.json())

    def grade(self) -> float:
        resp = self._session.get(f"{self.base_url}/grade", timeout=10)
        resp.raise_for_status()
        return float(resp.json()["score"])

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Factory — from docker image (mirrors OpenEnv pattern)
    # ------------------------------------------------------------------

    @classmethod
    def from_docker_image(cls, image_name: str, port: int = 7860) -> "IndiaITRClient":
        """
        Start a local Docker container and return a connected client.
        Requires Docker to be running.
        """
        import subprocess, time
        container_id = subprocess.check_output([
            "docker", "run", "-d", "--rm",
            "-p", f"{port}:7860",
            image_name
        ]).decode().strip()

        # Wait for health check
        client = cls(base_url=f"http://localhost:{port}")
        for _ in range(30):
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    client._container_id = container_id
                    return client
            except Exception:
                pass
            time.sleep(1)

        subprocess.run(["docker", "stop", container_id])
        raise RuntimeError("Container failed to start within 30 seconds")


class AsyncIndiaITRClient:
    """
    Async HTTP client — mirrors the OpenEnv async interface.
    Used by inference.py.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    async def reset(
        self,
        task_id: str = "task1_parse",
        seed: int = 42,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "seed": seed, "episode_id": episode_id},
            )
            resp.raise_for_status()
        data = resp.json()
        obs = ITRObservation.model_validate(data["observation"])
        return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

    async def step(self, action: Dict[str, Any]) -> StepResult:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/step",
                json={"action": action},
            )
            resp.raise_for_status()
        data = resp.json()
        obs = ITRObservation.model_validate(data["observation"])
        return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "AsyncIndiaITRClient":
        import subprocess, asyncio
        subprocess.Popen([
            "docker", "run", "--rm",
            "-p", f"{port}:7860",
            image_name
        ])
        client = cls(base_url=f"http://localhost:{port}")
        import httpx
        for _ in range(30):
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    r = await c.get(f"http://localhost:{port}/health")
                    if r.status_code == 200:
                        return client
            except Exception:
                pass
            await asyncio.sleep(1)
        raise RuntimeError("Container failed to start within 30 seconds")
