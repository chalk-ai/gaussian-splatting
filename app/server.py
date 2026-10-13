"""Live Gaussian Splatting training on one GPU, streamed to the browser.

HTTP API:
  GET  /                          the page (index.html)
  GET  /api/status                current run, latest finished run, limits
  POST /api/train                 {"iterations": 2000|4000|7000} -> {"run_id"}; 409 while busy
  GET  /api/runs/{id}/events      Server-Sent Events: every progress event so far, then live ones
  GET  /api/runs/{id}/scene.splat the trained Gaussians (32-byte .splat records)
  GET  /api/target.jpg            the held-out photo the live preview is rendered from

One run trains at a time. The replica trains once at boot so a visitor always has a finished
scene to explore, and a new run can start after COOLDOWN_S.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

import splat_trainer as st

SCENE_DIR = Path(os.environ.get("SCENE_DIR", "/data/treasure_chest"))
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")
ALLOWED_ITERATIONS = (2000, 4000, 7000)
BOOT_ITERATIONS = 7000
COOLDOWN_S = 20.0
MAX_RUNS_KEPT = 4

log = logging.getLogger("splat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Run:
    id: str
    iterations: int
    started_at: float
    events: list[dict[str, Any]]
    splat: bytes | None
    done: bool
    error: str | None
    # Wakes SSE readers; replaced (and the old one set) every time an event lands.
    changed: threading.Event

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        old, self.changed = self.changed, threading.Event()
        old.set()

    def summary(self) -> dict[str, Any]:
        last = self.events[-1] if self.events else {}
        return {
            "run_id": self.id,
            "iterations": self.iterations,
            "step": last.get("step", 0),
            "gaussians": last.get("gaussians"),
            "psnr": next((e["psnr"] for e in reversed(self.events) if e.get("psnr") is not None), None),
            "elapsed_s": last.get("elapsed_s"),
            "done": self.done,
            "error": self.error,
        }


class Trainer:
    """Owns the GPU: at most one training run at a time, and the recent runs' results."""

    def __init__(self, scene: st.Scene) -> None:
        super().__init__()
        self.scene = scene
        self.runs: dict[str, Run] = {}
        self.current: Run | None = None
        self.latest_done: Run | None = None
        self.last_finished_at = 0.0
        self.lock = threading.Lock()

    def start(self, iterations: int, enforce_cooldown: bool) -> Run:
        with self.lock:
            if self.current is not None:
                raise HTTPException(409, "a training run is already in progress")
            wait = self.last_finished_at + COOLDOWN_S - time.time()
            if enforce_cooldown and wait > 0:
                raise HTTPException(429, f"the GPU is cooling down; try again in {wait:.0f}s")
            run = Run(
                id=uuid.uuid4().hex[:10],
                iterations=iterations,
                started_at=time.time(),
                events=[],
                splat=None,
                done=False,
                error=None,
                changed=threading.Event(),
            )
            self.runs[run.id] = run
            self.current = run
            for stale in list(self.runs)[:-MAX_RUNS_KEPT]:
                if self.runs[stale] is not self.latest_done:
                    del self.runs[stale]
        threading.Thread(target=self._train, args=(run,), daemon=True).start()
        return run

    def _train(self, run: Run) -> None:
        config = st.TrainConfig(iterations=run.iterations, initial_gaussians=20_000, preview_every=100, eval_every=250)
        try:
            for item in st.train(self.scene, config, seed=int(run.started_at)):
                if isinstance(item, st.Progress):
                    event = asdict(item)
                    jpeg = event.pop("preview_jpeg")
                    event["type"] = "progress"
                    event["preview"] = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                    run.publish(event)
                else:
                    run.splat = st.to_splat(item)
            run.done = True
            run.publish({"type": "done", **run.summary()})
            log.info("run %s finished: %s", run.id, run.summary())
        except Exception as exc:  # noqa: BLE001 - surfaced to the page, not swallowed
            log.exception("run %s failed", run.id)
            run.error = str(exc)
            run.publish({"type": "error", "message": str(exc)})
        finally:
            torch.cuda.empty_cache()
            with self.lock:
                self.current = None
                self.last_finished_at = time.time()
                if run.done:
                    self.latest_done = run


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    scene = await asyncio.to_thread(st.load_scene, SCENE_DIR, torch.device("cuda"))
    log.info("loaded %d training and %d held-out views", len(scene.images), len(scene.test_images))
    trainer = Trainer(scene)
    state["trainer"] = trainer
    state["target_jpeg"] = st.to_jpeg(scene.test_images[0], 85)
    trainer.start(BOOT_ITERATIONS, enforce_cooldown=False)
    yield


app = FastAPI(lifespan=lifespan)


class TrainRequest(BaseModel):
    iterations: int


def _run_or_404(run_id: str) -> Run:
    run = state["trainer"].runs.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    return run


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/status")
async def status() -> JSONResponse:
    trainer: Trainer = state["trainer"]
    scene = trainer.scene
    return JSONResponse(
        {
            "current": trainer.current.summary() if trainer.current else None,
            "latest": trainer.latest_done.summary() if trainer.latest_done else None,
            "allowed_iterations": ALLOWED_ITERATIONS,
            "training_views": len(scene.images),
            "resolution": [scene.width, scene.height],
            "gpu": torch.cuda.get_device_name(0),
        }
    )


@app.post("/api/train")
async def train(req: TrainRequest) -> JSONResponse:
    if req.iterations not in ALLOWED_ITERATIONS:
        raise HTTPException(400, f"iterations must be one of {ALLOWED_ITERATIONS}")
    run = state["trainer"].start(req.iterations, enforce_cooldown=True)
    return JSONResponse({"run_id": run.id})


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str) -> StreamingResponse:
    run = _run_or_404(run_id)

    async def stream() -> AsyncIterator[str]:
        sent = 0
        while True:
            changed = run.changed
            while sent < len(run.events):
                yield f"data: {json.dumps(run.events[sent])}\n\n"
                sent += 1
            if run.done or run.error:
                return
            # Wait off the event loop; the timeout doubles as a keep-alive for proxies.
            if not await asyncio.to_thread(changed.wait, 15):
                yield ": keep-alive\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}/scene.splat")
async def scene_splat(run_id: str) -> Response:
    run = _run_or_404(run_id)
    if run.splat is None:
        raise HTTPException(404, "this run has not finished")
    return Response(
        run.splat,
        media_type="application/octet-stream",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/target.jpg")
async def target() -> Response:
    return Response(state["target_jpeg"], media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
