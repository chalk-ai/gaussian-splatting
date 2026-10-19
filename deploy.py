#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the live Gaussian Splatting trainer as a Chalk ScalingGroup on one GPU.

The replica loads the rendered views from `data/treasure_chest` (see render_views.py),
trains once at boot, and serves a page where visitors watch a run and orbit the result.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# gsplat publishes prebuilt CUDA wheels for Python 3.10 + torch 2.4 + CUDA 12.4, which avoids
# compiling its kernels (and needing nvcc) in the image. Everything installs in one step so the
# image builder snapshots the multi-gigabyte CUDA layer once. The PyTorch index is an *extra*
# index: as the only index, its mirror of typing-extensions fails pip's name check.
INSTALL = " && ".join(
    [
        "pip install --no-cache-dir torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124",
        "pip install --no-cache-dir ninja jaxtyping rich gsplat==1.5.3"
        + " --extra-index-url https://docs.gsplat.studio/whl/pt24cu124",
        "pip install --no-cache-dir fastapi==0.116.1 'uvicorn[standard]==0.35.0' pillow==11.3.0",
    ]
)


def main() -> None:
    if not (HERE / 'data' / 'treasure_chest' / "transforms.json").exists():
        raise SystemExit(f"{HERE / 'data' / 'treasure_chest'} has no transforms.json; run render_views.py first")

    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.10")
        .run_commands(INSTALL)
        # Mounted at start rather than copied: the views are ~20 MB and the code is iterated on.
        .add_local_dir(str(HERE / 'data' / 'treasure_chest'), "/data/treasure_chest", strategy="volume")
        .add_local_dir(str(HERE / "app"), "/app", strategy="volume")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name='gaussian-splatting',
        gpu='nvidia-l4',
        cpu="4",
        memory="16Gi",
        env={"SCENE_DIR": "/data/treasure_chest"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/api/status"),
        # One GPU trains one scene at a time; the server queues nothing and answers 409 instead.
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    sg.deploy(build_timeout=2400.0, ready_timeout=1800.0)
    print(f"ready in {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
