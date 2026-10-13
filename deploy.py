#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the live Gaussian Splatting trainer as a Chalk ScalingGroup on one GPU.

The replica loads the rendered views from `data/treasure_chest` (see render_views.py),
trains once at boot, and serves a page where visitors watch a run and orbit the result.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="gaussian-splatting")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--scene", type=Path, default=HERE / "data" / "treasure_chest")
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if not (args.scene / "transforms.json").exists():
        raise SystemExit(f"{args.scene} has no transforms.json; run render_views.py first")

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.10")
        .run_commands(INSTALL)
        # Mounted at start rather than copied: the views are ~20 MB and the code is iterated on.
        .add_local_dir(str(args.scene), "/data/treasure_chest", strategy="volume")
        .add_local_dir(str(HERE / "app"), "/app", strategy="volume")
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
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
    sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    print(f"ready in {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
