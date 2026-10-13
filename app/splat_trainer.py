"""A compact 3D Gaussian Splatting trainer built on gsplat's CUDA rasterizer.

The scene is a set of posed photos (NeRF-synthetic `transforms.json` layout). Training starts
from random Gaussians in a cube around the object and optimizes their positions, sizes,
orientations, colors and opacities so that rasterizing them from each training camera
reproduces that camera's photo. gsplat's DefaultStrategy splits, clones and prunes Gaussians
along the way, which is how a few thousand blobs grow into a detailed model.
"""

from __future__ import annotations

import io
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Blender cameras look down -Z with +Y up; gsplat expects OpenCV cameras (+Z forward, +Y down).
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Scene:
    images: torch.Tensor  # [N, H, W, 3] in [0, 1]
    viewmats: torch.Tensor  # [N, 4, 4] world-to-camera
    test_images: torch.Tensor
    test_viewmats: torch.Tensor
    K: torch.Tensor  # [3, 3]
    width: int
    height: int
    background: torch.Tensor  # [3]


@dataclass(frozen=True)
class TrainConfig:
    iterations: int
    initial_gaussians: int
    preview_every: int
    eval_every: int


@dataclass(frozen=True)
class Progress:
    step: int
    iterations: int
    loss: float
    gaussians: int
    elapsed_s: float
    psnr: float | None
    preview_jpeg: bytes | None


def viewmat_from_blender(c2w: list[list[float]]) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64) @ BLENDER_TO_OPENCV)


def intrinsics(fov_x: float, width: int, height: int) -> np.ndarray:
    focal = 0.5 * width / math.tan(0.5 * fov_x)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def _load_split(root: Path, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    images = np.stack([np.asarray(Image.open(root / f["file_path"]).convert("RGB"), np.float32) / 255 for f in frames])
    viewmats = np.stack([viewmat_from_blender(f["transform_matrix"]) for f in frames])
    return images, viewmats


def load_scene(root: Path, device: torch.device) -> Scene:
    meta = json.loads((root / "transforms.json").read_text())
    train_images, train_viewmats = _load_split(root, meta["train"])
    test_images, test_viewmats = _load_split(root, meta["test"])
    height, width = train_images.shape[1:3]
    # The renders sit on a flat backdrop; its color is the rasterizer's background.
    background = np.median(train_images[:, :4, :4].reshape(-1, 3), axis=0)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32, device=device)

    return Scene(
        images=t(train_images),
        viewmats=t(train_viewmats),
        test_images=t(test_images),
        test_viewmats=t(test_viewmats),
        K=t(intrinsics(meta["camera_angle_x"], width, height)),
        width=width,
        height=height,
        background=t(background),
    )


def init_params(n: int, device: torch.device) -> torch.nn.ParameterDict:
    """Random Gaussians in a cube that bounds the (unit-normalized) object."""
    means = (torch.rand(n, 3, device=device) * 2 - 1) * 1.2
    quats = F.normalize(torch.randn(n, 4, device=device), dim=-1)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.02), device=device)),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=device))),
            "colors": torch.nn.Parameter(torch.logit(torch.rand(n, 3, device=device) * 0.8 + 0.1)),
        }
    )


# Positions scale with the scene: 1.6e-4 per unit of extent, and the cameras orbit at radius ~4.4.
LEARNING_RATES = {"means": 1.6e-4 * 4.4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}


def render(params: torch.nn.ParameterDict, viewmats: torch.Tensor, scene: Scene) -> tuple[torch.Tensor, dict]:
    from gsplat import rasterization

    c = viewmats.shape[0]
    colors, _alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.sigmoid(params["colors"]),
        viewmats=viewmats,
        Ks=scene.K.expand(c, 3, 3),
        width=scene.width,
        height=scene.height,
        backgrounds=scene.background.expand(c, 3),
        # Packed mode mis-shapes per-camera backgrounds in gsplat 1.5.3, and DefaultStrategy is
        # told packed=False below, so the two must agree.
        packed=False,
    )
    return colors.clamp(0, 1), info


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of two [H, W, 3] images, with the standard 11-pixel Gaussian window."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    w = _gaussian_window(11, 1.5, a.device)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x**2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def to_jpeg(image: torch.Tensor, quality: int) -> bytes:
    arr = (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def train(scene: Scene, config: TrainConfig, seed: int) -> Iterator[Progress | torch.nn.ParameterDict]:
    """Yields Progress as training runs, then the trained parameters as the final item."""
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(seed)
    device = scene.images.device
    params = init_params(config.initial_gaussians, device)
    optimizers = {name: torch.optim.Adam([params[name]], lr=lr, eps=1e-15) for name, lr in LEARNING_RATES.items()}
    # Position steps decay exponentially to 1% so late iterations refine rather than move.
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / config.iterations))
    strategy = DefaultStrategy(
        refine_start_iter=300,
        refine_stop_iter=int(config.iterations * 0.75),
        refine_every=100,
        reset_every=2000,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    rng = torch.Generator().manual_seed(seed)
    start = time.perf_counter()

    for step in range(1, config.iterations + 1):
        i = int(torch.randint(len(scene.images), (1,), generator=rng))
        rendered, info = render(params, scene.viewmats[i : i + 1], scene)
        target = scene.images[i]
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = 0.8 * F.l1_loss(rendered[0], target) + 0.2 * (1 - ssim(rendered[0], target))
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_decay.step()

        last = step == config.iterations
        if step == 1 or step % config.preview_every == 0 or last:
            with torch.no_grad():
                score = None
                if step % config.eval_every == 0 or last:
                    renders, _ = render(params, scene.test_viewmats, scene)
                    score = sum(psnr(r, t) for r, t in zip(renders, scene.test_images)) / len(renders)
                preview, _ = render(params, scene.test_viewmats[:1], scene)
            yield Progress(
                step=step,
                iterations=config.iterations,
                loss=float(loss),
                gaussians=int(params["means"].shape[0]),
                elapsed_s=round(time.perf_counter() - start, 2),
                psnr=None if score is None else round(score, 2),
                preview_jpeg=to_jpeg(preview[0], 80),
            )
    yield params


def to_splat(params: torch.nn.ParameterDict) -> bytes:
    """Serializes Gaussians in the widely supported 32-byte `.splat` layout.

    Per Gaussian: position (3×f32), scale (3×f32), RGBA (4×u8), rotation quaternion wxyz
    (4×u8, mapped from [-1, 1] to [0, 255]). Ordered largest-and-most-opaque first, which lets
    viewers show a coarse result while the rest streams in.
    """
    with torch.no_grad():
        means = params["means"].float().cpu().numpy()
        scales = torch.exp(params["scales"]).float().cpu().numpy()
        rgb = torch.sigmoid(params["colors"]).float().cpu().numpy()
        alpha = torch.sigmoid(params["opacities"]).float().cpu().numpy()
        quats = F.normalize(params["quats"], dim=-1).float().cpu().numpy()
    order = np.argsort(-(scales.prod(axis=1) * alpha))
    out = np.zeros(len(order), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    out["pos"] = means[order]
    out["scale"] = scales[order]
    out["rgba"] = np.clip(np.concatenate([rgb, alpha[:, None]], axis=1)[order] * 255, 0, 255)
    out["rot"] = np.clip(quats[order] * 128 + 128, 0, 255)
    return out.tobytes()
