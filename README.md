# Gaussian splatting, trained live on Chalk GPUs

Train a 3D Gaussian Splatting scene from 100 posed photos in about half a minute on one NVIDIA L4,
watch it converge in the browser, then orbit the result with a WebGL splat renderer.

```
100 photos + camera poses ─▶ gsplat on an L4 (7,000 steps, ~36 s) ─▶ 100k Gaussians ─▶ .splat ─▶ browser
```

| | |
| --- | --- |
| Scene | [Treasure Chest](https://polyhaven.com/a/treasure_chest) by Rico Cilliers, CC0, from Poly Haven |
| Training views | 100 renders at 400×400 on a hemisphere around the object, plus 12 held-out test views |
| Trainer | [gsplat](https://github.com/nerfstudio-project/gsplat)'s CUDA rasterizer and densification strategy |
| Result | ~100,000 Gaussians at ~38 dB PSNR on the held-out views, after 7,000 steps |
| Viewer | [Spark](https://github.com/sparkjsdev/spark) on three.js, loading the trained `.splat` |

## Files

| File | Role |
| --- | --- |
| `render_views.py` | Renders the training set with Blender's Cycles: cameras on a Fibonacci spiral, NeRF-style `transforms.json`. |
| `data/treasure_chest/` | The rendered views and poses, so you can train without Blender. |
| `app/splat_trainer.py` | The trainer: random initial Gaussians, L1 + SSIM loss, gsplat's `DefaultStrategy` to split, clone and prune. About 250 lines. |
| `app/server.py` | FastAPI: one training run at a time, progress over Server-Sent Events, the result as `.splat`. |
| `app/index.html` | The page: live preview beside a held-out photo, PSNR curve, and the 3D viewer. |
| `deploy.py` | Deploys the whole thing as a `chalkcompute.ScalingGroup` on one GPU. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule an L4. Credentials
come from `chalk config` in the current directory, or from `CHALK_API_SERVER`, `CHALK_CLIENT_ID`,
`CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`.

```bash
./deploy.py --gpu nvidia-l4
```

The first deploy builds a 5.5 GB CUDA image, which takes about eight minutes. The replica trains
once as it boots, so the printed URL opens on a finished scene. Press **Train from scratch** to
watch a new run converge.

To train your own scene, render it (or bring real photos with known poses) into the same layout:

```bash
./render_views.py --model your_model.gltf --out data/your_scene
./deploy.py --scene data/your_scene
```

## How the trainer works

Every Gaussian has a position, a 3D scale, a rotation, an opacity and a color. gsplat projects them
into a camera, sorts them by depth per 16×16 tile, and alpha-blends them into an image. The whole
rasterizer is differentiable, so each step:

1. picks a training photo and renders the Gaussians from that photo's camera,
2. compares the render with the photo (0.8 × L1 + 0.2 × (1 − SSIM)),
3. backpropagates into every Gaussian's parameters and takes an Adam step.

Starting from 20,000 random Gaussians in a cube, the densification strategy splits Gaussians that
are too large, clones ones that are too small where gradients stay high, and prunes ones that turn
transparent. That growth is what turns a cloud of blobs into fine detail.

## API

| Method | Path | |
| --- | --- | --- |
| `GET` | `/api/status` | Current and latest runs, GPU name |
| `POST` | `/api/train` | `{"iterations": 2000 \| 4000 \| 7000}`; 409 while a run is in progress |
| `GET` | `/api/runs/{id}/events` | Server-Sent Events: step, loss, Gaussian count, PSNR, a JPEG preview |
| `GET` | `/api/runs/{id}/scene.splat` | The trained scene, 32 bytes per Gaussian |

## Notes

- gsplat publishes prebuilt CUDA wheels for Python 3.10 + torch 2.4 + CUDA 12.4, which is why the
  image pins exactly those: nothing compiles at build or start time.
- The `.splat` export stores one color per Gaussian, so the trainer uses plain RGB rather than
  view-dependent spherical harmonics. That keeps the browser file small at a small cost in sheen.
- The page is public. Delete the group when you are done:
  `uv run --with chalkcompute python -c "from chalkcompute import ScalingGroup; ScalingGroup.from_name('gaussian-splatting').delete()"`

## License

The code is Apache-2.0. The Treasure Chest model and the renders of it are CC0. gsplat is
Apache-2.0, and Spark and three.js are MIT.
