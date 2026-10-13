#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = "==3.11.*"
# dependencies = ["bpy==4.2.0"]
# ///
"""Render a multi-view training set of a glTF model with Blender's Cycles.

Writes NeRF-synthetic-style data: `images/*.png` plus `transforms.json`, whose
`transform_matrix` is each camera's camera-to-world pose in Blender's convention
(camera looks down -Z, +Y up).

    ./render_views.py --model treasure_chest_1k.gltf --out data/treasure_chest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def normalize(meshes: list[bpy.types.Object]) -> None:
    """Center the model at the origin and scale its largest dimension to 2."""
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    center = (lo + hi) / 2
    scale = 2.0 / max(hi - lo)
    roots = {o for o in bpy.context.scene.objects if o.parent is None}
    for root in roots:
        root.location = (root.location - center) * scale
        root.scale = root.scale * scale


def setup_render(resolution: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = 0.6

    for i, (rot, energy) in enumerate([((50, 0, 30), 3.5), ((60, 0, 210), 1.5)]):
        light = bpy.data.lights.new(f"sun{i}", type="SUN")
        light.energy = energy
        obj = bpy.data.objects.new(f"sun{i}", light)
        obj.rotation_euler = tuple(math.radians(a) for a in rot)
        scene.collection.objects.link(obj)


def camera_positions(n: int, radius: float, min_elev: float, max_elev: float) -> list[Vector]:
    """Points on a spherical band, spread evenly with a Fibonacci spiral."""
    golden = math.pi * (3 - math.sqrt(5))
    lo, hi = math.sin(math.radians(min_elev)), math.sin(math.radians(max_elev))
    out = []
    for i in range(n):
        z = lo + (hi - lo) * (i + 0.5) / n
        r = math.sqrt(1 - z * z)
        theta = golden * i
        out.append(Vector((r * math.cos(theta), r * math.sin(theta), z)) * radius)
    return out


def render_split(cam: bpy.types.Object, positions: list[Vector], out_dir: Path, split: str) -> list[dict]:
    frames = []
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    for i, pos in enumerate(positions):
        cam.location = pos
        cam.rotation_euler = (-pos).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        rel = f"{split}/{i:03d}.png"
        bpy.context.scene.render.filepath = str(out_dir / rel)
        bpy.ops.render.render(write_still=True)
        frames.append({"file_path": rel, "transform_matrix": [list(row) for row in cam.matrix_world]})
        print(f"{split} {i + 1}/{len(positions)}", flush=True)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()

    reset_scene()
    normalize(import_model(args.model))
    setup_render(args.resolution, args.samples)

    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(40)
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    radius = 4.0
    train = render_split(cam, camera_positions(args.train, radius, 8, 75), args.out, "train")
    # Test views sit between the training elevations so they are genuinely unseen.
    test = render_split(cam, camera_positions(args.test, radius * 1.05, 20, 60), args.out, "test")
    meta = {"camera_angle_x": cam_data.angle, "train": train, "test": test}
    (args.out / "transforms.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
