"""Render a scene's views: beauty image, true depth and material masks.

Run inside Blender, once per scene file:
    blender -b --factory-startup --disable-autoexec scene.blend --python-exit-code 1 \
        --python scripts/render_frames.py -- --views config/views.json --scene-id classroom --out frames

Each view gets a folder with:
    beauty.png          the Cycles render, as the scene's author set up its colours
    depth.npy           distance from the camera plane in metres, inf for sky
    depth.png           the same as inverse depth (near is white), the input for ControlNet depth
    material_index.npy  per-pixel material id, see meta.json for the names
    material_index.png  the same in random colours, to check by eye
    meta.json           camera, render settings, render time and visible materials
"""
import argparse
import json
import sys
import time
from math import radians
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix

BACKGROUND_DEPTH = 1e9


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", nargs="*", help="render only these view ids")
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--time-limit", type=float, default=0, help="seconds per view, 0 for no limit")
    return parser.parse_args(argv)


def enable_gpu(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        prefs.compute_device_type = device_type
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == device_type]
        if gpus:
            for device in prefs.devices:
                device.use = device.type == device_type
            scene.cycles.device = "GPU"
            return f"{device_type}: {gpus[0].name}"
    scene.cycles.device = "CPU"
    return "CPU"


def configure_render(scene, args):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.time_limit = args.time_limit
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    # Keep one view layer, so the render holds exactly one set of passes.
    # bpy.context.view_layer can point at a disabled one, so it is not used here.
    enabled = [layer for layer in scene.view_layers if layer.use]
    for layer in scene.view_layers:
        layer.use = layer == enabled[0]
    enabled[0].use_pass_z = True
    enabled[0].use_pass_position = True
    enabled[0].use_pass_material_index = True


def assign_material_indices():
    names = {}
    for index, material in enumerate(bpy.data.materials, start=1):
        material.pass_index = index
        names[index] = material.name
    return names


def view_matrix(camera, view):
    """Turn the source camera in place, tilt it, then move it along its own axes."""
    matrix = camera.matrix_world.copy()
    location = matrix.translation.copy()
    yaw = Matrix.Rotation(radians(view.get("yaw", 0)), 4, "Z")
    matrix = Matrix.Translation(location) @ yaw @ Matrix.Translation(-location) @ matrix
    matrix = matrix @ Matrix.Rotation(radians(view.get("pitch", 0)), 4, "X")
    return matrix @ Matrix.Translation(view.get("move", [0, 0, 0]))


def make_camera(scene, view):
    source = bpy.data.objects[view["camera"]]
    data = source.data.copy()
    data.dof.use_dof = False
    if "lens" in view:
        data.lens = view["lens"]
    camera = bpy.data.objects.new(f"benchmark_{view['id']}", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = view_matrix(source, view)
    scene.camera = camera
    return camera


def set_output_format(scene, media_type, file_format, color_depth):
    settings = scene.render.image_settings
    settings.media_type = media_type
    settings.file_format = file_format
    settings.color_depth = color_depth


def read_channel(image, names, suffix):
    matches = [i for i, name in enumerate(names) if name.endswith(suffix)]
    if not matches:
        raise RuntimeError(f"no channel ending in {suffix}, have: {names}")
    return image.read_image(0, 0, matches[0], matches[0] + 1, "float").squeeze()


def read_passes(path):
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(f"cannot read {path}: {oiio.geterror()}")
    names = list(image.spec().channelnames)
    depth_pass = read_channel(image, names, "Depth.Z")
    position = np.stack([read_channel(image, names, f"Position.{axis}") for axis in "XYZ"], axis=-1)
    material_index = read_channel(image, names, "Material Index.X")
    image.close()
    return depth_pass, position, material_index


def camera_depth(position, depth_pass, camera):
    """Distance from the camera plane, computed from world positions.

    The Depth pass alone is not enough. Some cameras stand behind a wall and use clip_start
    to see through it, and the values in that pass did not look like distances from the camera.
    """
    location = np.array(camera.matrix_world.translation)
    # normalized(), because a scaled camera object would otherwise scale the depth too
    forward = -np.array(camera.matrix_world.to_3x3().normalized().col[2])
    depth = (position - location) @ forward
    depth[depth_pass > BACKGROUND_DEPTH] = np.inf
    return depth.astype(np.float32)


def write_png(path, array):
    # OpenImageIO garbles a two-dimensional array, so grayscale also gets a channel axis.
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    spec = oiio.ImageSpec(array.shape[1], array.shape[0], array.shape[2], "uint8")
    output = oiio.ImageOutput.create(str(path))
    output.open(str(path), spec)
    output.write_image(array)
    output.close()


def depth_to_control(depth):
    """Inverse depth with outliers clipped, the convention of estimators such as MiDaS."""
    valid = np.isfinite(depth) & (depth > 0)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1.0 / depth[valid]
    low, high = np.percentile(inverse[valid], [1, 99])
    control = np.clip((inverse - low) / (high - low), 0, 1)
    control[~valid] = 0
    return (control * 255 + 0.5).astype(np.uint8)


def colorize_labels(labels):
    palette = np.random.default_rng(7).integers(40, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]


def visible_materials(labels, names):
    ids, counts = np.unique(labels, return_counts=True)
    shares = {
        names.get(int(i), "none"): round(float(c) / labels.size, 4)
        for i, c in zip(ids, counts)
    }
    return dict(sorted(shares.items(), key=lambda item: -item[1]))


def render_view(scene, view, args, device, material_names):
    # Absolute, because with a relative render path no EXR appeared in this folder.
    out_dir = Path(args.out).resolve() / view["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = make_camera(scene, view)

    exr_path = out_dir / "passes.exr"
    set_output_format(scene, "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32")
    scene.render.filepath = str(exr_path)
    started = time.time()
    bpy.ops.render.render(write_still=True)
    seconds = time.time() - started

    set_output_format(scene, "IMAGE", "PNG", "8")
    scene.render.image_settings.color_mode = "RGB"
    bpy.data.images["Render Result"].save_render(str(out_dir / "beauty.png"), scene=scene)

    depth_pass, position, material_index = read_passes(exr_path)
    exr_path.unlink()
    depth = camera_depth(position, depth_pass, camera)
    background = ~np.isfinite(depth)
    labels = np.rint(material_index).astype(np.int16)

    np.save(out_dir / "depth.npy", depth)
    np.save(out_dir / "material_index.npy", labels)
    write_png(out_dir / "depth.png", depth_to_control(depth))
    write_png(out_dir / "material_index.png", colorize_labels(labels))

    finite = depth[np.isfinite(depth)]
    meta = {
        "scene_id": args.scene_id,
        "view": view,
        "camera_matrix": [list(row) for row in camera.matrix_world],
        "lens_mm": camera.data.lens,
        "sensor_mm": camera.data.sensor_width,
        "clip_start": camera.data.clip_start,
        "camera_scale": [round(v, 3) for v in camera.matrix_world.to_scale()],
        "depth_minus_depth_pass": [
            round(float(np.percentile(depth[~background] - depth_pass[~background], p)), 3) for p in (1, 50, 99)
        ],
        "resolution": [args.width, args.height],
        "samples": args.samples,
        "device": device,
        "render_seconds": round(seconds, 1),
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "depth_range": [round(float(finite.min()), 3), round(float(finite.max()), 3)],
        "background_share": round(float(background.mean()), 4),
        "materials": visible_materials(labels, material_names),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VIEW_DONE {view['id']} {seconds:.1f}s depth {meta['depth_range']} on {device}")

    bpy.data.objects.remove(camera)


def main():
    args = parse_args()
    views = json.loads(Path(args.views).read_text(encoding="utf-8"))[args.scene_id]
    if args.only:
        views = [v for v in views if v["id"] in args.only]

    # Linked assets are read-only, and material indices have to be written to them.
    if bpy.data.libraries:
        bpy.ops.object.make_local(type="ALL")

    scene = bpy.context.scene
    device = enable_gpu(scene)
    configure_render(scene, args)
    material_names = assign_material_indices()
    for view in views:
        render_view(scene, view, args, device, material_names)


main()
