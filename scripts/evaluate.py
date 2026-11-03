"""Measure how well rendered images keep the source frame's geometry and materials.

For every image in runs/<run>/<frame>/*.png, compared with frames/<frame>/:
    edge_recall      share of true object and material boundaries with an image edge nearby
    edge_extra       share of image edges with no boundary or source edge nearby
    depth_error      relative error of depth estimated from the image against the true depth
    color_shift      colour difference per material from the source (CIE76 Delta E), weighted by area
    chroma_shift     the same with lightness left out, so only hue and saturation count
    lightness_shift  signed change in lightness per material, weighted by area
    pixel_change     mean colour difference per pixel from the source: how much changed at all

Every score here rewards closeness to the source, so none of them can say whether an image looks
more like a photograph. An image that changed nothing would score almost perfectly. pixel_change
shows how much was changed, and unlike the mean colours per material it also reacts to new texture.

The source render is measured too, as method "source", as a reference point: its geometry is correct
by definition, so its depth_error shows the error of the depth estimator itself.

Glass is left out of edge_extra, depth_error and the colour averages, because it shows whatever is
behind it. Scoring depth on glass would reward a result that turned a glass wall into a curtain.

Results go to metrics.csv and, per material, materials.csv in the run folder. With --overlays each
image also gets a picture of what the edge scores saw.

Usage:
    python scripts/evaluate.py --frames frames --run runs/01_baseline
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from kornia.color import rgb_to_lab
from kornia.filters import canny
from PIL import Image
from scipy.ndimage import distance_transform_edt, uniform_filter
from torch.nn.functional import interpolate
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"

CANNY_LOW = 0.1
CANNY_HIGH = 0.2
EDGE_TOLERANCE_PX = 3
DEPTH_JUMP = 0.05  # relative depth change between neighbouring pixels that counts as a boundary
DENSE_WINDOW_PX = 31
DENSE_SHARE = 0.2  # boundaries this dense are foliage or stochastic noise, not structure
MIN_MATERIAL_SHARE = 0.01

SCORES = ["edge_recall", "edge_extra", "depth_error", "color_shift", "chroma_shift", "lightness_shift", "pixel_change"]
METRIC_FIELDS = ["frame", "image", "method", "seed", "strength", "denoise"] + SCORES
MATERIAL_FIELDS = [
    "frame", "image", "method", "seed", "strength", "denoise", "material", "share", "counted",
    "delta_e", "chroma_shift", "lightness_shift", "pixel_delta_e", "lightness_spread", "source_lightness_spread",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="frames")
    parser.add_argument("--run", required=True)
    parser.add_argument("--only", nargs="*", help="evaluate only these frame ids")
    parser.add_argument("--overlays", action="store_true")
    return parser.parse_args()


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def to_tensor(rgb, device):
    return torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None]


def image_edges(rgb, device):
    _, edges = canny(to_tensor(rgb, device), low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    return edges[0, 0].cpu().numpy() > 0


def to_lab(rgb, device):
    return rgb_to_lab(to_tensor(rgb, device))[0].permute(1, 2, 0).cpu().numpy()


def depth_jumps(a, b):
    # Sky is inf: sky against sky gives nan, which compares as False, as it should.
    with np.errstate(invalid="ignore"):
        return np.abs(a - b) > DEPTH_JUMP * np.minimum(a, b)


def true_boundaries(depth, labels):
    """Pixels where Blender's depth jumps or the material changes, and the areas too dense to judge."""
    boundaries = np.zeros(depth.shape, dtype=bool)
    boundaries[:, :-1] |= depth_jumps(depth[:, :-1], depth[:, 1:]) | (labels[:, :-1] != labels[:, 1:])
    boundaries[:-1, :] |= depth_jumps(depth[:-1, :], depth[1:, :]) | (labels[:-1, :] != labels[1:, :])
    ignore = uniform_filter(boundaries.astype(np.float32), size=DENSE_WINDOW_PX) > DENSE_SHARE
    return boundaries & ~ignore, ignore


def material_names(labels, meta):
    """Rebuild id to name from meta.json, which lists names by pixel share, then by id."""
    ids, counts = np.unique(labels, return_counts=True)
    shares = np.round(counts / labels.size, 4)
    order = sorted(range(len(ids)), key=lambda i: (-shares[i], ids[i]))
    return {int(ids[i]): name for i, name in zip(order, meta["materials"])}


def is_glass(name):
    return "glass" in name.lower()


def load_frame(frame_dir, device):
    depth = np.load(frame_dir / "depth.npy")
    labels = np.load(frame_dir / "material_index.npy")
    meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
    rgb = load_rgb(frame_dir / "beauty.png")
    names = material_names(labels, meta)
    boundaries, ignore = true_boundaries(depth, labels)
    return {
        "id": frame_dir.name,
        "rgb": rgb,
        "lab": to_lab(rgb, device),
        "edges": image_edges(rgb, device),
        "depth": depth,
        "labels": labels,
        "names": names,
        "boundaries": boundaries,
        "ignore": ignore,
        "glass": np.isin(labels, [material_id for material_id, name in names.items() if is_glass(name)]),
    }


def edge_scores(edges, frame):
    distance_to_edge = distance_transform_edt(~edges)
    kept = distance_to_edge <= EDGE_TOLERANCE_PX
    known = frame["boundaries"] | frame["edges"]
    counted = edges & ~frame["ignore"] & ~frame["glass"]
    extra = counted & (distance_transform_edt(~known) > EDGE_TOLERANCE_PX)
    recall = float(kept[frame["boundaries"]].mean())
    share_extra = float(extra.sum() / max(counted.sum(), 1))
    return recall, share_extra, kept, extra


def estimate_depth(rgb, processor, model, device):
    """Relative inverse depth from the image, at the image's size."""
    image = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth
    resized = interpolate(predicted[None], size=rgb.shape[:2], mode="bicubic", align_corners=False)
    return resized[0, 0].cpu().numpy()


def depth_error(estimate, frame):
    """Mean relative error, after fitting the scale and shift that relative estimators leave open."""
    valid = np.isfinite(frame["depth"]) & ~frame["ignore"] & ~frame["glass"]
    true_depth = frame["depth"][valid]
    target = 1.0 / true_depth
    design = np.stack([estimate[valid], np.ones(target.size)], axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    aligned = np.clip(scale * estimate[valid] + shift, target.min() / 2, target.max() * 2)
    return float(np.mean(np.abs(1.0 / aligned - true_depth) / true_depth))


def material_scores(lab, frame):
    rows = []
    for material_id, name in frame["names"].items():
        mask = frame["labels"] == material_id
        share = float(mask.mean())
        if material_id == 0 or share < MIN_MATERIAL_SHARE:
            continue
        pixels, source_pixels = lab[mask], frame["lab"][mask]
        difference = pixels.mean(axis=0) - source_pixels.mean(axis=0)
        rows.append({
            "material": name,
            "share": round(share, 4),
            "counted": not is_glass(name),
            "delta_e": round(float(np.linalg.norm(difference)), 2),
            "chroma_shift": round(float(np.linalg.norm(difference[1:])), 2),
            "lightness_shift": round(float(difference[0]), 2),
            "pixel_delta_e": round(float(np.linalg.norm(pixels - source_pixels, axis=1).mean()), 2),
            "lightness_spread": round(float(np.ptp(np.percentile(pixels[:, 0], [5, 95]))), 2),
            "source_lightness_spread": round(float(np.ptp(np.percentile(source_pixels[:, 0], [5, 95]))), 2),
        })
    return rows


def weighted_mean(rows, field):
    counted = [row for row in rows if row["counted"]]
    weights = sum(row["share"] for row in counted)
    return sum(row[field] * row["share"] for row in counted) / weights if weights else 0.0


def image_settings(path):
    settings = {"method": "", "seed": "", "strength": "", "denoise": ""}
    workflow = json.loads(Image.open(path).info.get("prompt", "{}"))
    for node in workflow.values():
        kind, inputs = node.get("class_type"), node.get("inputs", {})
        if kind == "SetUnionControlNetType":
            settings["method"] = "depth" if inputs["type"] == "depth" else "canny"
        elif kind == "KSampler":
            settings["seed"] = inputs["seed"]
            settings["denoise"] = inputs["denoise"]
        elif kind == "ControlNetApplyAdvanced":
            settings["strength"] = inputs["strength"]
    return settings


def save_overlay(path, rgb, frame, kept, extra):
    """Kept boundaries in green, lost boundaries in red, edges with no basis in the source in yellow."""
    canvas = (rgb * 0.45 * 255).astype(np.uint8)
    canvas[frame["boundaries"] & kept] = (60, 220, 90)
    canvas[frame["boundaries"] & ~kept] = (235, 60, 60)
    canvas[extra] = (250, 220, 40)
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(canvas).save(path, quality=90)


def evaluate_image(rgb, settings, name, frame, processor, model, device):
    recall, share_extra, kept, extra = edge_scores(image_edges(rgb, device), frame)
    lab = to_lab(rgb, device)
    materials = material_scores(lab, frame)
    row = {
        "frame": frame["id"],
        "image": name,
        **settings,
        "edge_recall": round(recall, 4),
        "edge_extra": round(share_extra, 4),
        "depth_error": round(depth_error(estimate_depth(rgb, processor, model, device), frame), 4),
        "color_shift": round(weighted_mean(materials, "delta_e"), 2),
        "chroma_shift": round(weighted_mean(materials, "chroma_shift"), 2),
        "lightness_shift": round(weighted_mean(materials, "lightness_shift"), 2),
        "pixel_change": round(float(np.linalg.norm(lab - frame["lab"], axis=2).mean()), 2),
    }
    identity = {"frame": frame["id"], "image": name, **settings}
    return row, [{**identity, **material} for material in materials], kept, extra


def print_summary(rows):
    frames = sorted({row["frame"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    print(f"{'frame':20} {'method':8} {'n':>3} " + " ".join(f"{score:>15}" for score in SCORES))
    for frame in frames + ["ALL"]:
        for method in methods:
            group = [r for r in rows if r["method"] == method and frame in (r["frame"], "ALL")]
            if group:
                means = " ".join(f"{np.mean([r[score] for r in group]):15.3f}" for score in SCORES)
                print(f"{frame:20} {method:8} {len(group):3} {means}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, revision=DEPTH_MODEL_REVISION)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, revision=DEPTH_MODEL_REVISION).to(device).eval()

    run_dir = Path(args.run)
    rows, material_rows = [], []
    for image_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        if args.only and image_dir.name not in args.only:
            continue
        frame = load_frame(Path(args.frames) / image_dir.name, device)
        source_settings = {"method": "source", "seed": "", "strength": "", "denoise": ""}
        jobs = [("source", frame["rgb"], source_settings)]
        for path in sorted(image_dir.glob("*.png")):
            if path.name != "edges.png":
                jobs.append((path.stem, load_rgb(path), image_settings(path)))

        for name, rgb, settings in jobs:
            row, materials, kept, extra = evaluate_image(rgb, settings, name, frame, processor, model, device)
            rows.append(row)
            material_rows.extend(materials)
            if args.overlays:
                save_overlay(image_dir / "overlays" / f"{name}.jpg", rgb, frame, kept, extra)
            print(f"{frame['id']} {name}: " + " ".join(f"{score} {row[score]}" for score in SCORES), flush=True)

    for filename, fields, table in (("metrics.csv", METRIC_FIELDS, rows), ("materials.csv", MATERIAL_FIELDS, material_rows)):
        with open(run_dir / filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    print_summary(rows)


main()
