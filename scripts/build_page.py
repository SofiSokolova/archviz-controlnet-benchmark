"""Build the compressed images and data for the before and after page in docs/.

Every frame shows seed 1 of both methods, so the comparison images are not picked by hand.
The failure examples and the denoise series are chosen on purpose, and the page says so.

Run from the repository root, after evaluate.py has scored the runs:
    python scripts/build_page.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path("docs")
IMAGES = DOCS / "images"
WEBP_QUALITY = 82

BASELINE = Path("runs/01_baseline")
FROM_RENDER = Path("runs/03_img2img")
HAND_RUNS = Path("runs/02_hands_on")
TERRACE = "barcelona_terrace"

SCORES = ["edge_recall", "edge_extra", "depth_error", "color_shift", "pixel_change"]
METHODS = ["canny", "depth"]
FRAMES = [
    ("barcelona_facade", "Pavilion facade"),
    ("barcelona_terrace", "Pavilion terrace"),
    ("classroom_main", "Classroom"),
    ("classroom_left", "Classroom, second view"),
    ("flat_kitchen", "Kitchen"),
    ("flat_living", "Living room"),
]
DENOISE_SERIES = [
    ("1.0", "terrace_canny_00001_", "From noise. The lighting and the colour of the glass changed, but overall the frame feels improved."),
    ("0.8", "terrace_img2img_00003_", "Worse than 0.6. The column is round, the windows look strange and the floor is glossier."),
    ("0.6", "terrace_img2img_00001_", "The column became round and extra hatches appeared on the ceiling. The floor is less glossy, but has random lines."),
    ("0.5", "terrace_img2img_00004_", "More like a photo than a 3D render. Depth feels stronger. The floor is acceptable, but the colour seems a bit dark to me."),
    ("0.4", "terrace_img2img_00002_", "Closer to the original and even more like a photo. The column is no longer round, the hatches are gone and the floor is practically identical."),
]
HAND_RUN_RESULTS = {
    "terrace_structure": "terrace_canny_00014_",
    "terrace_wording": "terrace_canny_00009_",
}
FAILURES = [
    {
        "title": "Materials travel",
        "text": "Red marble from the wall's description spread onto the pale stone floor, in all six images of this "
                "frame. The model never sees the source colours, only the words. Starting from the render brought "
                "the chroma shift on this floor from 11.9 down to 1.1.",
        "frame": "barcelona_facade",
        "result": "barcelona_facade_canny_noise",
        "divider": 30,
    },
    {
        "title": "Glass becomes a curtain",
        "text": "A depth map shows glass as a plain flat wall, so the model decides what it is. All three depth images "
                "of the terrace lost the glass, while all three canny images kept it.",
        "frame": "barcelona_terrace",
        "result": "barcelona_terrace_depth_noise",
        "divider": 40,
    },
    {
        "title": "A cork board that is never cork",
        "text": "The cork board became a chalkboard in five of six images and a blank wall in the sixth. Starting from "
                "the render brought its colour difference from 25.8 down to 1.3.",
        "frame": "classroom_left",
        "result": "classroom_left_canny_noise",
        "divider": 12,
    },
    {
        "title": "Furniture out of nowhere",
        "text": "I added words against a glossy floor to the negative prompt and forgot to set the seed back, and a "
                "strange structure made of chairs appeared. Its depth error of 0.160 is the highest of all my "
                "hand-run images, so the score does notice it.",
        "frame": "barcelona_terrace",
        "result": "terrace_structure",
        "divider": 8,
    },
    {
        "title": "Words only go so far",
        "text": "I tried seven wordings for a matte stone floor. The floor colour moved closer to the source, from a "
                "difference of 28.9 to 20.8, but that was an improvement and not a fix.",
        "frame": "barcelona_terrace",
        "result": "terrace_wording",
        "divider": 25,
    },
]


def read_rows(path):
    with open(path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def export_image(path, name):
    """Save a compressed copy in docs/images and return its path from docs/."""
    Image.open(path).convert("RGB").save(IMAGES / f"{name}.webp", "WEBP", quality=WEBP_QUALITY, method=6)
    return f"images/{name}.webp"


def scores_of(row):
    return {score: float(row[score]) for score in SCORES}


def find_row(rows, frame, image):
    return next(row for row in rows if row["frame"] == frame and row["image"] == image)


def build_frames(noise_rows, render_rows):
    frames = []
    for frame_id, title in FRAMES:
        results = {}
        for method in METHODS:
            image = f"{method}_s1"
            results[method] = {
                "noise": {
                    "image": export_image(BASELINE / frame_id / f"{image}.png", f"{frame_id}_{method}_noise"),
                    "scores": scores_of(find_row(noise_rows, frame_id, image)),
                },
                "render": {
                    "image": export_image(FROM_RENDER / frame_id / f"{image}.png", f"{frame_id}_{method}_render"),
                    "scores": scores_of(find_row(render_rows, frame_id, image)),
                },
            }
        frames.append({
            "id": frame_id,
            "title": title,
            "source": export_image(Path("frames") / frame_id / "beauty.png", f"{frame_id}_source"),
            "sourceScores": scores_of(find_row(render_rows, frame_id, "source")),
            "results": results,
        })
    return frames


def mean_score(rows, method, score):
    return float(np.mean([float(row[score]) for row in rows if row["method"] == method]))


def build_summary(noise_rows, render_rows):
    return {
        score: {
            "source": mean_score(render_rows, "source", score),
            "cannyNoise": mean_score(noise_rows, "canny", score),
            "cannyRender": mean_score(render_rows, "canny", score),
            "depthNoise": mean_score(noise_rows, "depth", score),
            "depthRender": mean_score(render_rows, "depth", score),
        }
        for score in SCORES
    }


def build_series():
    metrics = read_rows(HAND_RUNS / "metrics.csv")
    materials = read_rows(HAND_RUNS / "materials.csv")

    def floor_of(image):
        return next(row for row in materials if row["image"] == image and row["material"] == "pavet")

    steps = []
    for denoise, image, note in DENOISE_SERIES:
        floor = floor_of(image)
        steps.append({
            "denoise": denoise,
            "image": export_image(HAND_RUNS / TERRACE / f"{image}.png", f"terrace_denoise_{denoise.replace('.', '')}"),
            "note": note,
            "scores": scores_of(find_row(metrics, TERRACE, image)),
            "floorDeltaE": float(floor["delta_e"]),
        })
    return {"steps": steps, "source": f"images/{TERRACE}_source.webp"}


def build_failures():
    for name, image in HAND_RUN_RESULTS.items():
        export_image(HAND_RUNS / TERRACE / f"{image}.png", name)
    return [
        {
            "title": failure["title"],
            "text": failure["text"],
            "source": f"images/{failure['frame']}_source.webp",
            "result": f"images/{failure['result']}.webp",
            "divider": failure["divider"],  # starts where the failure is in view
        }
        for failure in FAILURES
    ]


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    noise_rows = read_rows(BASELINE / "metrics.csv")
    render_rows = read_rows(FROM_RENDER / "metrics.csv")
    data = {
        "frames": build_frames(noise_rows, render_rows),
        "summary": build_summary(noise_rows, render_rows),
        "series": build_series(),
        "failures": build_failures(),
    }
    (DOCS / "data.js").write_text(f"window.BENCHMARK = {json.dumps(data, indent=1)};\n", encoding="utf-8")
    sizes = [path.stat().st_size for path in IMAGES.glob("*.webp")]
    print(f"{len(sizes)} images, {sum(sizes) / 1e6:.1f} MB in total, largest {max(sizes) / 1e3:.0f} KB")


main()
