"""Put each frame's inputs and results side by side, one image per frame, to look through a run.

Usage:
    python scripts/contact_sheet.py --frames frames --run runs/01_baseline
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_WIDTH = 480
LABEL_HEIGHT = 28


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="frames")
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def make_cell(path, label):
    image = Image.open(path).convert("RGB")
    height = round(image.height * CELL_WIDTH / image.width)
    cell = Image.new("RGB", (CELL_WIDTH, height + LABEL_HEIGHT), "white")
    cell.paste(image.resize((CELL_WIDTH, height), Image.LANCZOS), (0, LABEL_HEIGHT))
    ImageDraw.Draw(cell).text((6, 4), label, fill="black", font=ImageFont.load_default(size=17))
    return cell


def make_sheet(rows):
    cells = [[make_cell(path, label) for path, label in row] for row in rows]
    width = max(len(row) for row in cells) * CELL_WIDTH
    height = sum(row[0].height for row in cells)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in cells:
        for x, cell in enumerate(row):
            sheet.paste(cell, (x * CELL_WIDTH, y))
        y += row[0].height
    return sheet


def main():
    args = parse_args()
    frames_dir = Path(args.frames)
    run_dir = Path(args.run)
    for frame_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        source = frames_dir / frame_dir.name
        rows = []
        depth_results = sorted(frame_dir.glob("depth_s*.png"))
        if depth_results:
            row = [(source / "beauty.png", "Blender render"), (source / "depth.png", "depth input")]
            rows.append(row + [(path, path.stem) for path in depth_results])
        canny_results = sorted(frame_dir.glob("canny_s*.png"))
        if canny_results:
            row = [(source / "material_index.png", "material masks"), (frame_dir / "edges.png", "canny input")]
            rows.append(row + [(path, path.stem) for path in canny_results])
        out_path = run_dir / f"sheet_{frame_dir.name}.jpg"
        make_sheet(rows).save(out_path, quality=88)
        print(out_path)


main()
