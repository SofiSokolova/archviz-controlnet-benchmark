"""Render frames through a running ComfyUI with each ControlNet method and collect the results.

Uses only the ComfyUI HTTP API, so the workflow built here is the whole setup. Results that
already exist are skipped, so an interrupted run can be started again with the same command.

Usage:
    python scripts/run_comfy.py --frames frames --prompts config/prompts.json --run runs/01_baseline
"""
import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests

CHECKPOINT = "sd_xl_base_1.0_0.9vae.safetensors"
CONTROLNET = "controlnet-union-sdxl-1.0-promax.safetensors"
UNION_TYPES = {"depth": "depth", "canny": "canny/lineart/anime_lineart/mlsd"}
CSV_FIELDS = [
    "frame", "method", "seed", "seconds", "peak_vram_mb", "peak_temp_c", "min_free_ram_gb",
    "busy_clock_mhz", "thermal_slowdown_share", "file",
]
GPU_QUERY = (
    "memory.used,temperature.gpu,utilization.gpu,clocks.current.graphics,"
    "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown"
)
BUSY_GPU_PERCENT = 80


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="frames")
    parser.add_argument("--prompts", default="config/prompts.json")
    parser.add_argument("--run", required=True, help="folder for this run's results")
    parser.add_argument("--only", nargs="*", help="render only these frame ids")
    parser.add_argument("--methods", nargs="+", default=["depth", "canny"], choices=sorted(UNION_TYPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=6.5)
    parser.add_argument("--sampler", default="dpmpp_2m")
    parser.add_argument("--scheduler", default="karras")
    parser.add_argument("--strength", type=float, default=0.8, help="ControlNet strength")
    parser.add_argument("--denoise", type=float, default=1.0, help="below 1, start from the render instead of noise")
    parser.add_argument("--canny-low", type=float, default=0.2)
    parser.add_argument("--canny-high", type=float, default=0.5)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    return parser.parse_args()


class HardwareMonitor:
    """Samples GPU memory, temperature, clock and free RAM once a second."""

    def __init__(self):
        self.reset()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def reset(self):
        self.peak_vram_mb = 0
        self.peak_temp_c = 0
        self.min_free_ram_gb = float("inf")
        self._busy_clocks = []
        self._busy_slowdowns = 0

    def _sample(self):
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        values = [value.strip() for value in output.strip().split(",")]
        vram, temp, load, clock = (int(value) for value in values[:4])
        self.peak_vram_mb = max(self.peak_vram_mb, vram)
        self.peak_temp_c = max(self.peak_temp_c, temp)
        self.min_free_ram_gb = min(self.min_free_ram_gb, psutil.virtual_memory().available / 1024**3)
        # Between renders the clock drops on its own, so only busy moments say anything about throttling.
        if load >= BUSY_GPU_PERCENT:
            self._busy_clocks.append(clock)
            self._busy_slowdowns += "Active" in values[4:]

    def busy_clock_mhz(self):
        return round(sum(self._busy_clocks) / len(self._busy_clocks)) if self._busy_clocks else 0

    def thermal_slowdown_share(self):
        return round(self._busy_slowdowns / len(self._busy_clocks), 3) if self._busy_clocks else 0

    def _loop(self):
        while not self._stop.wait(1.0):
            self._sample()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


def upload_image(url, path, name):
    """Copy an image into ComfyUI's input folder and return the name LoadImage expects."""
    with open(path, "rb") as image:
        response = requests.post(
            f"{url}/upload/image",
            files={"image": (name, image, "image/png")},
            data={"subfolder": "benchmark", "type": "input", "overwrite": "true"},
            timeout=60,
        )
    response.raise_for_status()
    return f"benchmark/{name}"


def build_workflow(args, method, control_image, start_image, positive, negative, seed, size, prefix, save_edges_as):
    """Text prompt plus one ControlNet. Starting from noise, the model invents everything the control
    image leaves open. With denoise below 1 it starts from the render, so it also sees the source colours.
    """
    workflow = {
        "checkpoint": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["checkpoint", 1]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["checkpoint", 1]}},
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": size[0], "height": size[1], "batch_size": 1},
        },
        "control_source": {"class_type": "LoadImage", "inputs": {"image": control_image}},
        "controlnet": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "controlnet_type": {
            "class_type": "SetUnionControlNetType",
            "inputs": {"control_net": ["controlnet", 0], "type": UNION_TYPES[method]},
        },
        "apply_controlnet": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "control_net": ["controlnet_type", 0],
                "image": ["control_source", 0],
                "strength": args.strength,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "vae": ["checkpoint", 2],
            },
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["checkpoint", 0],
                "positive": ["apply_controlnet", 0],
                "negative": ["apply_controlnet", 1],
                "latent_image": ["latent", 0],
                "seed": seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "sampler_name": args.sampler,
                "scheduler": args.scheduler,
                "denoise": args.denoise,
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["checkpoint", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
    if args.denoise < 1:
        workflow["start_image"] = {"class_type": "LoadImage", "inputs": {"image": start_image}}
        workflow["latent"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["start_image", 0], "vae": ["checkpoint", 2]},
        }
    if method == "canny":
        workflow["edges"] = {
            "class_type": "Canny",
            "inputs": {
                "image": ["control_source", 0],
                "low_threshold": args.canny_low,
                "high_threshold": args.canny_high,
            },
        }
        workflow["apply_controlnet"]["inputs"]["image"] = ["edges", 0]
        if save_edges_as:
            workflow["save_edges"] = {
                "class_type": "SaveImage",
                "inputs": {"images": ["edges", 0], "filename_prefix": save_edges_as},
            }
    return workflow


def queue_and_wait(url, workflow, timeout=1200):
    response = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the workflow: {response.text}")
    prompt_id = response.json()["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = requests.get(f"{url}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        status = entry.get("status", {}).get("status_str") if entry else None
        if status == "success":
            return entry["outputs"]
        if status:
            raise RuntimeError(f"render failed: {json.dumps(entry['status'], ensure_ascii=False)[:2000]}")
        time.sleep(0.25)
    raise TimeoutError(f"no result after {timeout} s")


def download_output(url, output, path):
    image = output["images"][0]
    params = {"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]}
    response = requests.get(f"{url}/view", params=params, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def load_frames(frames_dir, only):
    frames = []
    for meta_path in sorted(Path(frames_dir).glob("*/meta.json")):
        frame_id = meta_path.parent.name
        if not only or frame_id in only:
            frames.append((frame_id, meta_path.parent, json.loads(meta_path.read_text(encoding="utf-8"))))
    if not frames:
        raise SystemExit(f"no frames found in {frames_dir}")
    return frames


def write_run_config(args, run_dir, prompts):
    stats = requests.get(f"{args.url}/system_stats", timeout=30).json()["system"]
    settings = {k: v for k, v in vars(args).items() if k not in ("url", "only")}
    config = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CHECKPOINT,
        "controlnet": CONTROLNET,
        "comfyui": stats.get("comfyui_version"),
        "pytorch": stats.get("pytorch_version"),
        "settings": settings,
        "prompts": prompts,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    frames = load_frames(args.frames, args.only)
    write_run_config(args, run_dir, prompts)

    results_path = run_dir / "results.csv"
    is_new = not results_path.exists()
    monitor = HardwareMonitor()
    monitor.start()
    total = len(frames) * len(args.methods) * len(args.seeds)
    done = 0

    with open(results_path, "a", newline="", encoding="utf-8") as results_file:
        results = csv.DictWriter(results_file, fieldnames=CSV_FIELDS)
        if is_new:
            results.writeheader()

        for frame_id, frame_dir, meta in frames:
            out_dir = run_dir / frame_id
            out_dir.mkdir(exist_ok=True)
            sources = {
                "depth": upload_image(args.url, frame_dir / "depth.png", f"{frame_id}_depth.png"),
                "canny": upload_image(args.url, frame_dir / "beauty.png", f"{frame_id}_beauty.png"),
            }
            for method in args.methods:
                for seed in args.seeds:
                    done += 1
                    out_path = out_dir / f"{method}_s{seed}.png"
                    if out_path.exists():
                        continue

                    edges_path = out_dir / "edges.png"
                    prefix = f"benchmark/{run_dir.name}/{frame_id}_{method}_s{seed}"
                    save_edges = f"{prefix}_edges" if method == "canny" and not edges_path.exists() else None
                    workflow = build_workflow(
                        args, method, sources[method], sources["canny"], prompts["frames"][frame_id], prompts["negative"],
                        seed, meta["resolution"], prefix, save_edges,
                    )

                    monitor.reset()
                    started = time.time()
                    outputs = queue_and_wait(args.url, workflow)
                    seconds = time.time() - started

                    download_output(args.url, outputs["save"], out_path)
                    if save_edges:
                        download_output(args.url, outputs["save_edges"], edges_path)

                    results.writerow({
                        "frame": frame_id,
                        "method": method,
                        "seed": seed,
                        "seconds": round(seconds, 1),
                        "peak_vram_mb": monitor.peak_vram_mb,
                        "peak_temp_c": monitor.peak_temp_c,
                        "min_free_ram_gb": round(monitor.min_free_ram_gb, 2),
                        "busy_clock_mhz": monitor.busy_clock_mhz(),
                        "thermal_slowdown_share": monitor.thermal_slowdown_share(),
                        "file": out_path.relative_to(run_dir).as_posix(),
                    })
                    results_file.flush()
                    print(
                        f"[{done}/{total}] {frame_id} {method} seed {seed}: {seconds:.1f} s, "
                        f"VRAM {monitor.peak_vram_mb} MB, {monitor.peak_temp_c} C, "
                        f"clock {monitor.busy_clock_mhz()} MHz, free RAM {monitor.min_free_ram_gb:.1f} GB",
                        flush=True,
                    )

    monitor.stop()
    requests.post(f"{args.url}/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    print("run finished", flush=True)


main()
