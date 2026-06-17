"""Benchmark ONNX model: latency, RAM, model size, throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent

SAMPLE_TEXT = (
    "GUARDIAN HEALTH AND BEAUTY SDN BHD\n"
    "LOT B-005-006, BASEMENT LEVEL 1 THE STARLING MALL JALAN SS21/60, DAMANSARA UTAMA\n"
    "Invoice No: INV-2024-12345\n"
    "Date: 16/08/17\n"
    "Subtotal: 100.00\n"
    "VAT: 8.21\n"
    "Total: MYR 108.21\n"
    "Thank you for your business."
)


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmark(config: dict, text: str) -> dict:
    onnx_dir = ROOT / config["paths"]["onnx_model_dir"]
    model_path = onnx_dir / config["paths"]["onnx_model_file"]
    tokenizer_path = onnx_dir / "tokenizer"
    bench_cfg = config.get("benchmark", {})

    warmup = bench_cfg.get("warmup_runs", 5)
    timed = bench_cfg.get("timed_runs", 50)
    max_length = config["model"]["max_length"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    rss_before = get_process_rss_mb()

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
    )
    feeds = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }

    rss_after_load = get_process_rss_mb()

    for _ in range(warmup):
        session.run(None, feeds)

    rss_after_warmup = get_process_rss_mb()

    latencies_ms: list[float] = []
    for _ in range(timed):
        t0 = time.perf_counter()
        session.run(None, feeds)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    rss_peak = get_process_rss_mb()
    mean_ms = statistics.mean(latencies_ms)
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0

    return {
        "model_path": str(model_path),
        "model_size_mb": round(model_size_mb, 2),
        "input_words": len(words),
        "max_sequence_length": max_length,
        "warmup_runs": warmup,
        "timed_runs": timed,
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "median": round(statistics.median(latencies_ms), 2),
            "p95": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.95)], 2),
            "min": round(min(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
        },
        "throughput_docs_per_sec": round(throughput, 2),
        "ram_mb": {
            "before_load": round(rss_before, 1),
            "after_load": round(rss_after_load, 1),
            "after_warmup": round(rss_after_warmup, 1),
            "peak": round(rss_peak, 1),
            "inference_overhead": round(rss_peak - rss_before, 1),
        },
        "providers": session.get_providers(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else SAMPLE_TEXT

    results = run_benchmark(config, text)
    print(json.dumps(results, indent=2))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
