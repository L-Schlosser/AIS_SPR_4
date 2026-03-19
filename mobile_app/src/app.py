"""CLI demo application for the edge-AI document processing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ANSI colour helpers (no-op when stdout is not a terminal)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_GREEN = "\033[92m" if _USE_COLOR else ""
_YELLOW = "\033[93m" if _USE_COLOR else ""
_RED = "\033[91m" if _USE_COLOR else ""
_CYAN = "\033[96m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def _print_header(text: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {text}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'=' * 60}{_RESET}\n")


def _print_result(result: dict, elapsed_ms: float) -> None:
    """Pretty-print a ProcessingResult dict."""
    doc_type = result.get("document_type", "unknown")
    confidence = result.get("confidence", 0.0)

    if confidence >= 0.9:
        conf_color = _GREEN
    elif confidence >= 0.7:
        conf_color = _YELLOW
    else:
        conf_color = _RED

    print(f"  {_BOLD}Document type:{_RESET}  {doc_type}")
    print(f"  {_BOLD}Confidence:{_RESET}     {conf_color}{confidence:.2%}{_RESET}")
    print(f"  {_BOLD}Processing:{_RESET}     {elapsed_ms:.0f} ms")
    print()

    fields = result.get("fields", {})
    if fields:
        print(f"  {_BOLD}Extracted fields:{_RESET}")
        for key, value in fields.items():
            print(f"    {key}: {value}")
        print()

    raw_text = result.get("raw_text")
    if raw_text:
        preview = raw_text[:200] + ("..." if len(raw_text) > 200 else "")
        print(f"  {_BOLD}Raw text (preview):{_RESET}")
        print(f"    {preview}")
        print()


# ---- Commands ---------------------------------------------------------------


def cmd_process(args: argparse.Namespace) -> None:
    """Process a single document image."""
    from api.service import DocumentService

    _print_header("Processing Document")
    print(f"  Image: {args.image_path}")

    service = DocumentService(config_path=args.config)
    start = time.perf_counter()
    result = service.process_image_file(args.image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result_dict = result.model_dump() if hasattr(result, "model_dump") else vars(result)
    _print_result(result_dict, elapsed_ms)

    print(f"  {_BOLD}Full JSON:{_RESET}")
    print(json.dumps(result_dict, indent=2, default=str))


def cmd_info(args: argparse.Namespace) -> None:
    """Show model info and supported document types."""
    from mobile_app.src.model_manager import ModelManager

    _print_header("Model Information")

    manager = ModelManager(args.models_dir)
    info = manager.get_model_info()

    print(f"  {_BOLD}Models directory:{_RESET} {args.models_dir}")
    print()

    all_present, missing = manager.check_models_exist()
    for name, mi in info.items():
        status = f"{_GREEN}OK{_RESET}" if mi.exists else f"{_RED}MISSING{_RESET}"
        size = f"{mi.size_mb:.2f} MB" if mi.exists else "—"
        print(f"  [{status}] {name:30s} {size}")

    print()
    total = manager.get_total_size_mb()
    print(f"  {_BOLD}Total model size:{_RESET} {total:.2f} MB")
    print()

    if all_present:
        print(f"  {_GREEN}All required models are present.{_RESET}")
    else:
        print(f"  {_RED}Missing models:{_RESET} {', '.join(missing)}")

    print()
    print(f"  {_BOLD}Supported document types:{_RESET}")
    for dt in ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]:
        print(f"    - {dt}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Process all images in a directory."""
    from api.service import DocumentService

    _print_header("Batch Processing")

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"  {_RED}Error: {args.directory} is not a directory.{_RESET}")
        sys.exit(1)

    extensions = {".png", ".jpg", ".jpeg"}
    image_files = sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)

    if not image_files:
        print(f"  {_YELLOW}No image files found in {args.directory}.{_RESET}")
        return

    print(f"  Found {len(image_files)} image(s) in {args.directory}")
    print()

    service = DocumentService(config_path=args.config)

    results = []
    for img_path in image_files:
        print(f"  {_BOLD}Processing:{_RESET} {img_path.name}")
        start = time.perf_counter()
        result = service.process_image_file(str(img_path))
        elapsed_ms = (time.perf_counter() - start) * 1000
        result_dict = result.model_dump() if hasattr(result, "model_dump") else vars(result)
        result_dict["_file"] = str(img_path)
        result_dict["_processing_ms"] = round(elapsed_ms, 1)
        results.append(result_dict)
        _print_result(result_dict, elapsed_ms)

    print(f"  {_BOLD}Processed {len(results)} image(s).{_RESET}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  Results saved to {args.output}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Generate a sample image, process it, and show the result."""
    import tempfile

    _print_header("Demo Mode")

    print("  Generating sample document images...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate one image per document type
        try:
            from scripts.generate_samples import generate_arztbesuch, generate_lieferschein, generate_reisekosten

            for gen_fn, dtype in [
                (generate_arztbesuch, "arztbesuchsbestaetigung"),
                (generate_reisekosten, "reisekostenbeleg"),
                (generate_lieferschein, "lieferschein"),
            ]:
                gen_fn(Path(tmpdir), count=1)
                print(f"  {_GREEN}Generated:{_RESET} {dtype}")
        except ImportError:
            print(f"  {_RED}Error: generate_samples requires the 'train' dependency group.{_RESET}")
            print("  Run: uv sync --group train")
            sys.exit(1)

        # Try to process each with the pipeline (requires trained models)
        try:
            from api.service import DocumentService

            service = DocumentService(config_path=args.config)
            for dtype in ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]:
                out_dir = os.path.join(tmpdir, dtype)
                images = [f for f in os.listdir(out_dir) if f.endswith(".png")]
                if images:
                    img_path = os.path.join(out_dir, images[0])
                    print(f"\n  {_BOLD}Processing {dtype}:{_RESET}")
                    start = time.perf_counter()
                    result = service.process_image_file(img_path)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    result_dict = result.model_dump() if hasattr(result, "model_dump") else vars(result)
                    _print_result(result_dict, elapsed_ms)
        except Exception as exc:
            print(f"\n  {_YELLOW}Pipeline not available (models may not be trained yet): {exc}{_RESET}")
            print("  Demo generated sample images successfully. Train models to enable full processing.")

    print(f"\n  {_GREEN}Demo complete.{_RESET}")


# ---- CLI entry point --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI app."""
    parser = argparse.ArgumentParser(
        prog="mobile_app",
        description="Edge-AI Document Processing — CLI Demo",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline config YAML (default: config.yaml)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # process
    p_process = subparsers.add_parser("process", help="Process a single document image")
    p_process.add_argument("image_path", help="Path to the document image (PNG/JPEG)")

    # info
    p_info = subparsers.add_parser("info", help="Show model info and supported types")
    p_info.add_argument("--models-dir", default="edge_model", help="Base directory for models (default: edge_model)")

    # batch
    p_batch = subparsers.add_parser("batch", help="Process all images in a directory")
    p_batch.add_argument("directory", help="Directory containing document images")
    p_batch.add_argument("--output", "-o", help="Save results to a JSON file")

    # demo
    subparsers.add_parser("demo", help="Generate sample images and process them")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "process": cmd_process,
        "info": cmd_info,
        "batch": cmd_batch,
        "demo": cmd_demo,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
