#!/usr/bin/env python3
"""Build the first formal bridge benchmark export bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge_benchmark import FIXED_DATA_SOURCES, build_export_bundle
from src.logging_config import setup_logger
from src.open_model_probe.io_utils import prepare_output_dir


def main() -> int:
    setup_logger(name="bridge_benchmark_protocol", level="INFO")
    source_paths = [
        project_root / "data" / "external" / "sycophancy_database" / name
        for name in FIXED_DATA_SOURCES
    ]
    output_dir = prepare_output_dir(
        project_root / "outputs" / "experiments" / "bridge_benchmark_protocol",
        run_name="formal_bridge_protocol",
    )
    manifest = build_export_bundle(source_paths=source_paths, output_dir=output_dir)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
