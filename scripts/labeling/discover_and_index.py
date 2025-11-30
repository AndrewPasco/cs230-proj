"""
Discover and index data runs.

Recursively scans DATA_ROOT for folders matching pattern {phantom}_{motion}_{force}.
Validates each folder and outputs run_manifest.json.
"""

import json
import os
import re
from pathlib import Path
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected files in each run folder
REQUIRED_FILES = ["frames", "camera_frames.csv", "tcp_pose.csv", "wrench_data.csv"]
RUN_PATTERN = re.compile(r"^(p\d+)_([a-z0-9]+)_(\d+N)(?:-.+)?$")


def discover_runs(data_root):
    """
    Recursively scan data_root for run folders.

    Returns list of dicts with: run_id, path, phantom_type, motion_type, force_label, valid, missing_files
    """
    runs = []
    data_root = Path(data_root)

    if not data_root.exists():
        logger.warning(f"DATA_ROOT does not exist: {data_root}")
        return runs

    for item in data_root.iterdir():
        if not item.is_dir():
            continue

        folder_name = item.name
        match = RUN_PATTERN.match(folder_name)

        if not match:
            logger.debug(f"Skipping non-matching folder: {folder_name}")
            continue

        phantom_type, motion_type, force_label = match.groups()

        # Validate required files
        missing_files = []
        for required in REQUIRED_FILES:
            full_path = item / required
            if not full_path.exists():
                missing_files.append(required)

        valid = len(missing_files) == 0

        run_entry = {
            "run_id": folder_name,
            "path": str(item.absolute()),
            "phantom_type": phantom_type,
            "motion_type": motion_type,
            "force_label": force_label,
            "valid": valid,
            "missing_files": missing_files,
        }

        runs.append(run_entry)

        if valid:
            logger.info(f"✓ Valid run: {folder_name}")
        else:
            logger.warning(f"✗ Invalid run: {folder_name} (missing: {missing_files})")

    return runs


def main():
    logger.info(f"Scanning DATA_ROOT: {config.DATA_ROOT}")

    runs = discover_runs(config.DATA_ROOT)

    valid_count = sum(1 for r in runs if r["valid"])
    invalid_count = len(runs) - valid_count

    manifest = {
        "runs": runs,
        "summary": {
            "total": len(runs),
            "valid": valid_count,
            "invalid": invalid_count,
        }
    }

    output_path = Path(config.DATA_ROOT).parent / "run_manifest.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to: {output_path}")
    logger.info(f"Summary: {valid_count} valid, {invalid_count} invalid out of {len(runs)} total")


if __name__ == "__main__":
    main()
