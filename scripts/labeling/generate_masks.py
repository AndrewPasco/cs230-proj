"""
Generate masks and create dataset manifest.

For each frame in valid_frames.json, generates a segmentation mask
using the TendonLabeler class and saves organized output.
"""

import csv
import json
import logging
from pathlib import Path
import shutil

import cv2

import config
from tendon_labeler import TendonLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_masks_for_run(run, centerlines_data, output_root, stl_dir):
    """Generate masks for all frames in a run."""
    run_id = run["run_id"]
    phantom_type = run["phantom_type"]

    logger.info(f"Processing run: {run_id}")

    # Get STL file
    stl_filename = config.PHANTOM_STL_MAP.get(phantom_type)
    if not stl_filename:
        logger.error(f"Unknown phantom type: {phantom_type}")
        return []

    stl_path = Path(stl_dir) / stl_filename
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        return []

    # Initialize TendonLabeler
    try:
        labeler = TendonLabeler(str(stl_path))
    except Exception as e:
        logger.error(f"Failed to load TendonLabeler: {e}")
        return []

    # Get centerlines for this run
    run_centerlines = centerlines_data.get(run_id, {})
    interpolated_centerlines = run_centerlines.get("interpolated", {})

    if not interpolated_centerlines:
        logger.warning(f"No centerlines found for {run_id}")
        return []

    # Create output directories
    output_dir = Path(output_root) / phantom_type
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    source_path = Path(run.get("path", config.DATA_ROOT) or config.DATA_ROOT)

    for frame in run["frames"]:
        frame_idx = frame["frame_idx"]
        timestamp = frame["timestamp"]
        tcp_y = frame["tcp_y"]
        force_magnitude = frame["force_magnitude"]

        # Get centerline (convert frame_idx to string for JSON key lookup)
        centerline_px = interpolated_centerlines.get(str(frame_idx))

        if centerline_px is None:
            logger.warning(f"No centerline for frame {frame_idx}, skipping")
            continue

        # Load source image
        image_path = source_path / frame["image_path"]
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue

        # Generate mask
        height, width = img.shape[:2]
        labeler.set_center(int(centerline_px))

        try:
            mask = labeler.generate_mask(width, height, cam_y=tcp_y)
        except Exception as e:
            logger.error(f"Failed to generate mask for frame {frame_idx}: {e}")
            continue

        # Save image and mask
        img_filename = f"{run_id}_frame_{frame_idx:05d}.jpg"
        mask_filename = f"{run_id}_frame_{frame_idx:05d}.png"

        img_output_path = images_dir / img_filename
        mask_output_path = masks_dir / mask_filename

        # Copy image (convert BGR to RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(img_output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Save mask
        cv2.imwrite(str(mask_output_path), mask)

        # Record manifest row
        manifest_rows.append({
            "phantom_type": phantom_type,
            "run_id": run_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "tcp_y": tcp_y,
            "force_magnitude": force_magnitude,
            "centerline_px": int(centerline_px),
            "image_path": f"{phantom_type}/images/{img_filename}",
            "mask_path": f"{phantom_type}/masks/{mask_filename}",
        })

        logger.info(f"  Generated mask for frame {frame_idx}")

    return manifest_rows


def main():
    valid_frames_path = Path(config.DATA_ROOT).parent / "valid_frames.json"
    centerlines_path = Path(config.DATA_ROOT).parent / "centerlines.json"

    if not valid_frames_path.exists():
        logger.error(f"valid_frames.json not found at {valid_frames_path}")
        logger.info("Run extract_valid_windows.py first.")
        return

    if not centerlines_path.exists():
        logger.error(f"centerlines.json not found at {centerlines_path}")
        logger.info("Run annotate_centerlines.py first.")
        return

    # Load data
    with open(valid_frames_path) as f:
        valid_frames_data = json.load(f)

    with open(centerlines_path) as f:
        centerlines_data = json.load(f).get("runs", {})

    # Generate masks
    all_manifest_rows = []

    for run in valid_frames_data["runs"]:
        rows = generate_masks_for_run(
            run,
            centerlines_data,
            config.OUTPUT_ROOT,
            config.STL_DIR,
        )
        all_manifest_rows.extend(rows)

    # Save manifest CSV
    output_path = Path(config.OUTPUT_ROOT)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_csv_path = output_path / "dataset_manifest.csv"

    if all_manifest_rows:
        fieldnames = [
            "phantom_type", "run_id", "frame_idx", "timestamp",
            "tcp_y", "force_magnitude", "centerline_px",
            "image_path", "mask_path"
        ]

        with open(manifest_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_manifest_rows)

        logger.info(f"Dataset manifest saved to {manifest_csv_path}")
        logger.info(f"Total frames processed: {len(all_manifest_rows)}")
    else:
        logger.warning("No frames were processed")


if __name__ == "__main__":
    main()
