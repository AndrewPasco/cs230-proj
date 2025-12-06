"""
Generate masks and create dataset manifest.

For each frame in valid_frames.json, generates a segmentation mask
using the TendonLabeler class and saves organized output.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import shutil

import cv2
import numpy as np

import config
from tendon_labeler import TendonLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_ellipse_mask(mask, ellipse_params):
    """Apply ellipse ROI mask to set pixels outside ellipse to 0."""
    if ellipse_params is None:
        return mask

    height, width = mask.shape

    # Create ellipse mask
    center_x = int(ellipse_params["center_x"])
    center_y = int(ellipse_params["center_y"])
    semi_major = int(ellipse_params["semi_major"])
    semi_minor = int(ellipse_params["semi_minor"])
    rotation_angle = ellipse_params["rotation_angle"]

    # Create empty mask for ellipse
    ellipse_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw filled ellipse
    cv2.ellipse(
        ellipse_mask,
        center=(center_x, center_y),
        axes=(semi_major, semi_minor),
        angle=rotation_angle,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,  # filled
    )

    # Apply ellipse mask: keep only pixels inside ellipse
    masked = mask & ellipse_mask

    return masked


def generate_masks_for_run(run, centerlines_data, ellipses_data, output_root, stl_dir):
    """Generate masks for all frames in a run."""
    run_id = run["run_id"]
    phantom_type = run["phantom_type"]

    logger.info(f"Processing run: {run_id}")

    # Get STL file and rotation from run manifest
    stl_filename = run.get("stl_file")
    rotation_deg = run.get("rotation_deg", 0)

    if not stl_filename:
        logger.error(f"No STL file specified for run: {run_id}")
        return []

    stl_path = Path(stl_dir) / stl_filename
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        return []

    # Initialize TendonLabeler with rotation
    try:
        labeler = TendonLabeler(str(stl_path), rotation_deg=rotation_deg)
    except Exception as e:
        logger.error(f"Failed to load TendonLabeler: {e}")
        return []

    # Get centerlines for this run
    run_centerlines = centerlines_data.get(run_id, {})
    interpolated_centerlines = run_centerlines.get("interpolated", {})

    if not interpolated_centerlines:
        logger.warning(f"No centerlines found for {run_id}")
        return []

    # Get ellipses for this run (optional)
    run_ellipses = ellipses_data.get(run_id, {})
    interpolated_ellipses = run_ellipses.get("interpolated", {})

    # Create output directories
    output_dir = Path(output_root) / phantom_type
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    depths_dir = output_dir / "depths"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)

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

        # Generate mask and depth map
        height, width = img.shape[:2]
        labeler.set_center(int(centerline_px))

        try:
            mask, depth_map = labeler.generate_mask(width, height, cam_y=tcp_y)
        except Exception as e:
            logger.error(f"Failed to generate mask for frame {frame_idx}: {e}")
            continue


        # Save image, mask, and depth
        img_filename = f"{run_id}_frame_{frame_idx:05d}.jpg"
        mask_filename = f"{run_id}_frame_{frame_idx:05d}.png"
        depth_filename = f"{run_id}_frame_{frame_idx:05d}.npy"

        img_output_path = images_dir / img_filename
        mask_output_path = masks_dir / mask_filename
        depth_output_path = depths_dir / depth_filename

        # Copy image (convert BGR to RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(img_output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Save mask
        cv2.imwrite(str(mask_output_path), mask)

        # Save depth map (float32, meters)
        np.save(str(depth_output_path), depth_map.astype(np.float32))

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
            "depth_path": f"{phantom_type}/depths/{depth_filename}",
        })

        logger.info(f"  Generated mask for frame {frame_idx}")

    return manifest_rows


def main_single_run(target_run_id):
    """Generate masks for a specific run_id only."""
    valid_frames_path = Path(config.CONFIGS_ROOT) / "valid_frames.json"
    centerlines_path = Path(config.CONFIGS_ROOT) / "centerlines.json"
    ellipses_path = Path(config.CONFIGS_ROOT) / "ellipses.json"
    run_manifest_path = Path(config.CONFIGS_ROOT) / "run_manifest.json"

    if not valid_frames_path.exists():
        logger.error(f"valid_frames.json not found at {valid_frames_path}")
        logger.info("Run extract_valid_windows.py first.")
        return

    if not centerlines_path.exists():
        logger.error(f"centerlines.json not found at {centerlines_path}")
        logger.info("Run annotate_centerlines.py first.")
        return

    if not run_manifest_path.exists():
        logger.error(f"run_manifest.json not found at {run_manifest_path}")
        logger.info("Run discover_and_index.py first.")
        return

    # Load data
    with open(valid_frames_path) as f:
        valid_frames_data = json.load(f)

    with open(centerlines_path) as f:
        centerlines_data = json.load(f).get("runs", {})

    # Load ellipses (optional)
    ellipses_data = {}
    if ellipses_path.exists():
        with open(ellipses_path) as f:
            ellipses_data = json.load(f).get("runs", {})
        logger.info("Loaded ellipses data")
    else:
        logger.info("No ellipses.json found, proceeding without ROI masking")

    with open(run_manifest_path) as f:
        run_manifest_data = json.load(f)

    # Create lookup for run metadata (stl_file, rotation_deg)
    run_metadata = {}
    for run_info in run_manifest_data.get("runs", []):
        run_id = run_info["run_id"]
        run_metadata[run_id] = {
            "stl_file": run_info.get("stl_file"),
            "rotation_deg": run_info.get("rotation_deg", 0),
        }

    # Generate masks for the specific run
    all_manifest_rows = []

    for run in valid_frames_data["runs"]:
        # Only process the target run
        run_id = run["run_id"]
        if run_id != target_run_id:
            continue

        # Merge run data with STL config from manifest
        if run_id in run_metadata:
            run["stl_file"] = run_metadata[run_id]["stl_file"]
            run["rotation_deg"] = run_metadata[run_id]["rotation_deg"]

        rows = generate_masks_for_run(
            run,
            centerlines_data,
            ellipses_data,
            config.OUTPUT_ROOT,
            config.STL_DIR,
        )
        all_manifest_rows.extend(rows)

    if not all_manifest_rows:
        logger.warning(f"No frames were processed for run: {target_run_id}")
        return

    # Append to existing CSV or create new one
    output_path = Path(config.OUTPUT_ROOT)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = output_path / "dataset_manifest.csv"

    fieldnames = [
        "phantom_type", "run_id", "frame_idx", "timestamp",
        "tcp_y", "force_magnitude", "centerline_px",
        "image_path", "mask_path", "depth_path"
    ]

    # Check if CSV exists to append without header
    csv_exists = manifest_csv_path.exists()
    with open(manifest_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerows(all_manifest_rows)

    logger.info(f"Dataset manifest updated: {manifest_csv_path}")
    logger.info(f"Processed {len(all_manifest_rows)} frames for run: {target_run_id}")


def main():
    valid_frames_path = Path(config.CONFIGS_ROOT) / "valid_frames.json"
    centerlines_path = Path(config.CONFIGS_ROOT) / "centerlines.json"
    ellipses_path = Path(config.CONFIGS_ROOT) / "ellipses.json"
    run_manifest_path = Path(config.CONFIGS_ROOT) / "run_manifest.json"

    if not valid_frames_path.exists():
        logger.error(f"valid_frames.json not found at {valid_frames_path}")
        logger.info("Run extract_valid_windows.py first.")
        return

    if not centerlines_path.exists():
        logger.error(f"centerlines.json not found at {centerlines_path}")
        logger.info("Run annotate_centerlines.py first.")
        return

    if not run_manifest_path.exists():
        logger.error(f"run_manifest.json not found at {run_manifest_path}")
        logger.info("Run discover_and_index.py first.")
        return

    # Load data
    with open(valid_frames_path) as f:
        valid_frames_data = json.load(f)

    with open(centerlines_path) as f:
        centerlines_data = json.load(f).get("runs", {})

    # Load ellipses (optional)
    ellipses_data = {}
    if ellipses_path.exists():
        with open(ellipses_path) as f:
            ellipses_data = json.load(f).get("runs", {})
        logger.info("Loaded ellipses data")
    else:
        logger.info("No ellipses.json found, proceeding without ROI masking")

    with open(run_manifest_path) as f:
        run_manifest_data = json.load(f)

    # Create lookup for run metadata (stl_file, rotation_deg)
    run_metadata = {}
    for run_info in run_manifest_data.get("runs", []):
        run_id = run_info["run_id"]
        run_metadata[run_id] = {
            "stl_file": run_info.get("stl_file"),
            "rotation_deg": run_info.get("rotation_deg", 0),
        }

    # Generate masks
    all_manifest_rows = []

    for run in valid_frames_data["runs"]:
        # Merge run data with STL config from manifest
        run_id = run["run_id"]
        if run_id in run_metadata:
            run["stl_file"] = run_metadata[run_id]["stl_file"]
            run["rotation_deg"] = run_metadata[run_id]["rotation_deg"]
        rows = generate_masks_for_run(
            run,
            centerlines_data,
            ellipses_data,
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
            "image_path", "mask_path", "depth_path"
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
    parser = argparse.ArgumentParser(
        description="Generate masks for runs in valid_frames.json"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Only process a specific run_id. If not provided, processes all runs."
    )
    args = parser.parse_args()

    if args.run_id:
        logger.info(f"Processing only run: {args.run_id}")
        main_single_run(args.run_id)
    else:
        logger.info("Processing all runs")
        main()
