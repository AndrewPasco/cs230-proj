"""
Extract valid contact windows and frames.

For each valid run, detects contact windows using force threshold and hysteresis.
Applies frame sampling and nearest-neighbor TCP pose matching.
Outputs valid_frames.json.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_force_magnitude(wrench_data):
    """Compute force magnitude from wrench components."""
    return np.sqrt(
        wrench_data["fx"] ** 2
        + wrench_data["fy"] ** 2
        + wrench_data["fz"] ** 2
    )


def extract_contact_windows(wrench_data, threshold=config.FORCE_THRESHOLD_N):
    """
    Apply hysteresis state machine to detect contact windows.

    Returns list of (start_time, end_time) tuples.
    """
    force_mag = compute_force_magnitude(wrench_data)
    timestamps = wrench_data[config.TIME_COL].values
    force_values = force_mag.values

    windows = []
    in_contact = False
    start_time = None

    for i, (t, f) in enumerate(zip(timestamps, force_values)):
        if not in_contact and f >= threshold:
            in_contact = True
            start_time = t
        elif in_contact and f < threshold:
            in_contact = False
            windows.append((start_time, t))

    # Handle case where last window extends to end
    if in_contact:
        windows.append((start_time, timestamps[-1]))

    return windows


def sample_frames(frame_timestamps, window_start, window_end, mode="every_n", n=1, m=50):
    """
    Sample frames from a window based on config.

    Returns list of indices to keep.
    """
    in_window = (frame_timestamps >= window_start) & (frame_timestamps <= window_end)
    window_indices = np.where(in_window)[0]

    if len(window_indices) == 0:
        return []

    if mode == "every_n":
        return window_indices[::n]
    elif mode == "uniform_m":
        if len(window_indices) <= m:
            return window_indices
        return np.linspace(0, len(window_indices) - 1, m, dtype=int)

    return window_indices


def find_nearest_tcp(frame_time, tcp_times, tcp_data):
    """Find nearest TCP pose for given frame timestamp."""
    idx = np.searchsorted(tcp_times, frame_time)
    idx = np.clip(idx, 0, len(tcp_times) - 1)

    # Check both idx and idx-1 to find closest
    if idx > 0 and abs(tcp_times[idx - 1] - frame_time) < abs(tcp_times[idx] - frame_time):
        idx = idx - 1

    return tcp_data.iloc[idx]


def process_run(run_entry, manifest_dir):
    """Process a single valid run and extract frames."""
    run_path = Path(run_entry["path"])
    run_id = run_entry["run_id"]

    logger.info(f"Processing run: {run_id}")

    # Load wrench data
    wrench_path = run_path / "wrench_data.csv"
    wrench_data = pd.read_csv(wrench_path)

    # Extract windows
    windows = extract_contact_windows(wrench_data, config.FORCE_THRESHOLD_N)
    logger.info(f"  Found {len(windows)} contact window(s)")

    # Load camera frames
    camera_path = run_path / "camera_frames.csv"
    camera_frames = pd.read_csv(camera_path)

    # Load TCP poses
    tcp_path = run_path / "tcp_pose.csv"
    tcp_poses = pd.read_csv(tcp_path)

    frame_timestamps = camera_frames[config.TIME_COL].values
    tcp_timestamps = tcp_poses[config.TIME_COL].values

    # Collect frames from all windows
    selected_frames = []

    for window_start, window_end in windows:
        sampled_indices = sample_frames(
            frame_timestamps,
            window_start,
            window_end,
            mode=config.FRAME_SAMPLING["mode"],
            n=config.FRAME_SAMPLING["n"],
            m=config.FRAME_SAMPLING["m"],
        )

        for idx in sampled_indices:
            frame_row = camera_frames.iloc[idx]
            tcp_row = find_nearest_tcp(frame_row[config.TIME_COL], tcp_timestamps, tcp_poses)
            force_mag = compute_force_magnitude(
                pd.DataFrame(
                    {
                        "fx": [wrench_data.loc[
                            (wrench_data[config.TIME_COL] - frame_row[config.TIME_COL]).abs().idxmin(), "fx"
                        ]],
                        "fy": [wrench_data.loc[
                            (wrench_data[config.TIME_COL] - frame_row[config.TIME_COL]).abs().idxmin(), "fy"
                        ]],
                        "fz": [wrench_data.loc[
                            (wrench_data[config.TIME_COL] - frame_row[config.TIME_COL]).abs().idxmin(), "fz"
                        ]],
                    }
                )
            )[0]

            selected_frames.append({
                "frame_idx": int(frame_row[config.FRAME_IDX_COL]),
                "timestamp": float(frame_row[config.TIME_COL]),
                "image_path": str(frame_row[config.IMAGE_PATH_COL]),
                "tcp_y": float(tcp_row["y"]),
                "force_magnitude": float(force_mag),
            })

    return {
        "run_id": run_id,
        "path": str(run_path),
        "phantom_type": run_entry["phantom_type"],
        "windows": [
            {"start_time": float(s), "end_time": float(e)}
            for s, e in windows
        ],
        "frames": selected_frames,
    }


def main():
    manifest_path = Path(config.DATA_ROOT).parent / "run_manifest.json"

    if not manifest_path.exists():
        logger.error(f"run_manifest.json not found at {manifest_path}")
        logger.info("Run discover_and_index.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_runs = []

    for run_entry in manifest["runs"]:
        if run_entry["valid"]:
            try:
                processed = process_run(run_entry, manifest_path.parent)
                all_runs.append(processed)
            except Exception as e:
                logger.error(f"Error processing {run_entry['run_id']}: {e}")

    output_data = {"runs": all_runs}

    output_path = manifest_path.parent / "valid_frames.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Valid frames saved to: {output_path}")
    logger.info(f"Processed {len(all_runs)} runs")


if __name__ == "__main__":
    main()
