"""
Interactive GUI for annotating tendon centerlines.

User clicks on keyframe images to mark the centerline x-pixel position.
Interpolates centerlines for non-keyframe frames.
Outputs centerlines.json.
"""

import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CenterlineAnnotator:
    def __init__(self, valid_frames_path, data_root):
        self.data_root = Path(data_root)

        with open(valid_frames_path) as f:
            self.valid_frames_data = json.load(f)

        self.runs = self.valid_frames_data["runs"]
        self.current_run_idx = 0
        self.current_keyframe_idx = 0
        self.keyframe_annotations = {}

        logger.info(f"Loaded {len(self.runs)} runs")

    def get_keyframe_indices(self, run):
        """Get keyframe indices for a run (every KEYFRAME_INTERVAL frames)."""
        if not run["frames"]:
            return []

        frame_indices = [f["frame_idx"] for f in run["frames"]]
        frame_indices.sort()

        keyframe_indices = list(range(0, len(frame_indices), config.KEYFRAME_INTERVAL))
        if len(frame_indices) - 1 not in keyframe_indices:
            keyframe_indices.append(len(frame_indices) - 1)

        return keyframe_indices

    def find_frame(self, run, frame_idx):
        """Find a frame by index in a run."""
        for frame in run["frames"]:
            if frame["frame_idx"] == frame_idx:
                return frame
        return None

    def load_image(self, run, frame):
        """Load image from disk."""
        run_path = Path(run["path"]) if "path" in run else self.data_root / run["run_id"]
        image_path = run_path / frame["image_path"]

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        img = cv2.imread(str(image_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def show_keyframe(self, run_idx, keyframe_idx):
        """Display a keyframe and get user input."""
        run = self.runs[run_idx]
        run_id = run["run_id"]
        keyframe_indices = self.get_keyframe_indices(run)

        if keyframe_idx >= len(keyframe_indices):
            logger.warning(f"Keyframe index out of range: {keyframe_idx}")
            return None

        frame_list_idx = keyframe_indices[keyframe_idx]
        frame = run["frames"][frame_list_idx] if frame_list_idx < len(run["frames"]) else None

        if frame is None:
            logger.warning(f"Frame not found at index {frame_list_idx}")
            return None

        img = self.load_image(run, frame)
        if img is None:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(img)
        ax.set_title(
            f"Run {run_id} | Frame {frame['frame_idx']} | "
            f"Keyframe {keyframe_idx+1}/{len(keyframe_indices)} | "
            f"Click on centerline (ESC to skip)"
        )

        # Capture click
        centerline_px = None

        def on_click(event):
            nonlocal centerline_px
            if event.xdata is not None:
                centerline_px = int(event.xdata)
                logger.info(f"Centerline set to {centerline_px} px")
                plt.close()

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()

        return centerline_px

    def annotate_interactive(self):
        """Interactive annotation loop."""
        while self.current_run_idx < len(self.runs):
            run = self.runs[self.current_run_idx]
            run_id = run["run_id"]
            keyframe_indices = self.get_keyframe_indices(run)

            if run_id not in self.keyframe_annotations:
                self.keyframe_annotations[run_id] = {}

            while self.current_keyframe_idx < len(keyframe_indices):
                frame_list_idx = keyframe_indices[self.current_keyframe_idx]
                frame = run["frames"][frame_list_idx]
                frame_idx = frame["frame_idx"]

                if frame_idx not in self.keyframe_annotations[run_id]:
                    centerline_px = self.show_keyframe(self.current_run_idx, self.current_keyframe_idx)
                    if centerline_px is not None:
                        self.keyframe_annotations[run_id][frame_idx] = centerline_px

                self.current_keyframe_idx += 1

            self.current_run_idx += 1
            self.current_keyframe_idx = 0

    def interpolate_centerlines(self):
        """Interpolate centerlines for all frames."""
        interpolated_data = {}

        for run in self.runs:
            run_id = run["run_id"]
            keyframes = self.keyframe_annotations.get(run_id, {})

            if not keyframes:
                logger.warning(f"No keyframes annotated for {run_id}")
                continue

            keyframe_indices = sorted(keyframes.keys())
            keyframe_values = [keyframes[idx] for idx in keyframe_indices]

            all_frame_indices = [f["frame_idx"] for f in run["frames"]]
            interpolated = {}

            for frame_idx in all_frame_indices:
                if frame_idx in keyframes:
                    interpolated[frame_idx] = keyframes[frame_idx]
                else:
                    # Linear interpolation
                    if frame_idx < keyframe_indices[0]:
                        interpolated[frame_idx] = keyframe_values[0]
                    elif frame_idx > keyframe_indices[-1]:
                        interpolated[frame_idx] = keyframe_values[-1]
                    else:
                        # Find surrounding keyframes
                        for i, kf_idx in enumerate(keyframe_indices):
                            if kf_idx > frame_idx:
                                kf1_idx = keyframe_indices[i - 1]
                                kf2_idx = kf_idx
                                kf1_val = keyframe_values[i - 1]
                                kf2_val = keyframe_values[i]

                                # Linear interpolation
                                t = (frame_idx - kf1_idx) / (kf2_idx - kf1_idx)
                                interpolated[frame_idx] = kf1_val + t * (kf2_val - kf1_val)
                                break

            interpolated_data[run_id] = {
                "keyframes": [
                    {"frame_idx": idx, "centerline_px": keyframes[idx]}
                    for idx in keyframe_indices
                ],
                "interpolated": interpolated,
            }

        return interpolated_data

    def save_centerlines(self, output_path):
        """Save centerlines to JSON."""
        self.interpolate_centerlines()
        data = {
            "runs": {
                run_id: {
                    "keyframes": self.keyframe_annotations.get(run_id, {}),
                    "interpolated": {},
                }
                for run_id in [r["run_id"] for r in self.runs]
            }
        }

        # Rebuild with proper structure
        final_data = {}
        for run in self.runs:
            run_id = run["run_id"]
            keyframes = self.keyframe_annotations.get(run_id, {})

            if not keyframes:
                continue

            keyframe_list = [
                {"frame_idx": idx, "centerline_px": centerline_px}
                for idx, centerline_px in keyframes.items()
            ]

            final_data[run_id] = {
                "keyframes": keyframe_list,
                "interpolated": self.interpolate_centerlines().get(run_id, {}).get("interpolated", {}),
            }

        output = {"runs": final_data}

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Centerlines saved to {output_path}")


def main():
    valid_frames_path = Path(config.DATA_ROOT).parent / "valid_frames.json"

    if not valid_frames_path.exists():
        logger.error(f"valid_frames.json not found at {valid_frames_path}")
        logger.info("Run extract_valid_windows.py first.")
        return

    annotator = CenterlineAnnotator(valid_frames_path, config.DATA_ROOT)

    logger.info("Starting interactive centerline annotation.")
    logger.info("For each keyframe, click on the tendon centerline in the image.")
    logger.info("This is MANUAL - you must identify where the actual tendon center is!")
    annotator.annotate_interactive()

    output_path = valid_frames_path.parent / "centerlines.json"
    annotator.save_centerlines(output_path)


if __name__ == "__main__":
    main()
