"""
Interactive mask viewer using OpenCV.

Navigate through frames with arrow keys or slider.
Keyboard controls:
  - LEFT/RIGHT arrow: Previous/Next frame
  - SPACE: Toggle mask overlay on/off
  - ESC: Quit
"""

import csv
from pathlib import Path

import cv2
import numpy as np

import config


class MaskViewer:
    def __init__(self, output_root=None):
        self.output_root = Path(output_root or config.OUTPUT_ROOT)
        self.manifest_path = self.output_root / "dataset_manifest.csv"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Load manifest
        self.rows = []
        with open(self.manifest_path) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError("No rows in manifest")

        self.current_idx = 0
        self.show_overlay = True
        self.window_name = "Mask Viewer - Arrow keys: navigate | SPACE: toggle overlay | ESC: quit"

    def get_blended_image(self, idx):
        """Load and blend image with mask."""
        row = self.rows[idx]
        img_path = self.output_root / row["image_path"]
        mask_path = self.output_root / row["mask_path"]

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return None, None, row

        # Create overlay
        if self.show_overlay:
            overlay = img.copy().astype(float)
            # Red overlay for mask (BGR format)
            overlay[mask > 0] = [0, 0, 255]
            blended = (0.7 * img + 0.3 * overlay).astype("uint8")
        else:
            blended = img

        return blended, mask, row

    def create_info_text(self, row):
        """Create info text for display."""
        info = (
            f"Frame {self.current_idx + 1}/{len(self.rows)} | "
            f"Frame ID: {row['frame_idx']} | "
            f"Run: {row['run_id']}\n"
            f"Centerline: {row['centerline_px']} px | "
            f"TCP Y: {row['tcp_y']} | "
            f"Force: {row['force_magnitude']} N\n"
            f"Overlay: {'ON' if self.show_overlay else 'OFF'}"
        )
        return info

    def display_frame(self):
        """Display current frame with info."""
        blended, mask, row = self.get_blended_image(self.current_idx)

        if blended is None:
            print(f"Failed to load frame {self.current_idx}")
            return

        # Create display with info
        display = blended.copy()
        info = self.create_info_text(row)

        # Add text to image
        y_offset = 30
        for line in info.split("\n"):
            cv2.putText(
                display,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        cv2.imshow(self.window_name, display)

    def run(self):
        """Main viewer loop."""
        self.display_frame()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 82:  # UP arrow
                self.current_idx = max(0, self.current_idx - 10)
                self.display_frame()
            elif key == 84:  # DOWN arrow
                self.current_idx = min(len(self.rows) - 1, self.current_idx + 10)
                self.display_frame()
            elif key == 81:  # LEFT arrow
                self.current_idx = max(0, self.current_idx - 1)
                self.display_frame()
            elif key == 83:  # RIGHT arrow
                self.current_idx = min(len(self.rows) - 1, self.current_idx + 1)
                self.display_frame()
            elif key == 32:  # SPACE
                self.show_overlay = not self.show_overlay
                self.display_frame()
            elif key == ord("g"):  # Go to frame
                try:
                    frame_num = int(input("Enter frame number (0-indexed): "))
                    if 0 <= frame_num < len(self.rows):
                        self.current_idx = frame_num
                        self.display_frame()
                    else:
                        print(f"Invalid frame number. Range: 0-{len(self.rows) - 1}")
                except ValueError:
                    print("Invalid input")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = MaskViewer()
    print(f"Loaded {len(viewer.rows)} frames")
    print("Controls:")
    print("  LEFT/RIGHT arrow: Previous/Next frame")
    print("  UP/DOWN arrow: Jump -/+ 10 frames")
    print("  SPACE: Toggle mask overlay")
    print("  G: Go to specific frame number")
    print("  ESC: Quit")
    viewer.run()
