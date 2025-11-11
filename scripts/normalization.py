import os
from argparse import ArgumentParser
import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--video",
    default="data-2025-11-07.mp4",
    help="Path to the video file to analyze",
)
parser.add_argument(
    "--downsample_factor",
    type=int,
    default=1,
    help="Factor to downsample frames for faster processing",
)
args = parser.parse_args()

dir = os.path.join(os.path.dirname(__file__), "../data/")
path = os.path.join(dir, args.video)

cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {path}")

means = []
stds = []

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count//args.downsample_factor} frames...")
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % args.downsample_factor != 0:
        idx += 1
        continue
    print(f"Processing frame {idx+1}/{frame_count}", end="\r")
    # Convert BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0  # normalize to [0,1]

    # Compute mean and std per channel
    means.append(frame.mean(axis=(0, 1)))
    stds.append(frame.std(axis=(0, 1)))

    idx += 1

cap.release()

# Average across all frames
mean_rgb = np.mean(means, axis=0)
std_rgb = np.mean(stds, axis=0)

print("Mean RGB:", mean_rgb)
print("Std RGB:", std_rgb)
