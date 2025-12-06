import os
import cv2
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

# relative paths to the image directories
RELATIVE_PATHS = [
    "../data/p1/images",
    "../data/p2/images",
    "../data/p3/images",
]

parser = ArgumentParser()
parser.add_argument(
    "--downsample_factor",
    type=int,
    default=1,
    help="Factor to downsample frames for faster processing (e.g., process every Nth image).",
)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

all_image_paths = []

for relative_path in RELATIVE_PATHS:
    full_path = os.path.join(script_dir, relative_path)
    all_image_paths.extend(glob(os.path.join(full_path, "*.jpg")))
    all_image_paths.extend(glob(os.path.join(full_path, "*.png")))

if not all_image_paths:
    raise RuntimeError(f"No images found in the specified paths: {RELATIVE_PATHS}")

print(f"Found {len(all_image_paths)} images.")

means = []
stds = []
image_counter = 0

for path in tqdm(all_image_paths, desc="Processing Images"):

    # Apply downsample factor
    if image_counter % args.downsample_factor != 0:
        image_counter += 1
        continue

    frame = cv2.imread(path)

    if frame is None:
        print(f"Warning: Could not read image at {path}. Skipping.")
        image_counter += 1
        continue

    # Convert BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and convert to float32
    frame = frame.astype(np.float32) / 255.0

    # Compute mean and std per channel (R, G, B)
    means.append(frame.mean(axis=(0, 1)))
    stds.append(frame.std(axis=(0, 1)))

    image_counter += 1

mean_rgb = np.mean(means, axis=0)
std_rgb = np.mean(stds, axis=0)

print("\n--- Results ---")
print(f"Processed {len(means)} images (after downsampling).")
print("Mean RGB:", mean_rgb)
print("Std RGB:", std_rgb)
