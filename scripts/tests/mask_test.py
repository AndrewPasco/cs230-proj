#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--frame_num",
    default=580,
    help="Frame number to be displayed",
)
args = parser.parse_args()

masks_dir = os.path.join(os.path.dirname(__file__), "../../data/masks/")
mask_path = os.path.join(masks_dir, f"frame_{int(args.frame_num):06d}.png")
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Convert pixels=1 to 255 for visibility
display_mask = (mask > 0).astype(np.uint8) * 255

plt.figure(figsize=(8, 6))
plt.imshow(display_mask, cmap="gray")
plt.title("Ground Truth Mask (scaled)")
plt.axis("off")
plt.show()
