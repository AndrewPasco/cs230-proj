#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import cv2
import matplotlib.pyplot as plt
import numpy as np

# parser = ArgumentParser()
# parser.add_argument(
#     "--frame_num",
#     default=580,
#     help="Frame number to be displayed",
# )
# args = parser.parse_args()

masks_dir = os.path.join(os.path.dirname(__file__), "../../output/p2/masks/")
masks_list = os.listdir(masks_dir)
count = 0
different_pixels = set()
for mask_file in masks_list:
    # print(mask_file)
    # print whether any pixel is not 0
    mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_UNCHANGED)
    if mask is not None and np.any(mask != 0):
        count += 1
        different_pixels.update(np.unique(mask))
print("Total masks:", len(masks_list))
print(f"Number of masks with non-zero pixels: {count}")
print(f"Unique pixel values in non-zero masks: {sorted(different_pixels)}")
mask_path = os.path.join(masks_dir, "p2_d2s_25N-2025-11-25_22.26.58_frame_00551.png")
# print image size
print(f"Image size: {cv2.imread(mask_path).shape}")

# print if any of the pixels are not 0:
print(f"Any pixels with value not 0: {np.any(cv2.imread(mask_path) != 0)}")

mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Convert pixels=1 to 255 for visibility
display_mask = (mask > 0).astype(np.uint8) * 255

plt.figure(figsize=(8, 6))
plt.imshow(display_mask, cmap="gray")
plt.title("Ground Truth Mask (scaled)")
plt.axis("off")
plt.show()
