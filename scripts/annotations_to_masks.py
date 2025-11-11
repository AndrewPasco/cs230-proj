#!/usr/bin/env python3

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image

# Frame ranges to include
frames = [
    [0, 268],
    [286, 295],
    [354, 440],
    [460, 597],
    [641, 692],
    [703, 756],
    [764, 812],
]

# Paths
video_path = "/home/apasco/cs230/cs230-proj/data/data-2025-10-09.mp4"
xml_path = "/home/apasco/cs230/cs230-proj/data/annotations-2025-10-09-v2.xml"
images_dir = "/home/apasco/cs230/cs230-proj/data/images"
masks_dir = "/home/apasco/cs230/cs230-proj/data/masks"
labels_csv = "/home/apasco/cs230/cs230-proj/data/labels.csv"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Load XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Build mapping: image_name -> list of annotations
frame_annotations = {}
for image_tag in root.findall("image"):
    img_name = image_tag.attrib["name"]
    annos = []

    # Ellipse annotations
    for ellipse in image_tag.findall("ellipse"):
        cx = float(ellipse.attrib["cx"])
        cy = float(ellipse.attrib["cy"])
        rx = float(ellipse.attrib["rx"])
        ry = float(ellipse.attrib["ry"])
        label = ellipse.attrib["label"]
        annos.append(
            {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "label": label}
        )

    # Polygon annotations
    for poly in image_tag.findall("polygon"):
        label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = np.array(
            [
                [int(float(x)), int(float(y))]
                for x, y in (p.split(",") for p in points_str.split(";"))
            ],
            np.int32,
        )
        annos.append({"type": "polygon", "points": points, "label": label})

    if annos:
        frame_annotations[img_name] = annos

print(f"Found {len(frame_annotations)} annotated frames.")

# Label mapping
class_labels = sorted(
    {anno["label"] for annos in frame_annotations.values() for anno in annos}
)
class_map = {label: i + 1 for i, label in enumerate(class_labels)}  # background=0
print("Class map:", class_map)


# Video iteration helper
def frame_in_ranges(idx, ranges):
    for lo, hi in ranges:
        if lo <= idx <= hi:
            return True
    return False


# Process video
cap = cv2.VideoCapture(video_path)
frame_idx = 0
label_records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not frame_in_ranges(frame_idx, frames):
        frame_idx += 1
        continue

    frame_name = f"frame_{frame_idx:06d}"
    rgb_path = os.path.join(images_dir, f"{frame_name}.png")
    mask_path = os.path.join(masks_dir, f"{frame_name}.png")

    # Save the RGB frame
    cv2.imwrite(rgb_path, frame)

    # Create a mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    has_annotation = 0

    if frame_name in frame_annotations:
        annos = frame_annotations[frame_name]

        # If there are two ellipses, remove the inner one (smaller rx+ry)
        ellipses = [a for a in annos if a["type"] == "ellipse"]
        if len(ellipses) == 2:
            ellipses.sort(key=lambda e: e["rx"] + e["ry"], reverse=True)
            annos = [ellipses[0]] + [a for a in annos if a["type"] != "ellipse"]

        for anno in annos:
            if anno["type"] == "ellipse":
                center = (int(round(anno["cx"])), int(round(anno["cy"])))
                axes = (int(round(anno["rx"])), int(round(anno["ry"])))
                cv2.ellipse(mask, center, axes, 0, 0, 360, class_map[anno["label"]], -1)
            elif anno["type"] == "polygon":
                cv2.fillPoly(mask, [anno["points"]], class_map[anno["label"]])

        has_annotation = 1

        # Only save mask if non-empty
        if np.any(mask):
            Image.fromarray(mask).save(mask_path)

    # Record binary label (even if no annotation)
    label_records.append({"frame_name": frame_name, "has_feature": has_annotation})

    print(f"[{frame_idx}] saved {rgb_path}, label={has_annotation}")
    frame_idx += 1

cap.release()

# Write labels.csv
df = pd.DataFrame(label_records)
df.to_csv(labels_csv, index=False)
print(f"Labels saved to {labels_csv} with {len(df)} entries.")
print("Done!")
