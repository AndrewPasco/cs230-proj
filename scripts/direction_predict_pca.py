import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# Paths
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pred_dir = os.path.join(root_dir, "data/pred/")
out_dir = os.path.join(root_dir, "data/pred_viz/")
os.makedirs(out_dir, exist_ok=True)

# CSV to store directions
csv_data = []

# Loop over prediction images
for img_path in sorted(glob(os.path.join(pred_dir, "*_pred.png"))):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is binary
    mask_binary = (img > 0).astype(np.uint8)

    # Find connected components (each separate object)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Prepare visualization
    vis = cv2.cvtColor(img * 255, cv2.COLOR_GRAY2BGR)

    for label_id in range(1, num_labels):  # skip background
        component_mask = (labels == label_id).astype(np.uint8)
        coords = np.column_stack(np.where(component_mask > 0))  # (y,x)

        if len(coords) < 2:
            continue

        # PCA for direction
        mean = coords.mean(axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov)
        principal_dir = eigvecs[:, np.argmax(eigvals)]

        # Scale up for viz
        scale = 50
        pt1 = tuple(mean[::-1].astype(int))
        pt2 = tuple((mean + principal_dir * scale)[::-1].astype(int))
        cv2.arrowedLine(vis, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)

        # Save data
        csv_data.append(
            {
                "filename": filename,
                "object_id": int(label_id),
                "cx": mean[1],
                "cy": mean[0],
                "vx": principal_dir[1],
                "vy": principal_dir[0],
            }
        )

    # Save visualization
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, vis)

# Save CSV
df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(out_dir, "predicted_directions.csv"), index=False)

print(f"Processed {len(glob(os.path.join(pred_dir, '*_pred.png')))} images.")
print(f"Results saved to {out_dir}")
