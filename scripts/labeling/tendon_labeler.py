"""
TendonLabeler class for generating masks of tendon in phantom images.

Based on original implementation from test1.py.
"""

import math
import trimesh
import numpy as np


class TendonLabeler:
    def __init__(self, phantom_path):
        # Load and extract tendon from phantom
        phantom = trimesh.load(phantom_path, process=True)

        # Tendon bounding box (We just select the tendon from the STL)
        X_low, X_high = -0.01, 0.01
        Y_low, Y_high = -0.1, 0.1
        Z_low, Z_high = 0.005, 0.02

        verts = phantom.vertices
        mask = (verts[:, 0] >= X_low) & (verts[:, 0] <= X_high) & \
               (verts[:, 1] >= Y_low) & (verts[:, 1] <= Y_high) & \
               (verts[:, 2] >= Z_low) & (verts[:, 2] <= Z_high)

        vertex_indices = np.where(mask)[0]
        face_mask = np.all(np.isin(phantom.faces, vertex_indices), axis=1)
        tendon_face_indices = np.where(face_mask)[0]

        # Extract Tendon Mesh
        self.tendon_mesh = phantom.submesh([tendon_face_indices], append=True)

        # Manual center (per frame)
        self.tendon_center_px = None

        # Camera params. XY is assumed to be aligned with phantom
        self.cam_z = 0.0127 + 0.006594 + 0.005

    def set_center(self, center_px):
        """Manually set tendon center in pixels."""
        self.tendon_center_px = center_px

    def generate_mask(self, width_px, height_px, cam_y=0.32):
        """Generate mask for given image size and camera Y position."""

        image_center_px = width_px // 2
        offset_px = self.tendon_center_px - image_center_px if self.tendon_center_px else 0

        # Camera intrinsics
        HFOV_deg, VFOV_deg = 120, 66
        fx = (0.5 * width_px) / math.tan(math.radians(HFOV_deg) / 2)
        fy = (0.5 * height_px) / math.tan(math.radians(VFOV_deg) / 2)
        cx, cy = width_px / 2, height_px / 2

        # Sample surface points
        points, _ = trimesh.sample.sample_surface(self.tendon_mesh, count=500000)

        # The argument cam_y will come from a csv with the X,Y,Z TCP positions
        y_camera = cam_y - 0.327  # cam y goes from 0.26 to 0.38, so we center at 0.32
        # Transform to camera frame
        X = points[:, 0]
        Y = points[:, 1] - y_camera
        Z = self.cam_z - points[:, 2]

        # Filter valid points
        valid = Z > 0.001
        X, Y, Z = X[valid], Y[valid], Z[valid]

        # Project to pixels + apply offset
        u = (fx * X / Z + cx + offset_px).astype(int)
        v = (fy * Y / Z + cy).astype(int)

        # Create mask
        in_bounds = (u >= 0) & (u < width_px) & (v >= 0) & (v < height_px)
        mask = np.zeros((height_px, width_px), dtype=np.uint8)
        mask[v[in_bounds], u[in_bounds]] = 255

        return mask
