"""
TendonLabeler class for generating masks of tendon in phantom images.

Based on original implementation from test1.py.
"""

import math
import trimesh
import numpy as np
import cv2
from scipy.interpolate import NearestNDInterpolator



class TendonLabeler:
    def __init__(self, phantom_path, rotation_deg=0):
        # Load and extract tendon from phantom
        phantom = trimesh.load(phantom_path, process=True)

        # Apply rotation if specified (around Z-axis) depends on the experiment run.
        if rotation_deg != 0:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                math.radians(rotation_deg),
                [0, 0, 1]  # Z-axis
            )
            phantom.apply_transform(rotation_matrix)


        # Tendon bounding box (We just select the tendon from the STL)
        X_low, X_high = -0.01, 0.01
        Y_low, Y_high = -0.1, 0.8
        Z_low, Z_high = 0.009, 0.021 # Tendon Try this 0.015 before it was 0.02

        verts = phantom.vertices
        mask = (verts[:, 0] >= X_low) & (verts[:, 0] <= X_high) & \
               (verts[:, 1] >= Y_low) & (verts[:, 1] <= Y_high) & \
               (verts[:, 2] >= Z_low) & (verts[:, 2] <= Z_high)

        vertex_indices = np.where(mask)[0]
        face_mask = np.all(np.isin(phantom.faces, vertex_indices), axis=1)
        tendon_face_indices = np.where(face_mask)[0]

        # Extract Tendon Mesh100
        self.tendon_mesh = phantom.submesh([tendon_face_indices], append=True)
        rotation_matrix_y = trimesh.transformations.rotation_matrix( # Allows us to see the tendon from the other face.
            math.radians(180), [0, 1, 0])
        self.tendon_mesh.apply_transform(rotation_matrix_y)

        # Manual center (per frame)
        self.tendon_center_px = None

        # Camera params. XY is assumed to be aligned with phantom
        self.cam_z = 0.0040

    def set_center(self, center_px):
        """Manually set tendon center in pixels."""
        self.tendon_center_px = center_px

    def generate_mask(self, width_px, height_px, cam_y=0.32):
        """Generate mask and depth map for given image size and camera Y position.

        Returns:
            mask: Binary mask (uint8, 0-255) of tendon projection
            depth_map: Depth map (float32, meters) from camera to tendon surface
        """

        image_center_px = width_px // 2
        offset_px = self.tendon_center_px - image_center_px if self.tendon_center_px else 0

        # Camera intrinsics
        HFOV_deg, VFOV_deg = 120, 66
        fx = (0.5 * width_px) / math.tan(math.radians(HFOV_deg) / 2)
        fy = (0.5 * height_px) / math.tan(math.radians(VFOV_deg) / 2)
        cx, cy = width_px / 2, height_px / 2

        # Sample surface points
        points, _ = trimesh.sample.sample_surface(self.tendon_mesh, count=100000)

        # The argument cam_y will come from a csv with the X,Y,Z TCP positions
        y_camera = cam_y - 0.327  # cam y goes from 0.26 to 0.38, so we center at 0.32
        # Transform to camera frame
        X = points[:, 0]
        Y = points[:, 1] - y_camera
        Z = self.cam_z - points[:, 2]

        # Filter valid points
        centerline = 0.0  # Approximate Z of centerline#TODO CHECK THIS decrease but no less tjhan 0.002
        valid = Z > centerline # So far the centerline is around 0.0145 m 
        X, Y, Z = X[valid], Y[valid], Z[valid]

        # Project to pixels + apply offset
        u = (fx * X / Z + cx + offset_px).astype(int)
        v = (fy * Y / Z + cy).astype(int)

        # Create mask and depth map
        in_bounds = (u >= 0) & (u < width_px) & (v >= 0) & (v < height_px)
        mask = np.zeros((height_px, width_px), dtype=np.uint8)
        mask[v[in_bounds], u[in_bounds]] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        sparse_depth = np.zeros((height_px, width_px), dtype=np.float32)
        np.maximum.at(sparse_depth, (v[in_bounds], u[in_bounds]), Z[in_bounds])

        # Filter for half the points above the tendon.
        pixel_min = np.min(sparse_depth)
        pixel_max = np.max(sparse_depth)
        depth_threshold = pixel_min + 0.9 * (pixel_max - pixel_min)
        sparse_depth[sparse_depth < depth_threshold] = 0.0

        # Fill depth using the mask as a contour.
        depth_map = self._fill_depth_in_mask(sparse_depth, mask)
        depth_map_in_mm = depth_map.astype(np.float32) * 1000.0  # convert to mm
        return mask, depth_map_in_mm

    def _fill_depth_in_mask(self, sparse_depth, mask):
        """Interpolate depth within mask region."""
        valid = sparse_depth > 0
        if not np.any(valid):
            return sparse_depth

        coords = np.array(np.nonzero(valid)).T
        values = sparse_depth[valid]
        interp = NearestNDInterpolator(coords, values)

        # Fill everywhere mask is valid
        mask_coords = np.array(np.nonzero(mask > 0)).T
        
        depth_map = np.zeros_like(sparse_depth)
        depth_map[mask > 0] = interp(mask_coords)

        return depth_map