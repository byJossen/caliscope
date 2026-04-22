"""Visualize lens model effects for user inspection.

The intrinsic-calibration preview should match the geometry used elsewhere in
the app. In particular, fisheye previews should use the same output camera
matrix that the calibrated camera uses for point undistortion and reprojection.
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from caliscope.cameras.camera_array import CameraData

logger = logging.getLogger(__name__)


def _draw_dashed_line(
    frame: NDArray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 6,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2

    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)

    if length == 0:
        return

    ux = dx / length
    uy = dy / length

    segment_length = dash_length + gap_length
    pos = 0.0

    while pos < length:
        start_x = int(x1 + ux * pos)
        start_y = int(y1 + uy * pos)

        end_pos = min(pos + dash_length, length)
        end_x = int(x1 + ux * end_pos)
        end_y = int(y1 + uy * end_pos)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

        pos += segment_length


def _draw_dashed_rect(
    frame: NDArray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 6,
) -> None:
    """Draw a dashed rectangle."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    _draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x2, y2), (x1, y2), color, thickness, dash_length, gap_length)
    _draw_dashed_line(frame, (x1, y2), (x1, y1), color, thickness, dash_length, gap_length)


class LensModelVisualizer:
    """Visualizes lens model effects for user inspection.

    The preview uses the calibrated camera's native output geometry so the
    intrinsic widget matches the rest of Caliscope. That means the preview may
    clip content near the image edges, but it avoids the misleading "zoomed out"
    full-content view that made valid fisheye calibrations look wrong.
    """

    BOUNDARY_COLOR = (255, 255, 0)  # BGR: cyan
    BOUNDARY_THICKNESS = 2

    def __init__(self, camera: CameraData):
        """Initialize the visualizer.

        Args:
            camera: CameraData with calibrated intrinsics
        """
        self._camera = camera

        self._map1: NDArray | None = None
        self._map2: NDArray | None = None
        self._content_expands = False
        self._boundary_rect: tuple[tuple[int, int], tuple[int, int]] | None = None

        self._compute_undistortion_params()

    @property
    def is_ready(self) -> bool:
        """Check if the visualizer has valid parameters."""
        return self._map1 is not None

    @property
    def content_expands_beyond_frame(self) -> bool:
        """True if undistortion causes content to extend past original frame bounds.

        When True, undistort() draws a dashed boundary showing the original frame.
        View can use this to conditionally display a legend.
        """
        return self._content_expands

    def _compute_undistortion_params(self) -> None:
        """Compute remap tables using the camera's native output geometry."""
        if self._camera.matrix is None or self._camera.distortions is None:
            logger.debug(f"Camera {self._camera.cam_id} lacks calibration")
            return

        w, h = self._camera.size
        matrix = self._camera.matrix
        distortions = self._camera.distortions
        self._content_expands = False
        self._boundary_rect = None

        if self._camera.uses_fisheye_model:
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                matrix, distortions, np.eye(3), matrix, (w, h), cv2.CV_16SC2
            )
        else:
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                matrix, distortions, np.eye(3), matrix, (w, h), cv2.CV_16SC2
            )

        logger.debug(
            f"LensModelVisualizer cam {self._camera.cam_id}: preview uses native output geometry"
        )

    def undistort(self, frame: NDArray) -> NDArray:
        """Undistort a frame for visualization.

        If content expands beyond the original frame, draws a dashed boundary
        showing where the original frame was.

        Args:
            frame: Input image (possibly with composited overlays)

        Returns:
            Undistorted frame with boundary overlay if applicable
        """
        if self._map1 is None or self._map2 is None:
            logger.warning(f"Cannot undistort frame for cam {self._camera.cam_id}: not ready")
            return frame

        result = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

        if self._content_expands and self._boundary_rect is not None:
            _draw_dashed_rect(
                result,
                self._boundary_rect[0],
                self._boundary_rect[1],
                self.BOUNDARY_COLOR,
                self.BOUNDARY_THICKNESS,
            )

        return result
