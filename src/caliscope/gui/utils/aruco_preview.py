"""Shared ArUco marker preview rendering utility.

Used by ProjectSetupView and potentially other views to render
ArUco marker previews without coupling the domain model to Qt.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from caliscope.core.aruco_target import ArucoTarget


def render_aruco_pixmap(target: ArucoTarget, marker_id: int, size: int) -> QPixmap:
    """Render ArUco marker as a QPixmap for display.

    Uses ArucoTarget.generate_marker_image() to create the annotated marker,
    then converts BGR to QPixmap. Scales to fit within target size while
    maintaining aspect ratio.
    """
    # Scale proportionally for requested display size
    # 4x multiplier renders at high resolution for crisp text after downscale
    ppm = int(size / target.marker_size_m * 4.0)
    bgr = target.generate_marker_image(marker_id, pixels_per_meter=ppm)

    # Convert BGR to RGB for Qt
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w

    # Create QImage from numpy array; .copy() ensures data ownership
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimage.copy())
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def render_aruco_target_pixmap(target: ArucoTarget, size: int) -> QPixmap:
    """Render the appropriate preview for a single marker or cube target.

    A 1x1 cube still previews like the legacy single-marker face. NxN cubes use
    the cube-net image so the setup panel communicates that each physical face
    contains a full marker grid.
    """
    if target.is_cube:
        markers_per_side = target.cube_markers_per_side or 1
        if markers_per_side > 1:
            ppm = max(200, int(size / max(target.cube_edge_length_m or target.marker_size_m, 1e-6) * 1.5))
            bgr = target.generate_cube_layout_image(pixels_per_meter=ppm)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage.copy())
            return pixmap.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        marker_id = target.ordered_marker_ids[0] if target.ordered_marker_ids else 0
        return render_aruco_pixmap(target, marker_id, size)

    marker_ids = target.ordered_marker_ids
    if len(marker_ids) <= 1:
        marker_id = marker_ids[0] if marker_ids else 0
        return render_aruco_pixmap(target, marker_id, size)

    columns = min(3, len(marker_ids))
    rows = int(np.ceil(len(marker_ids) / columns))
    tile_size = max(120, size // max(1, columns))
    ppm = int(tile_size / target.marker_size_m * 4.0)
    images = [target.generate_marker_image(marker_id, pixels_per_meter=ppm) for marker_id in marker_ids]

    tile_height = max(image.shape[0] for image in images)
    tile_width = max(image.shape[1] for image in images)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        y0 = row * tile_height
        x0 = col * tile_width
        canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimage.copy())
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
