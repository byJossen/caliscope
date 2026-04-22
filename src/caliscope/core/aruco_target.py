from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import rtoml
from numpy.typing import NDArray


_CUBE_FACE_NAMES = ("front", "right", "back", "left", "top", "bottom")
_CUBE_NET_POSITIONS = {
    "top": (0, 1),
    "left": (1, 0),
    "front": (1, 1),
    "right": (1, 2),
    "back": (1, 3),
    "bottom": (2, 1),
}


def _cube_face_specs(
    edge_length_m: float,
) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...]:
    """Return the fixed face coordinate frames used by every ArUco cube target.

    Each tuple contains ``(face_name, face_center, local_x_axis, local_y_axis)``.
    The axes describe how a printed face is viewed from outside the cube:
    local X points to the viewer's right and local Y points up. This convention
    lets row/column marker placement stay human-readable while still producing
    outward-facing OpenCV corner coordinates in the cube object frame.
    """
    half_edge = edge_length_m / 2.0
    return (
        ("front", (0.0, 0.0, +half_edge), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("right", (+half_edge, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        ("back", (0.0, 0.0, -half_edge), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("left", (-half_edge, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ("top", (0.0, +half_edge, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ("bottom", (0.0, -half_edge, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )


def _infer_cube_markers_per_side(marker_count: int) -> int:
    """Infer ``N`` for legacy cube TOMLs that predate ``markers_per_side``.

    Old cube files stored only marker corners. If the total marker count is a
    perfect ``6 * N * N`` grid, this recovers ``N``; otherwise the loader falls
    back to 1 so historic one-marker-per-face projects remain valid.
    """
    if marker_count <= 0 or marker_count % len(_CUBE_FACE_NAMES) != 0:
        return 1

    per_face_count = marker_count // len(_CUBE_FACE_NAMES)
    candidate = int(round(np.sqrt(per_face_count)))
    if candidate > 0 and candidate * candidate == per_face_count:
        return candidate
    return 1


def _draw_dashed_line(
    image: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 20,
    gap_length: int = 10,
) -> None:
    """Draw a dashed line between two points."""
    start_arr = np.array(start, dtype=np.float64)
    end_arr = np.array(end, dtype=np.float64)
    segment = end_arr - start_arr
    total_length = float(np.linalg.norm(segment))
    if total_length == 0:
        return

    direction = segment / total_length
    drawn = 0.0
    while drawn < total_length:
        dash_end = min(drawn + dash_length, total_length)
        p0 = tuple(np.round(start_arr + direction * drawn).astype(int))
        p1 = tuple(np.round(start_arr + direction * dash_end).astype(int))
        cv2.line(image, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        drawn += dash_length + gap_length


def _draw_dashed_rectangle(
    image: NDArray[np.uint8],
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 20,
    gap_length: int = 10,
) -> None:
    """Draw a dashed rectangle."""
    x0, y0 = top_left
    x1, y1 = bottom_right
    _draw_dashed_line(image, (x0, y0), (x1, y0), color, thickness, dash_length, gap_length)
    _draw_dashed_line(image, (x1, y0), (x1, y1), color, thickness, dash_length, gap_length)
    _draw_dashed_line(image, (x1, y1), (x0, y1), color, thickness, dash_length, gap_length)
    _draw_dashed_line(image, (x0, y1), (x0, y0), color, thickness, dash_length, gap_length)


@dataclass(frozen=True, slots=True)
class ArucoTarget:
    """A rigid calibration target with ArUco markers at known 3D positions.

    Used for extrinsic calibration. The target defines a coordinate frame
    where markers have known corner positions (in meters).
    """

    dictionary: int
    corners: dict[int, NDArray[np.float64]]  # marker_id -> (4, 3) positions
    marker_size_m: float
    inverted: bool = False
    layout: str = "single_marker"
    marker_order: tuple[int, ...] | None = None
    cube_edge_length_m: float | None = None
    cube_markers_per_side: int | None = None

    @staticmethod
    def _build_marker_corners(
        center: Sequence[float],
        x_axis: Sequence[float],
        y_axis: Sequence[float],
        marker_size_m: float,
    ) -> NDArray[np.float64]:
        """Build one marker's 3D corner coordinates from its face-local frame.

        ``center`` is the marker center in object coordinates. ``x_axis`` and
        ``y_axis`` are the marker's local right/up directions when viewing that
        cube face from outside. The returned order is OpenCV ArUco order:
        top-left, top-right, bottom-right, bottom-left.
        """
        center_arr = np.array(center, dtype=np.float64)
        x_axis_arr = np.array(x_axis, dtype=np.float64)
        y_axis_arr = np.array(y_axis, dtype=np.float64)

        x_axis_arr /= np.linalg.norm(x_axis_arr)
        y_axis_arr /= np.linalg.norm(y_axis_arr)

        s = marker_size_m / 2.0
        return np.array(
            [
                center_arr - s * x_axis_arr + s * y_axis_arr,  # TL
                center_arr + s * x_axis_arr + s * y_axis_arr,  # TR
                center_arr + s * x_axis_arr - s * y_axis_arr,  # BR
                center_arr - s * x_axis_arr - s * y_axis_arr,  # BL
            ],
            dtype=np.float64,
        )

    @staticmethod
    def single_marker(
        marker_id: int = 0,
        marker_size_m: float = 0.05,
        dictionary: int = cv2.aruco.DICT_4X4_100,
        inverted: bool = False,
    ) -> "ArucoTarget":
        """Factory for single-marker target (most common case).

        Creates a target with one marker centered at origin. Corner positions
        follow OpenCV's ArUco convention: origin at center, X right, Y up, Z out.
        Corners ordered TL, TR, BR, BL as returned by detectMarkers.
        """
        s = marker_size_m / 2
        # OpenCV ArUco convention: Y points UP (standard math frame, not image frame)
        corner_positions = np.array(
            [
                [-s, +s, 0.0],  # TL
                [+s, +s, 0.0],  # TR
                [+s, -s, 0.0],  # BR
                [-s, -s, 0.0],  # BL
            ],
            dtype=np.float64,
        )

        return ArucoTarget(
            dictionary=dictionary,
            corners={marker_id: corner_positions},
            marker_size_m=marker_size_m,
            inverted=inverted,
            layout="single_marker",
            marker_order=(marker_id,),
        )

    @staticmethod
    def compute_cube_face_gap_m(
        *,
        markers_per_side: int,
        marker_size_m: float,
        edge_length_m: float,
    ) -> float:
        """Compute the derived spacing for an ``N x N`` marker grid on a face.

        The cube edge length and marker size are user-controlled. The remaining
        face width is split evenly into ``N + 1`` gaps: one at each edge and one
        between each neighboring marker. This preserves centered 1x1 behavior
        and avoids adding a separate user-facing gap parameter.
        """
        if markers_per_side < 1:
            raise ValueError("markers_per_side must be at least 1")
        if edge_length_m <= 0:
            raise ValueError("edge_length_m must be positive")
        if marker_size_m <= 0:
            raise ValueError("marker_size_m must be positive")
        if markers_per_side * marker_size_m >= edge_length_m:
            raise ValueError("markers_per_side * marker_size_m must be less than edge_length_m")
        return (edge_length_m - markers_per_side * marker_size_m) / (markers_per_side + 1)

    @staticmethod
    def _cube_face_grid_centers_m(
        *,
        markers_per_side: int,
        marker_size_m: float,
        edge_length_m: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Return face-local marker-center offsets for rows and columns.

        ``x_centers[col]`` moves left-to-right across the printed face.
        ``y_centers[row]`` moves top-to-bottom, so row 0 is physically the top
        row when the face is viewed from outside the cube.
        """
        gap = ArucoTarget.compute_cube_face_gap_m(
            markers_per_side=markers_per_side,
            marker_size_m=marker_size_m,
            edge_length_m=edge_length_m,
        )
        half_edge = edge_length_m / 2.0
        step = marker_size_m + gap
        x_centers = np.array(
            [-half_edge + gap + marker_size_m / 2.0 + col * step for col in range(markers_per_side)],
            dtype=np.float64,
        )
        y_centers = np.array(
            [half_edge - gap - marker_size_m / 2.0 - row * step for row in range(markers_per_side)],
            dtype=np.float64,
        )
        return x_centers, y_centers, gap

    @staticmethod
    def cube(
        first_marker_id: int = 0,
        marker_size_m: float = 0.05,
        edge_length_m: float = 0.12,
        dictionary: int = cv2.aruco.DICT_4X4_100,
        marker_ids: Sequence[int] | None = None,
        inverted: bool = False,
        markers_per_side: int = 1,
    ) -> "ArucoTarget":
        """Factory for a rigid cube with an ``N x N`` marker grid on each face.

        Marker IDs are assigned face-major in fixed face order:
        front, right, back, left, top, bottom; then row-major within each face.
        The cube edge length stays fixed as ``markers_per_side`` changes, and
        marker centers are distributed with the uniform derived gap from
        :meth:`compute_cube_face_gap_m`.
        """
        ArucoTarget.compute_cube_face_gap_m(
            markers_per_side=markers_per_side,
            marker_size_m=marker_size_m,
            edge_length_m=edge_length_m,
        )

        total_markers = len(_CUBE_FACE_NAMES) * markers_per_side * markers_per_side

        if marker_ids is None:
            marker_ids = [first_marker_id + i for i in range(total_markers)]
        elif len(marker_ids) != total_markers:
            raise ValueError(f"marker_ids must contain exactly {total_markers} IDs for a cube target")

        marker_order = tuple(int(marker_id) for marker_id in marker_ids)
        if len(set(marker_order)) != len(marker_order):
            raise ValueError("marker_ids must be unique")

        face_specs = _cube_face_specs(edge_length_m)
        x_centers, y_centers, _ = ArucoTarget._cube_face_grid_centers_m(
            markers_per_side=markers_per_side,
            marker_size_m=marker_size_m,
            edge_length_m=edge_length_m,
        )

        corners = {}
        marker_iter = iter(marker_order)
        for _, face_center, x_axis, y_axis in face_specs:
            face_center_arr = np.array(face_center, dtype=np.float64)
            x_axis_arr = np.array(x_axis, dtype=np.float64)
            y_axis_arr = np.array(y_axis, dtype=np.float64)
            for row in range(markers_per_side):
                for col in range(markers_per_side):
                    marker_id = next(marker_iter)
                    center = face_center_arr + x_centers[col] * x_axis_arr + y_centers[row] * y_axis_arr
                    corners[marker_id] = ArucoTarget._build_marker_corners(center, x_axis, y_axis, marker_size_m)

        return ArucoTarget(
            dictionary=dictionary,
            corners=corners,
            marker_size_m=marker_size_m,
            inverted=inverted,
            layout="cube",
            marker_order=marker_order,
            cube_edge_length_m=edge_length_m,
            cube_markers_per_side=markers_per_side,
        )

    @property
    def marker_ids(self) -> list[int]:
        """All marker IDs this target tracks, sorted."""
        return sorted(self.corners.keys())

    @property
    def ordered_marker_ids(self) -> list[int]:
        """Marker IDs in their configured layout order when available."""
        if self.marker_order is not None:
            return list(self.marker_order)
        return self.marker_ids

    @property
    def is_cube(self) -> bool:
        """Whether this target uses the six-face rigid cube layout."""
        return self.layout == "cube"

    def _require_cube_marker_layout(self, marker_id: int) -> tuple[int, int, int]:
        """Return ``(face_index, row, col)`` metadata for one cube marker.

        The metadata is derived from ``marker_order`` rather than geometry so it
        remains stable across save/load round trips. Rows and columns are
        zero-based here; user-facing labels add 1 for readability.
        """
        if not self.is_cube or self.marker_order is None:
            raise ValueError("Cube marker metadata is only available for cube targets")
        if self.cube_markers_per_side is None:
            raise ValueError("Cube target is missing markers_per_side metadata")
        if marker_id not in self.marker_order:
            raise KeyError(f"Marker {marker_id} not in cube target")

        face_marker_count = self.cube_markers_per_side * self.cube_markers_per_side
        marker_index = self.marker_order.index(marker_id)
        face_index, within_face_index = divmod(marker_index, face_marker_count)
        row, col = divmod(within_face_index, self.cube_markers_per_side)
        return face_index, row, col

    def get_cube_face_name(self, marker_id: int) -> str:
        """Return the physical face name that owns ``marker_id``."""
        face_index, _, _ = self._require_cube_marker_layout(marker_id)
        return _CUBE_FACE_NAMES[face_index]

    def get_cube_face_row(self, marker_id: int) -> int:
        """Return the zero-based row index for ``marker_id`` on its face."""
        _, row, _ = self._require_cube_marker_layout(marker_id)
        return row

    def get_cube_face_col(self, marker_id: int) -> int:
        """Return the zero-based column index for ``marker_id`` on its face."""
        _, _, col = self._require_cube_marker_layout(marker_id)
        return col

    def get_cube_face_marker_ids(self, face_name: str) -> list[int]:
        """Return marker IDs for one cube face in row-major print order."""
        if not self.is_cube or self.marker_order is None:
            raise ValueError("Cube face marker IDs are only available for cube targets")
        if self.cube_markers_per_side is None:
            raise ValueError("Cube target is missing markers_per_side metadata")
        if face_name not in _CUBE_FACE_NAMES:
            raise ValueError(f"Unknown cube face: {face_name}")

        face_index = _CUBE_FACE_NAMES.index(face_name)
        face_marker_count = self.cube_markers_per_side * self.cube_markers_per_side
        start = face_index * face_marker_count
        end = start + face_marker_count
        return list(self.marker_order[start:end])

    def get_cube_face_gap_m(self) -> float | None:
        """Return the derived cube face gap, or ``None`` for non-cube targets."""
        if not self.is_cube or self.cube_edge_length_m is None or self.cube_markers_per_side is None:
            return None
        return self.compute_cube_face_gap_m(
            markers_per_side=self.cube_markers_per_side,
            marker_size_m=self.marker_size_m,
            edge_length_m=self.cube_edge_length_m,
        )

    def get_marker_label(self, marker_id: int) -> str:
        """Return a stable label for previews, exports, and debugging.

        Cube labels include face, 1-based row, 1-based column, and marker ID
        so a printed sheet can be checked against the generated geometry.
        """
        if self.is_cube and self.marker_order is not None and marker_id in self.marker_order:
            face_name = self.get_cube_face_name(marker_id)
            row = self.get_cube_face_row(marker_id) + 1
            col = self.get_cube_face_col(marker_id) + 1
            return f"{face_name}_r{row}_c{col}_id_{marker_id}"
        return f"id_{marker_id}"

    @classmethod
    def from_toml(cls, path: Path) -> "ArucoTarget":
        """Load ArucoTarget from TOML file.

        TOML format:
            dictionary = 0
            marker_size_m = 0.05
            [corners.0]
            positions = [[-0.025, -0.025, 0.0], [0.025, -0.025, 0.0], ...]

        Raises:
            PersistenceError: If file doesn't exist or format is invalid
        """
        from caliscope.persistence import PersistenceError

        if not path.exists():
            raise PersistenceError(f"ArucoTarget file not found: {path}")

        try:
            data = rtoml.load(path)

            dictionary = data["dictionary"]
            marker_size_m = data["marker_size_m"]
            inverted = data.get("inverted", False)
            layout = data.get("layout", "single_marker")
            marker_order = None
            cube_edge_length_m = None
            cube_markers_per_side = None

            if layout == "cube":
                cube_data = data.get("cube", {})
                marker_ids = cube_data.get("marker_ids")
                if marker_ids is not None:
                    marker_order = tuple(int(marker_id) for marker_id in marker_ids)
                edge_length = cube_data.get("edge_length_m")
                if edge_length is not None:
                    cube_edge_length_m = float(edge_length)
                markers_per_side = cube_data.get("markers_per_side")
                if markers_per_side is not None:
                    cube_markers_per_side = int(markers_per_side)

            corners: dict[int, NDArray[np.float64]] = {}
            for marker_id_str, corner_data in data.get("corners", {}).items():
                marker_id = int(marker_id_str)
                positions = np.array(corner_data["positions"], dtype=np.float64)
                if positions.shape != (4, 3):
                    raise ValueError(f"Marker {marker_id} has invalid shape: {positions.shape}")
                corners[marker_id] = positions

            if layout == "cube" and marker_order is None:
                marker_order = tuple(sorted(corners.keys()))
            if layout == "cube" and cube_markers_per_side is None:
                cube_markers_per_side = _infer_cube_markers_per_side(len(marker_order or corners))

            return cls(
                dictionary=dictionary,
                corners=corners,
                marker_size_m=marker_size_m,
                inverted=inverted,
                layout=layout,
                marker_order=marker_order,
                cube_edge_length_m=cube_edge_length_m,
                cube_markers_per_side=cube_markers_per_side,
            )
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to load ArucoTarget from {path}: {e}") from e

    def to_toml(self, path: Path) -> None:
        """Save ArucoTarget to TOML file.

        Raises:
            PersistenceError: If write fails
        """
        from caliscope.persistence import PersistenceError, _safe_write_toml

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            corners_data = {}
            for marker_id, positions in self.corners.items():
                corners_data[str(marker_id)] = {"positions": positions.tolist()}

            data = {
                "dictionary": self.dictionary,
                "marker_size_m": self.marker_size_m,
                "inverted": self.inverted,
                "layout": self.layout,
                "corners": corners_data,
            }

            if self.is_cube:
                data["cube"] = {
                    "edge_length_m": self.cube_edge_length_m,
                    "markers_per_side": self.cube_markers_per_side if self.cube_markers_per_side is not None else 1,
                    "marker_ids": self.ordered_marker_ids,
                }

            _safe_write_toml(data, path)
        except PersistenceError:
            raise
        except Exception as e:
            raise PersistenceError(f"Failed to save ArucoTarget to {path}: {e}") from e

    def get_corner_positions(self, marker_id: int) -> NDArray[np.float64]:
        """Get (4, 3) corner positions for a marker.

        Raises:
            KeyError: If marker_id not in this target
        """
        return self.corners[marker_id]

    def generate_marker_image(self, marker_id: int, pixels_per_meter: int = 4000) -> NDArray:
        """Generate annotated printable marker image.

        All annotations are in the white border. A small axis legend in the
        bottom-right shows the coordinate frame orientation without occluding
        the marker pattern.

        Raises:
            KeyError: If marker_id not in this target
        """
        if marker_id not in self.corners:
            raise KeyError(f"Marker {marker_id} not in target (available: {self.marker_ids})")

        pixel_size = int(self.marker_size_m * pixels_per_meter)
        face_size_px = None
        if self.is_cube and self.cube_edge_length_m is not None and (self.cube_markers_per_side or 1) == 1:
            face_size_px = int(round(self.cube_edge_length_m * pixels_per_meter))

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixel_size)

        border = pixel_size // 2
        if face_size_px is not None:
            face_padding = max(0, (face_size_px - pixel_size + 1) // 2)
            border = max(border, face_padding + pixel_size // 6)
        bordered = cv2.copyMakeBorder(
            marker_img,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(255.0,),
        )

        if self.inverted:
            bordered = cv2.bitwise_not(bordered)

        annotated = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

        mx, my = border, border  # top-left of marker in bordered image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = pixel_size / 400
        thickness = max(1, int(pixel_size / 100))
        label_thick = max(1, thickness - 1)
        gap = border // 5  # padding between marker edge and labels

        # Corner labels in the border, outside the marker
        label_positions = [
            (mx - gap - int(font_scale * 10), my - gap),  # TL
            (mx + pixel_size + gap, my - gap),  # TR
            (mx + pixel_size + gap, my + pixel_size + gap + int(font_scale * 12)),  # BR
            (mx - gap - int(font_scale * 10), my + pixel_size + gap + int(font_scale * 12)),  # BL
        ]
        for i, (lx, ly) in enumerate(label_positions):
            cv2.putText(annotated, str(i), (lx, ly), font, font_scale, (0, 0, 0), thickness)

        # Axis legend: small square with arrows in the bottom-right border
        legend_size = border // 3
        legend_margin = border // 6
        # Bottom-right border area, below corner label 2
        lx = mx + pixel_size + legend_margin
        ly = my + pixel_size + border * 2 // 3

        # Draw a small gray square representing the marker
        cv2.rectangle(annotated, (lx, ly), (lx + legend_size, ly + legend_size), (200, 200, 200), -1)
        cv2.rectangle(annotated, (lx, ly), (lx + legend_size, ly + legend_size), (0, 0, 0), 1)

        # Origin dot at center of legend square
        cx = lx + legend_size // 2
        cy = ly + legend_size // 2
        cv2.circle(annotated, (cx, cy), max(2, thickness), (0, 0, 0), -1)

        arrow_len = legend_size // 2 + legend_margin
        arrow_thick = max(1, thickness)

        # X-axis: red, pointing right from center
        cv2.arrowedLine(annotated, (cx, cy), (cx + arrow_len, cy), (0, 0, 255), arrow_thick, tipLength=0.25)
        cv2.putText(
            annotated,
            "X",
            (cx + arrow_len + 2, cy + int(font_scale * 5)),
            font,
            font_scale * 0.4,
            (0, 0, 255),
            label_thick,
        )
        # Y-axis: green, pointing UP from center (negative y in image coords)
        cv2.arrowedLine(annotated, (cx, cy), (cx, cy - arrow_len), (0, 180, 0), arrow_thick, tipLength=0.25)
        cv2.putText(
            annotated,
            "Y",
            (cx - int(font_scale * 8), cy - arrow_len - int(font_scale * 3)),
            font,
            font_scale * 0.4,
            (0, 180, 0),
            label_thick,
        )

        # Info text at bottom of border
        size_cm = self.marker_size_m * 100
        info_y = my + pixel_size + border - int(font_scale * 5)
        cv2.putText(
            annotated,
            f"ID: {marker_id}  Size: {size_cm:.1f} cm",
            (mx, info_y),
            font,
            font_scale * 0.5,
            (0, 0, 0),
            label_thick,
        )

        if face_size_px is not None:
            face_x0 = mx - (face_size_px - pixel_size) // 2
            face_y0 = my - (face_size_px - pixel_size) // 2
            face_x1 = face_x0 + face_size_px
            face_y1 = face_y0 + face_size_px

            cut_thickness = max(2, thickness - 1)
            _draw_dashed_rectangle(
                annotated,
                (face_x0, face_y0),
                (face_x1, face_y1),
                (0, 0, 0),
                thickness=cut_thickness,
                dash_length=max(14, face_size_px // 8),
                gap_length=max(7, face_size_px // 16),
            )

            cut_label = "CUT"
            cut_size, _ = cv2.getTextSize(cut_label, font, font_scale * 0.45, label_thick)
            cut_x = face_x0 + max(4, (face_size_px - cut_size[0]) // 2)
            cut_y = max(cut_size[1] + 4, face_y0 - 6)
            cv2.putText(
                annotated,
                cut_label,
                (cut_x, cut_y),
                font,
                font_scale * 0.45,
                (0, 0, 0),
                label_thick,
            )

        return annotated

    def generate_marker_images(self, pixels_per_meter: int = 4000) -> dict[int, NDArray]:
        """Generate marker images for all markers in export order."""
        return {
            marker_id: self.generate_marker_image(marker_id, pixels_per_meter=pixels_per_meter)
            for marker_id in self.ordered_marker_ids
        }

    def generate_cube_face_sheet_image(self, face_name: str, pixels_per_meter: int = 4000) -> NDArray[np.uint8]:
        """Generate one printable PNG image for a single cube face.

        The image includes the face boundary, face name, marker ID range, and
        every marker in that face's ``N x N`` grid using the same derived gap as
        the 3D geometry. This is the export path used for cube construction.
        """
        if not self.is_cube or self.marker_order is None:
            raise ValueError("Cube face sheets are only available for cube targets")
        if self.cube_edge_length_m is None or self.cube_markers_per_side is None:
            raise ValueError("Cube face sheets require cube_edge_length_m and markers_per_side")

        marker_ids = self.get_cube_face_marker_ids(face_name)
        face_size_px = max(1, int(round(self.cube_edge_length_m * pixels_per_meter)))
        marker_size_px = max(1, int(round(self.marker_size_m * pixels_per_meter)))
        border_px = max(20, face_size_px // 8, marker_size_px // 4)
        canvas = np.full((face_size_px + 2 * border_px, face_size_px + 2 * border_px, 3), 255, dtype=np.uint8)

        face_x0 = border_px
        face_y0 = border_px
        face_x1 = face_x0 + face_size_px
        face_y1 = face_y0 + face_size_px

        _draw_dashed_rectangle(
            canvas,
            (face_x0, face_y0),
            (face_x1, face_y1),
            (0, 0, 0),
            thickness=max(1, face_size_px // 150),
            dash_length=max(10, face_size_px // 8),
            gap_length=max(5, face_size_px // 16),
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, face_size_px / 700.0)
        thickness = max(1, face_size_px // 180)
        first_marker = marker_ids[0]
        last_marker = marker_ids[-1]
        id_text = f"ID {first_marker}" if first_marker == last_marker else f"IDs {first_marker}-{last_marker}"
        label = f"{face_name.upper()}  {id_text}"
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(4, (canvas.shape[1] - text_size[0]) // 2)
        text_y = max(text_size[1] + 4, border_px // 2 + text_size[1] // 2)
        cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

        gap_px = (face_size_px - self.cube_markers_per_side * marker_size_px) / (self.cube_markers_per_side + 1)
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)

        for marker_id in marker_ids:
            row = self.get_cube_face_row(marker_id)
            col = self.get_cube_face_col(marker_id)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
            if self.inverted:
                marker_img = cv2.bitwise_not(marker_img)

            x0 = int(round(face_x0 + gap_px + col * (marker_size_px + gap_px)))
            y0 = int(round(face_y0 + gap_px + row * (marker_size_px + gap_px)))
            x0 = min(max(face_x0, x0), max(face_x0, face_x1 - marker_size_px))
            y0 = min(max(face_y0, y0), max(face_y0, face_y1 - marker_size_px))
            canvas[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = cv2.cvtColor(
                marker_img,
                cv2.COLOR_GRAY2BGR,
            )

        return canvas

    def generate_cube_face_sheet_images(self, pixels_per_meter: int = 4000) -> dict[str, NDArray[np.uint8]]:
        """Generate printable face sheets for all six cube faces."""
        if not self.is_cube:
            raise ValueError("Cube face sheets are only available for cube targets")
        return {
            face_name: self.generate_cube_face_sheet_image(face_name, pixels_per_meter=pixels_per_meter)
            for face_name in _CUBE_FACE_NAMES
        }

    def generate_cube_layout_image(self, pixels_per_meter: int = 4000) -> NDArray:
        """Generate a printable cube-net overview from the six face sheets."""
        if not self.is_cube or self.marker_order is None:
            raise ValueError("Cube layout image is only available for cube targets")
        if self.cube_edge_length_m is None:
            raise ValueError("Cube layout image requires cube_edge_length_m")

        face_images = self.generate_cube_face_sheet_images(pixels_per_meter=pixels_per_meter)
        tile_height = max(image.shape[0] for image in face_images.values())
        tile_width = max(image.shape[1] for image in face_images.values())
        gap = max(12, min(tile_width, tile_height) // 20)
        rows, cols = 3, 4

        canvas_height = gap + rows * tile_height + rows * gap
        canvas_width = gap + cols * tile_width + cols * gap
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

        for face_name, image in face_images.items():
            row, col = _CUBE_NET_POSITIONS[face_name]
            x0 = gap + col * (tile_width + gap)
            y0 = gap + row * (tile_height + gap)
            canvas[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image

        return canvas
