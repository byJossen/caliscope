"""ArUco target configuration panel for extrinsic calibration.

Allows user to configure either a single printed marker or a six-face
marker cube. Physical dimensions set the world scale gauge.
"""

import logging

import cv2
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from caliscope.core.aruco_target import ArucoTarget
from caliscope.gui.theme import Colors, Typography
from caliscope.gui.utils.spinbox_utils import setup_spinbox_sizing

logger = logging.getLogger(__name__)


# Dictionary options: display name -> cv2 constant value
ARUCO_DICTIONARIES = [
    ("4x4 (50 markers)", cv2.aruco.DICT_4X4_50),
    ("4x4 (100 markers)", cv2.aruco.DICT_4X4_100),
    ("4x4 (250 markers)", cv2.aruco.DICT_4X4_250),
    ("5x5 (50 markers)", cv2.aruco.DICT_5X5_50),
    ("5x5 (100 markers)", cv2.aruco.DICT_5X5_100),
]


class ArucoTargetConfigPanel(QWidget):
    """ArUco target configuration for extrinsic calibration.

    Layout:
    - Row 1: Layout mode
    - Row 2: Dictionary
    - Row 3: Marker ID / first marker ID
    - Row 4: Invert checkbox
    - Row 5: Marker Size
    - Row 6: Markers Per Side (cube mode only)
    - Row 7: Cube edge size (cube mode only)
    - Helper text explaining scale gauge, ID ordering, and spacing

    Emits `config_changed` whenever any value changes.
    Use `get_aruco_target()` to build an ArucoTarget from current values.
    """

    config_changed = Signal()

    def __init__(
        self,
        initial_target: ArucoTarget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._setup_ui(initial_target)

    def _setup_ui(self, initial_target: ArucoTarget) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Row 1: Layout mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Layout:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Single Marker", "single_marker")
        self._mode_combo.addItem("Cube (NxN per face)", "cube")
        self._mode_combo.setCurrentIndex(1 if initial_target.is_cube else 0)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Row 2: Dictionary
        dict_row = QHBoxLayout()
        dict_row.addWidget(QLabel("Dictionary:"))
        self._dict_combo = QComboBox()
        for display_name, value in ARUCO_DICTIONARIES:
            self._dict_combo.addItem(display_name, value)
        # Set initial selection
        for i, (_, value) in enumerate(ARUCO_DICTIONARIES):
            if value == initial_target.dictionary:
                self._dict_combo.setCurrentIndex(i)
                break
        dict_row.addWidget(self._dict_combo)
        dict_row.addStretch()
        layout.addLayout(dict_row)

        # Row 3: Marker ID
        id_row = QHBoxLayout()
        self._id_label = QLabel()
        id_row.addWidget(self._id_label)
        self._id_spin = QSpinBox()
        self._id_spin.setMinimum(0)
        initial_id = initial_target.ordered_marker_ids[0] if initial_target.ordered_marker_ids else 0
        self._id_spin.setValue(initial_id)
        setup_spinbox_sizing(self._id_spin, min_value=0, max_value=249)
        id_row.addWidget(self._id_spin)
        id_row.addStretch()
        layout.addLayout(id_row)

        # Row 4: Invert checkbox
        invert_row = QHBoxLayout()
        self._invert_checkbox = QCheckBox("&Invert")
        self._invert_checkbox.setChecked(initial_target.inverted)
        invert_row.addWidget(self._invert_checkbox)
        invert_row.addStretch()
        layout.addLayout(invert_row)

        # Row 5: Marker Size (in cm, domain uses meters)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Marker Size:"))
        self._size_spin = QDoubleSpinBox()
        self._size_spin.setDecimals(1)
        self._size_spin.setSingleStep(0.5)
        self._size_spin.setMinimum(0.5)
        self._size_spin.setMaximum(100.0)
        self._size_spin.setSuffix(" cm")
        # Convert meters to cm for display
        initial_cm = initial_target.marker_size_m * 100
        self._size_spin.setValue(initial_cm)
        setup_spinbox_sizing(self._size_spin, min_value=0.5, max_value=100.0)
        size_row.addWidget(self._size_spin)
        size_row.addStretch()
        layout.addLayout(size_row)

        # Row 6: Markers per side
        self._cube_grid_row = QWidget()
        cube_grid_row_layout = QHBoxLayout(self._cube_grid_row)
        cube_grid_row_layout.setContentsMargins(0, 0, 0, 0)
        cube_grid_row_layout.addWidget(QLabel("Markers Per Side:"))
        self._markers_per_side_spin = QSpinBox()
        self._markers_per_side_spin.setMinimum(1)
        self._markers_per_side_spin.setMaximum(999)
        self._markers_per_side_spin.setValue(self._get_initial_markers_per_side(initial_target))
        setup_spinbox_sizing(self._markers_per_side_spin, min_value=1, max_value=10)
        cube_grid_row_layout.addWidget(self._markers_per_side_spin)
        cube_grid_row_layout.addStretch()
        layout.addWidget(self._cube_grid_row)

        # Row 7: Cube edge size
        self._cube_row = QWidget()
        cube_row_layout = QHBoxLayout(self._cube_row)
        cube_row_layout.setContentsMargins(0, 0, 0, 0)
        cube_row_layout.addWidget(QLabel("Cube Edge:"))
        self._cube_size_spin = QDoubleSpinBox()
        self._cube_size_spin.setDecimals(1)
        self._cube_size_spin.setSingleStep(0.5)
        self._cube_size_spin.setMinimum(1.0)
        self._cube_size_spin.setMaximum(200.0)
        self._cube_size_spin.setSuffix(" cm")
        initial_cube_edge_cm = self._get_initial_cube_edge_cm(initial_target)
        self._cube_size_spin.setValue(initial_cube_edge_cm)
        setup_spinbox_sizing(self._cube_size_spin, min_value=1.0, max_value=200.0)
        cube_row_layout.addWidget(self._cube_size_spin)
        cube_row_layout.addStretch()
        layout.addWidget(self._cube_row)

        # Helper text
        self._helper = QLabel()
        self._helper.setWordWrap(True)
        layout.addWidget(self._helper)

        layout.addStretch()

        # Connect signals
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._dict_combo.currentIndexChanged.connect(self._on_config_changed)
        self._id_spin.valueChanged.connect(self._on_config_changed)
        self._invert_checkbox.stateChanged.connect(self._on_config_changed)
        self._size_spin.valueChanged.connect(self._on_config_changed)
        self._markers_per_side_spin.valueChanged.connect(self._on_config_changed)
        self._cube_size_spin.valueChanged.connect(self._on_config_changed)

        self._update_dictionary_limits()
        self._update_mode_ui()

    def _get_initial_cube_edge_cm(self, target: ArucoTarget) -> float:
        """Best-effort initial cube edge size for existing targets."""
        if target.cube_edge_length_m is not None:
            return target.cube_edge_length_m * 100.0

        if target.is_cube and target.corners:
            centers = np.array([positions.mean(axis=0) for positions in target.corners.values()], dtype=np.float64)
            edge_length_m = float(np.max(np.ptp(centers, axis=0)))
            if edge_length_m > 0:
                return edge_length_m * 100.0

        return 12.0

    def _get_initial_markers_per_side(self, target: ArucoTarget) -> int:
        """Best-effort initial cube grid density for existing targets."""
        if target.is_cube and target.cube_markers_per_side is not None:
            return target.cube_markers_per_side

        if target.is_cube and target.ordered_marker_ids:
            per_face_count = len(target.ordered_marker_ids) / 6.0
            inferred = int(round(np.sqrt(per_face_count)))
            if inferred > 0 and inferred * inferred * 6 == len(target.ordered_marker_ids):
                return inferred

        return 1

    def _selected_dictionary_max_marker_id(self) -> int:
        """Highest valid marker ID for the selected dictionary."""
        dictionary = self._dict_combo.currentData()
        return {
            cv2.aruco.DICT_4X4_50: 49,
            cv2.aruco.DICT_4X4_100: 99,
            cv2.aruco.DICT_4X4_250: 249,
            cv2.aruco.DICT_5X5_50: 49,
            cv2.aruco.DICT_5X5_100: 99,
        }.get(dictionary, 249)

    def _cube_total_markers(self) -> int:
        """Total markers required by the current cube UI state."""
        markers_per_side = self._markers_per_side_spin.value()
        return 6 * markers_per_side * markers_per_side

    def validation_error(self) -> str | None:
        """Hard validation error for the current panel state, if any."""
        if self._mode_combo.currentData() != "cube":
            return None

        markers_per_side = self._markers_per_side_spin.value()
        if markers_per_side < 1:
            return "Markers Per Side must be at least 1."

        marker_size_m = self._size_spin.value() / 100.0
        cube_edge_length_m = self._cube_size_spin.value() / 100.0
        if markers_per_side * marker_size_m >= cube_edge_length_m:
            return "Markers Per Side x Marker Size must be smaller than the cube edge."

        total_markers = self._cube_total_markers()
        dictionary_size = self._selected_dictionary_max_marker_id() + 1
        if total_markers > dictionary_size:
            return f"Selected dictionary only has {dictionary_size} marker IDs, but this cube needs {total_markers}."

        max_first_marker_id = self._selected_dictionary_max_marker_id() - total_markers + 1
        if self._id_spin.value() > max_first_marker_id:
            return f"First Marker ID must be {max_first_marker_id} or lower for this dictionary."

        return None

    def _spacing_warning(self) -> str | None:
        """Soft warning for dense cube faces that may print or detect poorly."""
        if self._mode_combo.currentData() != "cube":
            return None
        if self.validation_error() is not None:
            return None

        markers_per_side = self._markers_per_side_spin.value()
        marker_size_m = self._size_spin.value() / 100.0
        cube_edge_length_m = self._cube_size_spin.value() / 100.0
        gap_m = ArucoTarget.compute_cube_face_gap_m(
            markers_per_side=markers_per_side,
            marker_size_m=marker_size_m,
            edge_length_m=cube_edge_length_m,
        )
        if gap_m / marker_size_m < 0.15:
            return (
                f"Warning: gap is only {gap_m * 1000.0:.1f} mm "
                f"({gap_m / marker_size_m * 100.0:.0f}% of marker size); printing and detection may degrade."
            )
        return None

    def _update_dictionary_limits(self) -> None:
        """Keep the marker ID spinbox within the selected dictionary range."""
        max_marker_id = self._selected_dictionary_max_marker_id()
        if self._mode_combo.currentData() == "cube":
            max_marker_id = max(0, max_marker_id - self._cube_total_markers() + 1)
        self._id_spin.setMaximum(max_marker_id)

    def _update_helper_text(self) -> None:
        """Update helper copy and warning state for the current configuration."""
        if self._mode_combo.currentData() != "cube":
            self._helper.setStyleSheet(Typography.HELPER_TEXT)
            self._helper.setText("(Physical size sets the world scale gauge)")
            return

        total_markers = self._cube_total_markers()
        error = self.validation_error()
        if error is not None:
            self._helper.setStyleSheet(f"color: {Colors.ERROR};")
            self._helper.setText(error)
            return

        marker_size_m = self._size_spin.value() / 100.0
        cube_edge_length_m = self._cube_size_spin.value() / 100.0
        gap_m = ArucoTarget.compute_cube_face_gap_m(
            markers_per_side=self._markers_per_side_spin.value(),
            marker_size_m=marker_size_m,
            edge_length_m=cube_edge_length_m,
        )
        helper_lines = [
            (
                f"(Uses {total_markers} consecutive IDs in face-major, row-major order. "
                f"Cube size stays fixed, marker size stays fixed, and the derived gap is {gap_m * 1000.0:.1f} mm.)"
            )
        ]
        warning = self._spacing_warning()
        if warning is not None:
            helper_lines.append(warning)
            self._helper.setStyleSheet(f"color: {Colors.WARNING}; font-style: italic;")
        else:
            self._helper.setStyleSheet(Typography.HELPER_TEXT)
        self._helper.setText("\n".join(helper_lines))

    def _update_mode_ui(self) -> None:
        """Show fields and helper text relevant for the current layout mode."""
        is_cube = self._mode_combo.currentData() == "cube"
        self._id_label.setText("First Marker ID:" if is_cube else "Marker ID:")
        self._cube_grid_row.setVisible(is_cube)
        self._cube_row.setVisible(is_cube)
        self._update_helper_text()

    def _on_mode_changed(self) -> None:
        self._update_mode_ui()
        self._on_config_changed()

    def _on_config_changed(self) -> None:
        self._update_dictionary_limits()
        self._update_helper_text()
        self.config_changed.emit()

    def set_layout_mode(self, layout: str, *, lock: bool = False) -> None:
        """Set the target layout mode programmatically.

        Args:
            layout: Either ``single_marker`` or ``cube``.
            lock: If True, disables the layout selector so the outer workflow owns the choice.
        """
        target_index = self._mode_combo.findData(layout)
        if target_index >= 0 and self._mode_combo.currentIndex() != target_index:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(target_index)
            self._mode_combo.blockSignals(False)
            self._update_mode_ui()
            self._update_dictionary_limits()

        self._mode_combo.setEnabled(not lock)

    def get_aruco_target(self) -> ArucoTarget:
        """Build ArucoTarget from current widget values."""
        dictionary = self._dict_combo.currentData()
        marker_size_m = self._size_spin.value() / 100.0  # cm -> meters
        marker_id = self._id_spin.value()
        inverted = self._invert_checkbox.isChecked()

        if self._mode_combo.currentData() == "cube":
            error = self.validation_error()
            if error is not None:
                raise ValueError(error)
            cube_edge_length_m = self._cube_size_spin.value() / 100.0
            return ArucoTarget.cube(
                first_marker_id=marker_id,
                marker_size_m=marker_size_m,
                edge_length_m=cube_edge_length_m,
                dictionary=dictionary,
                inverted=inverted,
                markers_per_side=self._markers_per_side_spin.value(),
            )

        return ArucoTarget.single_marker(
            marker_id=marker_id,
            marker_size_m=marker_size_m,
            dictionary=dictionary,
            inverted=inverted,
        )
