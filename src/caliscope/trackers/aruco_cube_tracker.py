"""Dedicated tracker for rigid ArUco cube targets.

This tracker keeps the same corner-level PointPacket output as ``ArucoTracker``,
but also owns a cube-specific board model and pose helpers that mirror the
"detect markers + estimate cube pose" workflow used in standalone scripts.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from caliscope.core.aruco_target import ArucoTarget
from caliscope.packets import PointPacket
from caliscope.trackers.aruco_tracker import ArucoTracker

logger = logging.getLogger(__name__)


def _object_points_are_planar(obj_points: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True when a detected cube observation lies on one plane.

    Cube observations can be planar when every visible marker belongs to the
    same physical face, or non-planar when multiple faces are visible. The pose
    solver uses this distinction to pick an OpenCV PnP method.
    """
    if len(obj_points) < 4:
        return True
    centered = np.asarray(obj_points, dtype=np.float64) - np.mean(obj_points, axis=0, keepdims=True)
    return np.linalg.matrix_rank(centered, tol=atol) <= 2


class ArucoCubeTracker(ArucoTracker):
    """Tracker specialized for rigid six-face ArUco cube targets.

    The tracker returns the same corner-level ``PointPacket`` shape as the
    plain ArUco tracker, but it also builds a ``cv2.aruco.Board`` containing
    every cube marker and provides cube-specific pose-solving helpers.
    """

    _PERFECT_CAMERA = np.identity(3, dtype=np.float64)
    _ZERO_DISTORTION = np.zeros(5, dtype=np.float64)

    def __init__(self, aruco_target: ArucoTarget):
        """Create a tracker for one configured cube target.

        ``aruco_target`` must already contain all cube marker corner positions
        in object coordinates. The OpenCV board is built from the target's
        ordered marker IDs so 1x1 and NxN cubes use the same downstream path.
        """
        if not aruco_target.is_cube:
            raise ValueError("ArucoCubeTracker requires a cube ArUco target")

        super().__init__(
            dictionary=aruco_target.dictionary,
            inverted=aruco_target.inverted,
            mirror_flag_search=False,
            aruco_target=aruco_target,
        )
        self.board = cv2.aruco.Board(
            [
                np.asarray(aruco_target.corners[marker_id], dtype=np.float32)
                for marker_id in aruco_target.ordered_marker_ids
            ],
            self.dictionary_object,
            np.asarray(aruco_target.ordered_marker_ids, dtype=np.int32).reshape(-1, 1),
        )

    @property
    def name(self) -> str:
        """Return the storage and UI tracker name used for cube calibration."""
        return "ARUCO_CUBE"

    def _detect_marker_groups(
        self,
        gray_frame: np.ndarray,
    ) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        """Detect only markers that belong to the configured cube target.

        OpenCV returns one corner array per detected marker. This method keeps
        that grouped structure because ``cv2.aruco.Board.matchImagePoints``
        expects marker-level corners and IDs rather than a flattened point list.
        """
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        if ids is None or len(ids) == 0:
            return None, None

        if self.aruco_target is not None:
            tracked_ids = set(int(marker_id) for marker_id in self.aruco_target.marker_ids)
            filtered = [
                (corner, int(marker_id))
                for corner, marker_id in zip(corners, ids.flatten(), strict=True)
                if int(marker_id) in tracked_ids
            ]
            if not filtered:
                return None, None
            corners = [corner for corner, _ in filtered]
            ids = np.asarray([[marker_id] for _, marker_id in filtered], dtype=np.int32)

        return corners, ids

    def detect_cube_markers(
        self,
        frame: np.ndarray,
    ) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        """Detect cube markers in a BGR frame and return OpenCV grouped output."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.inverted:
            gray_frame = cv2.bitwise_not(gray_frame)

        corners, ids = self._detect_marker_groups(gray_frame)
        return corners, ids

    def get_points(self, frame: np.ndarray, cam_id: int = 0, rotation_count: int = 0) -> PointPacket:
        """Return cube marker corners as Caliscope point observations.

        Point IDs keep the existing ArUco convention ``marker_id * 10 +
        corner_index``. ``obj_loc`` is populated from the known cube target
        geometry so extrinsic calibration receives 2D observations and matching
        3D object-frame coordinates.
        """
        corners, ids = self.detect_cube_markers(frame)
        if corners is None or ids is None:
            return PointPacket(
                point_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float32),
                obj_loc=np.empty((0, 3), dtype=np.float32),
            )

        all_corners = np.vstack(corners).reshape(-1, 2)
        point_ids: list[int] = []
        for marker_id in ids.flatten():
            base_id = int(marker_id) * 10
            point_ids.extend([base_id + j for j in range(4)])

        point_id_array = np.asarray(point_ids, dtype=np.int32)
        point_id_array, all_corners, obj_loc = self._apply_target_filter(point_id_array, all_corners)
        return PointPacket(point_id=point_id_array, img_loc=all_corners, obj_loc=obj_loc)

    def estimate_pose(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Estimate cube pose directly from a camera frame.

        This is a convenience/debug path. The calibration pipeline usually
        consumes ``PointPacket`` observations, while this method mirrors a
        standalone ``detectMarkers`` plus ``solvePnP`` workflow.
        """
        corners, ids = self.detect_cube_markers(frame)
        if corners is None or ids is None:
            return False, None, None

        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
        point_ids = []
        for marker_id in ids.flatten():
            point_ids.extend([int(marker_id) * 10 + corner_index for corner_index in range(4)])

        return self._solve_pose_from_correspondences(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            point_ids=np.asarray(point_ids, dtype=np.int32),
        )

    def estimate_pose_from_packet(
        self,
        point_packet: PointPacket,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Estimate cube pose from an existing ``PointPacket``."""
        if point_packet.obj_loc is None or len(point_packet.point_id) < 4:
            return False, None, None

        return self._solve_pose_from_correspondences(
            np.asarray(point_packet.obj_loc, dtype=np.float32),
            np.asarray(point_packet.img_loc, dtype=np.float32),
            camera_matrix,
            dist_coeffs,
            point_ids=np.asarray(point_packet.point_id, dtype=np.int32),
        )

    def draw_pose_axes(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        axis_length: float | None = None,
    ) -> np.ndarray:
        """Draw object-frame axes for a solved cube pose on top of a frame."""
        axis_size = self.aruco_target.marker_size_m if axis_length is None else axis_length
        annotated = frame.copy()
        axis_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_size, 0.0, 0.0],
                [0.0, axis_size, 0.0],
                [0.0, 0.0, axis_size],
            ],
            dtype=np.float64,
        )
        projected = self._project_points(axis_points, camera_matrix, dist_coeffs, rvec, tvec)
        if projected is None or projected.shape[0] != 4:
            return annotated

        origin = self._as_pixel(projected[0])
        if origin is None:
            return annotated

        for endpoint, color in zip(projected[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
            pixel = self._as_pixel(endpoint)
            if pixel is None:
                continue
            cv2.line(annotated, origin, pixel, color, 3, lineType=cv2.LINE_AA)
        return annotated

    def project_target_from_pose(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> PointPacket:
        """Project every cube marker corner from a solved single-view pose.

        The preview/export overlay paths use this to draw the full target, not
        just the markers that happened to be detected in the current frame.
        """
        point_ids: list[int] = []
        obj_points: list[np.ndarray] = []
        for marker_id in self.aruco_target.ordered_marker_ids:
            marker_corners = np.asarray(self.aruco_target.corners[marker_id], dtype=np.float32)
            for corner_index, corner in enumerate(marker_corners):
                point_ids.append(marker_id * 10 + corner_index)
                obj_points.append(corner)

        if not obj_points:
            return PointPacket(
                point_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float32),
                obj_loc=np.empty((0, 3), dtype=np.float32),
            )

        obj_loc = np.asarray(obj_points, dtype=np.float32)
        projected = self._project_points(obj_loc, camera_matrix, dist_coeffs, rvec, tvec)
        if projected is None:
            return PointPacket(
                point_id=np.array([], dtype=np.int32),
                img_loc=np.empty((0, 2), dtype=np.float32),
                obj_loc=obj_loc,
            )

        finite_mask = np.isfinite(projected).all(axis=1)
        return PointPacket(
            point_id=np.asarray(point_ids, dtype=np.int32)[finite_mask],
            img_loc=projected[finite_mask].astype(np.float32),
            obj_loc=obj_loc[finite_mask],
        )

    def _solve_pose_from_correspondences(
        self,
        obj_points: np.ndarray,
        img_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        point_ids: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Solve a cube object-to-camera pose from 2D/3D correspondences.

        The solver branches by geometry:
        one detected square marker uses ``IPPE_SQUARE``; multiple markers on
        one physical face use planar ``IPPE``; multi-face observations use a
        non-planar PnP method. Every candidate is evaluated by reprojection
        error and by whether the observed face normals point toward the camera.
        """
        if len(obj_points) < 4:
            return False, None, None

        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
        undistorted_img_points = self._undistort_points(img_points, camera_matrix, dist_coeffs)

        is_planar = _object_points_are_planar(obj_points)
        observed_marker_ids = (
            sorted({int(point_id) // 10 for point_id in point_ids})
            if point_ids is not None and len(point_ids) == len(obj_points)
            else []
        )
        observed_face_names = {
            self.aruco_target.get_cube_face_name(marker_id)
            for marker_id in observed_marker_ids
            if marker_id in self.aruco_target.corners
        }
        is_single_marker_square = len(observed_marker_ids) == 1 and len(obj_points) == 4
        is_single_face = len(observed_face_names) == 1 and len(observed_marker_ids) > 0

        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        if is_single_marker_square:
            solve_obj_points, solve_img_points = self._sort_single_face_correspondences(
                obj_points,
                undistorted_img_points,
                point_ids,
            )
            try:
                success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    solve_obj_points,
                    solve_img_points,
                    self._PERFECT_CAMERA,
                    self._ZERO_DISTORTION,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except cv2.error as exc:
                logger.debug("Skipping IPPE_SQUARE candidate for single-marker cube observation: %s", exc)
                success, rvecs, tvecs = False, (), ()
            if success:
                candidates.extend((np.asarray(rvec), np.asarray(tvec)) for rvec, tvec in zip(rvecs, tvecs, strict=True))
        elif is_planar and is_single_face:
            try:
                success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_points,
                    undistorted_img_points,
                    self._PERFECT_CAMERA,
                    self._ZERO_DISTORTION,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error as exc:
                logger.debug("Skipping IPPE candidate for planar cube observation: %s", exc)
                success, rvecs, tvecs = False, (), ()
            if success:
                candidates.extend((np.asarray(rvec), np.asarray(tvec)) for rvec, tvec in zip(rvecs, tvecs, strict=True))
        else:
            flags = cv2.SOLVEPNP_ITERATIVE if len(obj_points) >= 6 else cv2.SOLVEPNP_SQPNP
            try:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    undistorted_img_points,
                    self._PERFECT_CAMERA,
                    self._ZERO_DISTORTION,
                    flags=flags,
                )
            except cv2.error as exc:
                logger.debug("Cube solvePnP failed for non-planar observation with %s points: %s", len(obj_points), exc)
                success, rvec, tvec = False, None, None
            if success and rvec is not None and tvec is not None:
                candidates.append((np.asarray(rvec), np.asarray(tvec)))

        if not candidates:
            return False, None, None

        best_valid_result: tuple[float, np.ndarray, np.ndarray] | None = None
        best_any_result: tuple[float, np.ndarray, np.ndarray] | None = None
        for rvec, tvec in candidates:
            refined = self._refine_pose_candidate(obj_points, undistorted_img_points, rvec, tvec)
            eval_rvec, eval_tvec = refined if refined is not None else (rvec, tvec)
            rmse, corrected_rvec, corrected_tvec, faces_camera = self._evaluate_pose_candidate(
                obj_points,
                undistorted_img_points,
                eval_rvec,
                eval_tvec,
                point_ids=point_ids,
            )
            if best_any_result is None or rmse < best_any_result[0]:
                best_any_result = (rmse, corrected_rvec, corrected_tvec)
            if faces_camera and (best_valid_result is None or rmse < best_valid_result[0]):
                best_valid_result = (rmse, corrected_rvec, corrected_tvec)

        best_result = best_valid_result or best_any_result
        if best_result is None:
            return False, None, None
        return True, best_result[1], best_result[2]

    def _evaluate_pose_candidate(
        self,
        obj_points: np.ndarray,
        undistorted_img_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        point_ids: np.ndarray | None,
    ) -> tuple[float, np.ndarray, np.ndarray, bool]:
        """Score one PnP pose candidate and validate cube face orientation."""
        rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)

        corrected_rvec, _ = cv2.Rodrigues(rotation_matrix)
        corrected_tvec = translation.reshape(3, 1)
        projected, _ = cv2.projectPoints(
            np.asarray(obj_points, dtype=np.float32),
            corrected_rvec,
            corrected_tvec,
            self._PERFECT_CAMERA,
            self._ZERO_DISTORTION,
        )
        residuals = undistorted_img_points - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        faces_camera = self._candidate_faces_camera(
            rotation_matrix,
            translation,
            point_ids,
        )
        return rmse, corrected_rvec, corrected_tvec, faces_camera

    def _refine_pose_candidate(
        self,
        obj_points: np.ndarray,
        undistorted_img_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Optionally refine a solved pose candidate with Levenberg-Marquardt."""
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                np.asarray(obj_points, dtype=np.float32),
                np.asarray(undistorted_img_points, dtype=np.float32),
                self._PERFECT_CAMERA,
                self._ZERO_DISTORTION,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            )
        except cv2.error:
            return None

        return np.asarray(refined_rvec), np.asarray(refined_tvec)

    @staticmethod
    def _sort_single_face_correspondences(
        obj_points: np.ndarray,
        img_points: np.ndarray,
        point_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order one-marker correspondences by corner index for ``IPPE_SQUARE``."""
        if point_ids is None or len(point_ids) != len(obj_points):
            return obj_points, img_points

        order = np.argsort(np.asarray(point_ids, dtype=np.int32) % 10)
        return obj_points[order], img_points[order]

    def _candidate_faces_camera(
        self,
        rotation_matrix: np.ndarray,
        translation: np.ndarray,
        point_ids: np.ndarray | None,
    ) -> bool:
        """Return True when observed cube faces are visible from the camera.

        PnP can produce mirror-symmetric solutions for planar observations. The
        cube face normal check rejects candidates where a detected face would
        be pointing away from the camera.
        """
        if point_ids is None or len(point_ids) == 0:
            return True

        observed_marker_ids = sorted({int(point_id) // 10 for point_id in point_ids})
        for marker_id in observed_marker_ids:
            marker_corners = np.asarray(self.aruco_target.corners.get(marker_id), dtype=np.float64)
            if marker_corners.shape != (4, 3):
                continue

            face_center = marker_corners.mean(axis=0)
            x_axis = marker_corners[1] - marker_corners[0]
            y_axis = marker_corners[0] - marker_corners[3]
            normal = np.cross(x_axis, y_axis)
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-8:
                continue
            normal /= normal_norm

            face_center_cam = rotation_matrix @ face_center + translation
            normal_cam = rotation_matrix @ normal
            camera_side_score = float(np.dot(normal_cam, -face_center_cam))
            if camera_side_score <= 0.0:
                return False

        return True

    @staticmethod
    def _undistort_points(
        img_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Convert image pixels to normalized coordinates for PnP.

        Four distortion coefficients are treated as OpenCV fisheye parameters;
        all other distortion vectors use the standard pinhole model.
        """
        matrix64 = np.asarray(camera_matrix, dtype=np.float64)
        distortions = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        reshaped = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)

        if distortions.size == 4:
            undistorted = cv2.fisheye.undistortPoints(
                reshaped,
                matrix64,
                distortions.reshape(-1, 1),
                P=ArucoCubeTracker._PERFECT_CAMERA,
            )
        else:
            undistorted = cv2.undistortPoints(
                reshaped,
                matrix64,
                distortions,
                P=ArucoCubeTracker._PERFECT_CAMERA,
            )

        return undistorted.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _project_points(
        obj_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray | None:
        """Project object-frame cube points into image pixels."""
        matrix64 = np.asarray(camera_matrix, dtype=np.float64)
        distortions = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        object_points = np.asarray(obj_points, dtype=np.float64).reshape(-1, 1, 3)
        rvec64 = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec64 = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        try:
            if distortions.size == 4:
                projected, _ = cv2.fisheye.projectPoints(
                    object_points,
                    rvec64,
                    tvec64,
                    matrix64,
                    distortions.reshape(-1, 1),
                )
            else:
                projected, _ = cv2.projectPoints(
                    object_points,
                    rvec64,
                    tvec64,
                    matrix64,
                    distortions,
                )
        except cv2.error:
            return None

        return projected.reshape(-1, 2)

    @staticmethod
    def _as_pixel(point: np.ndarray) -> tuple[int, int] | None:
        """Convert a finite floating-point image coordinate into an integer pixel."""
        point64 = np.asarray(point, dtype=np.float64).reshape(-1)
        if point64.size != 2 or not np.isfinite(point64).all():
            return None
        return int(round(float(point64[0]))), int(round(float(point64[1])))
