from __future__ import annotations
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal
import logging

from caliscope.cameras.camera_array import CameraArray
from caliscope.core.point_data import WORLD_POINT_COLUMNS, ImagePoints, WorldPoints
from caliscope.core.reprojection import (
    ErrorsXY,
    reprojection_errors,
    bundle_residuals,
    project_world_to_image,
    ImageCoords,
    WorldCoords,
    CameraIndices,
)
from caliscope.core.reprojection_report import ReprojectionReport
from caliscope.core.alignment import (
    estimate_similarity_transform,
    apply_similarity_transform,
    SimilarityTransform,
)
from caliscope.core.scale_accuracy import (
    compute_frame_scale_error,
    FrameScaleError,
    VolumetricScaleReport,
)

import pandas as pd
from caliscope.core.bootstrap_pose.pose_network_builder import (
    _canonicalize_planar_object_points,
    _object_points_are_planar,
    _object_points_use_canonical_board_plane,
)

logger = logging.getLogger(__file__)

OptimizationProgressCallback = Callable[[int, float], None]
StageProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RigidPoseEstimate:
    """One camera's cube/rigid-target pose estimate for one synchronized frame.

    The pose maps target object coordinates into the current world frame after
    the bootstrap camera network has provided approximate camera extrinsics.
    These candidates are later filtered, fused, and used to initialize the
    rigid-target bundle adjustment.
    """

    cam_id: int
    rmse: float
    weight: float
    rotation_obj_to_world: NDArray[np.float64]
    translation_obj_to_world: NDArray[np.float64]
    frame_time: float


@dataclass(frozen=True)
class MarkerFaceGeometry:
    """Per-marker face metadata derived from rigid object coordinates.

    ``face_key`` is a quantized identity for a physical cube face. It lets the
    pose solver tell whether several observed markers are on one face without
    relying on a fixed one-marker-per-face assumption.
    """

    face_center: NDArray[np.float64]
    normal: NDArray[np.float64]
    face_key: tuple[int, ...]


def _coerce_marker_face_geometry(
    face_geometry: MarkerFaceGeometry | tuple[NDArray[np.float64], NDArray[np.float64]],
) -> MarkerFaceGeometry | None:
    """Normalize legacy tuple face metadata into ``MarkerFaceGeometry``."""
    if isinstance(face_geometry, MarkerFaceGeometry):
        return face_geometry

    face_center, normal = face_geometry
    face_center_arr = np.asarray(face_center, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)
    normal_norm = np.linalg.norm(normal_arr)
    if normal_norm <= 1e-8:
        return None
    normal_unit = normal_arr / normal_norm
    face_key = tuple(
        int(round(value / 1e-6))
        for value in (*normal_unit.tolist(), float(np.dot(normal_unit, face_center_arr)))
    )
    return MarkerFaceGeometry(face_center=face_center_arr, normal=normal_unit, face_key=face_key)


def _average_rotation_matrices(
    rotations: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Average rotation matrices and project the result back onto SO(3)."""
    weighted_sum = np.zeros((3, 3), dtype=np.float64)
    for rotation, weight in zip(rotations, weights, strict=True):
        weighted_sum += float(weight) * rotation

    u, _, vt = np.linalg.svd(weighted_sum)
    averaged = u @ vt
    if np.linalg.det(averaged) < 0:
        u[:, -1] *= -1.0
        averaged = u @ vt
    return averaged


def _rotation_distance_degrees(rotation_a: NDArray[np.float64], rotation_b: NDArray[np.float64]) -> float:
    """Return the angular distance between two rotations in degrees."""
    relative_rotation = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(rotation_b, dtype=np.float64).T
    cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rigid_target_characteristic_scale(point_id_to_obj: dict[int, NDArray[np.float64]]) -> float:
    """Return target diameter used to scale pose-disagreement thresholds."""
    if len(point_id_to_obj) < 2:
        return 0.0

    object_points = np.asarray(list(point_id_to_obj.values()), dtype=np.float64)
    deltas = object_points[:, None, :] - object_points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return float(np.max(distances))


def _estimate_target_pose_world_pose_candidates(
    camera_array: CameraArray,
    image_points: ImagePoints,
    point_id_to_obj: dict[int, NDArray[np.float64]],
    *,
    min_points: int = 4,
) -> dict[int, list[RigidPoseEstimate]]:
    """Estimate one rigid-target world pose candidate per camera/sync group.

    Each camera independently solves object-to-camera pose from the cube points
    visible in that frame, then transforms the result into the bootstrap world
    frame. The output is grouped by sync index because the optimizer ultimately
    needs one fused cube pose per synchronized frame.
    """
    if not point_id_to_obj:
        return {}

    target_df = image_points.df[image_points.df["point_id"].isin(point_id_to_obj.keys())].copy()
    if target_df.empty:
        return {}

    posed_cameras = camera_array.posed_cameras
    K_perfect = np.identity(3)
    D_perfect = np.zeros(5)
    pose_candidates_by_sync: dict[int, list[RigidPoseEstimate]] = {}
    marker_face_geometry = _marker_face_geometry_from_point_ids(point_id_to_obj)

    for (cam_id, sync_index), group in target_df.groupby(["cam_id", "sync_index"]):
        if cam_id not in posed_cameras:
            continue

        camera = posed_cameras[cam_id]
        if camera.rotation is None or camera.translation is None:
            continue

        group = group.drop_duplicates(subset=["point_id"]).copy()
        if len(group) < min_points:
            continue

        img_points = group[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32)
        undistorted = camera.undistort_points(img_points, output="normalized")
        obj_points = np.array([point_id_to_obj[int(pid)] for pid in group["point_id"]], dtype=np.float32)
        point_ids = group["point_id"].to_numpy(dtype=np.int32)

        solved = _solve_target_pose_from_normalized_correspondences(
            obj_points=obj_points,
            undistorted_img_points=undistorted,
            perfect_camera_matrix=K_perfect,
            zero_distortion=D_perfect,
            point_ids=point_ids,
            marker_face_geometry=marker_face_geometry,
        )
        if solved is None:
            continue

        rotation_obj_to_cam, translation_obj_to_cam, rmse = solved
        frame_time = float(group["frame_time"].mean()) if "frame_time" in group.columns else float("nan")
        rotation_obj_to_world = camera.rotation.T @ rotation_obj_to_cam
        translation_obj_to_world = (translation_obj_to_cam - camera.translation) @ camera.rotation
        weight = 1.0 / max(rmse, 1e-6)
        pose_candidates_by_sync.setdefault(int(sync_index), []).append(
            RigidPoseEstimate(
                cam_id=int(cam_id),
                rmse=float(rmse),
                weight=float(weight),
                rotation_obj_to_world=np.asarray(rotation_obj_to_world, dtype=np.float64),
                translation_obj_to_world=np.asarray(translation_obj_to_world, dtype=np.float64),
                frame_time=frame_time,
            )
        )

    return pose_candidates_by_sync


def _filter_pose_estimates_by_cross_camera_agreement(
    estimates_by_sync: dict[int, list[RigidPoseEstimate]],
    point_id_to_obj: dict[int, NDArray[np.float64]],
    *,
    max_translation_fraction: float = 0.45,
    min_translation_m: float = 0.03,
    max_rotation_deg: float = 35.0,
    allow_single_candidate_fallback: bool = True,
) -> dict[int, list[RigidPoseEstimate]]:
    """Keep only sync pose candidates that agree across cameras.

    For each sync we keep the largest mutually consistent subset. If no
    two-camera consensus exists, the default is to keep the strongest single
    camera estimate as the optimizer's initial target pose. The robust
    multi-camera solve still receives all observations from that sync, so we
    avoid discarding many useful frames just because the bootstrap camera
    geometry is not yet accurate enough to make per-camera cube poses agree.
    """
    if not estimates_by_sync:
        return {}

    target_scale = _rigid_target_characteristic_scale(point_id_to_obj)
    translation_threshold = max(float(min_translation_m), float(max_translation_fraction) * target_scale)

    filtered: dict[int, list[RigidPoseEstimate]] = {}
    dropped_syncs = 0
    fallback_syncs = 0
    trimmed_syncs = 0

    for sync_index, estimates in estimates_by_sync.items():
        if len(estimates) < 2:
            filtered[int(sync_index)] = estimates
            continue

        best_subset: list[RigidPoseEstimate] = []
        best_weight = float("-inf")
        best_rmse = float("inf")

        for seed in estimates:
            subset = [
                candidate
                for candidate in estimates
                if np.linalg.norm(candidate.translation_obj_to_world - seed.translation_obj_to_world)
                <= translation_threshold
                and _rotation_distance_degrees(candidate.rotation_obj_to_world, seed.rotation_obj_to_world)
                <= max_rotation_deg
            ]
            subset_weight = float(sum(candidate.weight for candidate in subset))
            subset_rmse = float(np.mean([candidate.rmse for candidate in subset])) if subset else float("inf")
            if (
                len(subset) > len(best_subset)
                or (len(subset) == len(best_subset) and subset_weight > best_weight)
                or (
                    len(subset) == len(best_subset)
                    and np.isclose(subset_weight, best_weight)
                    and subset_rmse < best_rmse
                )
            ):
                best_subset = subset
                best_weight = subset_weight
                best_rmse = subset_rmse

        if len(best_subset) < 2 and allow_single_candidate_fallback:
            filtered[int(sync_index)] = [max(estimates, key=lambda candidate: candidate.weight)]
            fallback_syncs += 1
            continue

        if len(best_subset) < 2:
            dropped_syncs += 1
            continue

        if len(best_subset) < len(estimates):
            trimmed_syncs += 1
        filtered[int(sync_index)] = best_subset

    if dropped_syncs or trimmed_syncs or fallback_syncs:
        logger.info(
            "Rigid-target sync agreement filter kept %s/%s syncs (dropped=%s, trimmed=%s, "
            "single-camera-initialized=%s, translation<=%.3fm, rotation<=%.1fdeg)",
            len(filtered),
            len(estimates_by_sync),
            dropped_syncs,
            trimmed_syncs,
            fallback_syncs,
            translation_threshold,
            max_rotation_deg,
        )

    return filtered


def _max_pairwise_pose_disagreement(
    estimates: list[RigidPoseEstimate],
) -> tuple[float | None, float | None]:
    """Return maximum translation and rotation disagreement within one sync."""
    if len(estimates) < 2:
        return None, None

    max_translation = 0.0
    max_rotation = 0.0
    for idx_a in range(len(estimates)):
        for idx_b in range(idx_a + 1, len(estimates)):
            estimate_a = estimates[idx_a]
            estimate_b = estimates[idx_b]
            max_translation = max(
                max_translation,
                float(np.linalg.norm(estimate_a.translation_obj_to_world - estimate_b.translation_obj_to_world)),
            )
            max_rotation = max(
                max_rotation,
                _rotation_distance_degrees(estimate_a.rotation_obj_to_world, estimate_b.rotation_obj_to_world),
            )
    return max_translation, max_rotation


def _rigid_target_frame_quality_scores(
    image_points: ImagePoints,
    pose_candidates_by_sync: dict[int, list[RigidPoseEstimate]],
    point_id_to_obj: dict[int, NDArray[np.float64]],
) -> dict[int, float]:
    """Score rigid-target syncs before diversity selection.

    Higher scores favor frames with more camera pose candidates, more observed
    marker corners, lower PnP reprojection error, and lower cross-camera pose
    disagreement. The score is only used for ranking the quality pool; final
    frame selection still prioritizes pose diversity inside that pool.
    """
    if not pose_candidates_by_sync:
        return {}

    target_df = image_points.df[image_points.df["point_id"].isin(point_id_to_obj.keys())].copy()
    if target_df.empty:
        return {int(sync_index): float("-inf") for sync_index in pose_candidates_by_sync}

    target_scale = max(_rigid_target_characteristic_scale(point_id_to_obj), 1e-6)
    scores: dict[int, float] = {}
    for sync_index, estimates in pose_candidates_by_sync.items():
        sync_df = target_df[target_df["sync_index"] == sync_index]
        if sync_df.empty:
            scores[int(sync_index)] = float("-inf")
            continue

        total_corner_observations = int(sync_df.drop_duplicates(subset=["cam_id", "point_id"]).shape[0])
        num_pose_candidates = len(estimates)
        mean_rmse = float(np.mean([estimate.rmse for estimate in estimates])) if estimates else float("inf")
        max_translation, max_rotation = _max_pairwise_pose_disagreement(estimates)
        translation_penalty = 0.0 if max_translation is None else max_translation / target_scale
        rotation_penalty = 0.0 if max_rotation is None else max_rotation / 30.0

        scores[int(sync_index)] = (
            100.0 * num_pose_candidates
            + float(total_corner_observations)
            - 1000.0 * mean_rmse
            - 10.0 * translation_penalty
            - 10.0 * rotation_penalty
        )

    return scores


def _pose_diversity_distance(
    pose_a: tuple[NDArray[np.float64], NDArray[np.float64], float],
    pose_b: tuple[NDArray[np.float64], NDArray[np.float64], float],
    *,
    translation_scale_m: float,
    rotation_scale_deg: float,
) -> float:
    """Return normalized pose-space distance between two cube poses.

    Translation distance is scaled by target size, and rotation distance is
    scaled by ``rotation_scale_deg``. The combined Euclidean distance lets the
    diversity selector choose frames that differ in either position or angle.
    """
    rotation_a, translation_a, _ = pose_a
    rotation_b, translation_b, _ = pose_b
    translation_distance = float(np.linalg.norm(translation_a - translation_b)) / max(translation_scale_m, 1e-6)
    rotation_distance = _rotation_distance_degrees(rotation_a, rotation_b) / max(rotation_scale_deg, 1e-6)
    return float(np.sqrt(translation_distance**2 + rotation_distance**2))


def _select_pose_diverse_syncs(
    pose_estimates: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]],
    quality_scores: dict[int, float],
    *,
    max_syncs: int,
    translation_scale_m: float,
    rotation_scale_deg: float = 30.0,
    progress_callback: StageProgressCallback | None = None,
) -> set[int]:
    """Select frames that are far apart in fused cube-pose space.

    The first frame is the highest-quality candidate. Each following frame is
    the candidate whose nearest already-selected pose is farthest away, with
    quality and sync index as deterministic tie-breakers. Cached nearest
    distances keep this equivalent to the naive greedy algorithm while avoiding
    repeated distance recomputation.
    """
    if max_syncs <= 0 or not pose_estimates:
        return set()

    ranked_syncs = sorted(pose_estimates.keys(), key=lambda sync: (-quality_scores.get(sync, float("-inf")), sync))
    selected: list[int] = [int(ranked_syncs[0])]
    remaining = set(int(sync) for sync in ranked_syncs[1:])
    target_count = min(max_syncs, len(ranked_syncs))

    if progress_callback is not None:
        progress_callback(
            len(selected) / max(target_count, 1),
            f"Pose diversity selected {len(selected)}/{target_count} frames",
        )

    min_distances_to_selection = {
        sync: _pose_diversity_distance(
            pose_estimates[sync],
            pose_estimates[selected[0]],
            translation_scale_m=translation_scale_m,
            rotation_scale_deg=rotation_scale_deg,
        )
        for sync in remaining
    }
    report_every = max(1, target_count // 20)
    last_reported_count = len(selected)

    while remaining and len(selected) < max_syncs:
        best_sync = max(
            remaining,
            key=lambda sync: (
                min_distances_to_selection[sync],
                quality_scores.get(sync, float("-inf")),
                -sync,
            ),
        )
        selected.append(int(best_sync))
        remaining.remove(best_sync)
        min_distances_to_selection.pop(best_sync, None)

        for sync in remaining:
            distance_to_new_selection = _pose_diversity_distance(
                pose_estimates[sync],
                pose_estimates[best_sync],
                translation_scale_m=translation_scale_m,
                rotation_scale_deg=rotation_scale_deg,
            )
            min_distances_to_selection[sync] = min(
                min_distances_to_selection[sync],
                distance_to_new_selection,
            )

        if progress_callback is not None and (
            len(selected) == target_count or len(selected) - last_reported_count >= report_every
        ):
            progress_callback(
                len(selected) / max(target_count, 1),
                f"Pose diversity selected {len(selected)}/{target_count} frames",
            )
            last_reported_count = len(selected)

    return set(selected)


def _select_rigid_target_syncs_by_quality_and_pose_diversity(
    image_points: ImagePoints,
    pose_candidates_by_sync: dict[int, list[RigidPoseEstimate]],
    point_id_to_obj: dict[int, NDArray[np.float64]],
    *,
    max_syncs: int | None,
    quality_pool_multiplier: int = 4,
    rotation_scale_deg: float = 30.0,
    progress_callback: StageProgressCallback | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Select cube syncs with quality filtering followed by pose diversity.

    The quality stage keeps a pool of strong frames. The diversity stage caps
    the final count by selecting frames spread out in translation and rotation.
    ``reasons`` records why every candidate was kept or dropped for diagnostics.
    """
    candidate_syncs = set(int(sync_index) for sync_index in pose_candidates_by_sync)
    if max_syncs is None or len(candidate_syncs) <= max_syncs:
        if progress_callback is not None:
            progress_callback(1.0, f"Using all {len(candidate_syncs)} usable cube frames")
        return candidate_syncs, {sync_index: "all_usable" for sync_index in candidate_syncs}
    if max_syncs <= 0:
        return set(), {sync_index: "frame_limit_zero" for sync_index in candidate_syncs}

    if progress_callback is not None:
        progress_callback(0.05, f"Scoring {len(candidate_syncs)} cube frames")
    quality_scores = _rigid_target_frame_quality_scores(image_points, pose_candidates_by_sync, point_id_to_obj)
    ranked_by_quality = sorted(candidate_syncs, key=lambda sync: (-quality_scores.get(sync, float("-inf")), sync))
    quality_pool_size = min(len(ranked_by_quality), max_syncs * max(1, quality_pool_multiplier))
    quality_pool = set(ranked_by_quality[:quality_pool_size])
    if progress_callback is not None:
        progress_callback(0.25, f"Quality filter kept {len(quality_pool)} candidate frames")

    pooled_candidates = {
        sync_index: estimates
        for sync_index, estimates in pose_candidates_by_sync.items()
        if int(sync_index) in quality_pool
    }
    if progress_callback is not None:
        progress_callback(0.35, "Fusing cube pose candidates for diversity selection")
    pose_estimates = _fuse_pose_candidates_to_world_poses(pooled_candidates)

    def _diversity_progress(fraction: float, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(0.40 + 0.60 * max(0.0, min(1.0, float(fraction))), message)

    selected_syncs = _select_pose_diverse_syncs(
        pose_estimates,
        quality_scores,
        max_syncs=max_syncs,
        translation_scale_m=max(_rigid_target_characteristic_scale(point_id_to_obj), 1e-6),
        rotation_scale_deg=rotation_scale_deg,
        progress_callback=_diversity_progress if progress_callback is not None else None,
    )

    reasons: dict[int, str] = {}
    for sync_index in candidate_syncs:
        if sync_index in selected_syncs:
            reasons[sync_index] = "pose_diverse_selected"
        elif sync_index in quality_pool:
            reasons[sync_index] = "pose_diversity_rejected"
        else:
            reasons[sync_index] = "quality_filtered"
    return selected_syncs, reasons


def _fuse_pose_candidates_to_world_poses(
    candidates_by_sync: dict[int, list[RigidPoseEstimate]],
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    """Fuse per-camera cube pose candidates into one world pose per sync."""
    fused_world_poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]] = {}
    for sync_index, estimates in candidates_by_sync.items():
        weights = np.array([estimate.weight for estimate in estimates], dtype=np.float64)
        averaged_rotation = _average_rotation_matrices(
            [estimate.rotation_obj_to_world for estimate in estimates],
            weights,
        )
        averaged_translation = np.average(
            np.stack([estimate.translation_obj_to_world for estimate in estimates], axis=0),
            axis=0,
            weights=weights,
        )
        frame_time = float(np.mean([estimate.frame_time for estimate in estimates]))
        fused_world_poses[int(sync_index)] = (averaged_rotation, averaged_translation, frame_time)

    return fused_world_poses


def rigid_target_point_id_to_obj(image_points: ImagePoints) -> dict[int, NDArray[np.float64]]:
    """Extract stable object-frame coordinates for each tracked cube point.

    The cube tracker stores known object coordinates in ``obj_loc_*`` columns.
    This function condenses repeated observations into one coordinate per
    ``point_id`` so the optimizer has a fixed rigid target model.
    """
    df = image_points.df.copy()
    required = ["point_id", "obj_loc_x", "obj_loc_y"]
    if not all(column in df.columns for column in required):
        return {}

    df = df.dropna(subset=["obj_loc_x", "obj_loc_y"]).copy()
    if df.empty:
        return {}

    if "obj_loc_z" not in df.columns or df["obj_loc_z"].isna().all():
        df["obj_loc_z"] = 0.0
    else:
        df = df.dropna(subset=["obj_loc_z"]).copy()

    if df.empty:
        return {}

    grouped = (
        df.groupby("point_id")[["obj_loc_x", "obj_loc_y", "obj_loc_z"]]
        .median()
        .sort_index()
    )
    return {
        int(point_id): row.to_numpy(dtype=np.float64)
        for point_id, row in grouped.iterrows()
    }


def supports_rigid_target_optimization(image_points: ImagePoints) -> bool:
    """Return True when observations describe a non-planar rigid target."""
    point_id_to_obj = rigid_target_point_id_to_obj(image_points)
    if len(point_id_to_obj) < 4:
        return False

    obj_points = np.array(list(point_id_to_obj.values()), dtype=np.float32)
    return not _object_points_are_planar(obj_points)


def estimate_target_pose_world_poses(
    camera_array: CameraArray,
    image_points: ImagePoints,
    point_id_to_obj: dict[int, NDArray[np.float64]],
    *,
    min_points: int = 4,
    max_translation_fraction: float = 0.45,
    min_translation_m: float = 0.03,
    max_rotation_deg: float = 35.0,
    allow_single_candidate_fallback: bool = True,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    """Estimate fused rigid target poses in world coordinates for each sync."""
    pose_candidates_by_sync = _estimate_target_pose_world_pose_candidates(
        camera_array,
        image_points,
        point_id_to_obj,
        min_points=min_points,
    )
    filtered_candidates_by_sync = _filter_pose_estimates_by_cross_camera_agreement(
        pose_candidates_by_sync,
        point_id_to_obj,
        max_translation_fraction=max_translation_fraction,
        min_translation_m=min_translation_m,
        max_rotation_deg=max_rotation_deg,
        allow_single_candidate_fallback=allow_single_candidate_fallback,
    )

    return _fuse_pose_candidates_to_world_poses(filtered_candidates_by_sync)


def _solve_target_pose_from_normalized_correspondences(
    *,
    obj_points: NDArray[np.float32],
    undistorted_img_points: NDArray[np.float32],
    perfect_camera_matrix: NDArray[np.float64],
    zero_distortion: NDArray[np.float64],
    point_ids: NDArray[np.int32] | None,
    marker_face_geometry: dict[int, MarkerFaceGeometry | tuple[NDArray[np.float64], NDArray[np.float64]]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    """Solve one rigid target pose from normalized correspondences.

    Mirrors the dedicated cube tracker: a single square uses ``IPPE_SQUARE``,
    multiple markers on one face use planar ``IPPE``, and multi-face data uses
    non-planar PnP candidates. Planar points are canonicalized when OpenCV
    expects a board plane, every candidate is scored by normalized reprojection
    error, and face-normal validation rejects mirrored cube poses when possible.
    """
    is_planar = _object_points_are_planar(obj_points)
    solve_obj_points = obj_points
    plane_to_object = None
    plane_origin = None

    observed_marker_ids = (
        sorted({int(point_id) // 10 for point_id in point_ids})
        if point_ids is not None and len(point_ids) == len(obj_points)
        else []
    )
    observed_face_keys: set[tuple[int, ...]] = set()
    if marker_face_geometry is not None:
        for marker_id in observed_marker_ids:
            face_geometry = marker_face_geometry.get(marker_id)
            if face_geometry is None:
                continue
            normalized_geometry = _coerce_marker_face_geometry(face_geometry)
            if normalized_geometry is not None:
                observed_face_keys.add(normalized_geometry.face_key)
    is_single_marker_square = len(observed_marker_ids) == 1 and len(obj_points) == 4
    is_single_face = len(observed_face_keys) == 1 and len(observed_marker_ids) > 0

    if is_planar and not _object_points_use_canonical_board_plane(obj_points):
        solve_obj_points, plane_to_object, plane_origin = _canonicalize_planar_object_points(obj_points)

    n_points = int(len(solve_obj_points))
    candidates: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    if is_single_marker_square:
        sort_order = np.argsort(np.asarray(point_ids, dtype=np.int32) % 10) if point_ids is not None else None
        solve_img_points = undistorted_img_points if sort_order is None else undistorted_img_points[sort_order]
        solve_square_points = solve_obj_points if sort_order is None else solve_obj_points[sort_order]
        try:
            success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                solve_square_points,
                solve_img_points,
                perfect_camera_matrix,
                zero_distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error as exc:
            logger.debug("Skipping IPPE_SQUARE candidate for single-marker cube observation: %s", exc)
            success, rvecs, tvecs = False, (), ()
        if success:
            candidates.extend((np.asarray(rvec), np.asarray(tvec)) for rvec, tvec in zip(rvecs, tvecs, strict=True))
        candidate_flags = [cv2.SOLVEPNP_SQPNP]
        if n_points >= 4:
            candidate_flags.append(cv2.SOLVEPNP_ITERATIVE)
    elif is_planar and is_single_face:
        try:
            success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                solve_obj_points,
                undistorted_img_points,
                perfect_camera_matrix,
                zero_distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error as exc:
            logger.debug("Skipping IPPE candidate for %s planar points: %s", n_points, exc)
            success, rvecs, tvecs = False, (), ()
        if success:
            candidates.extend((np.asarray(rvec), np.asarray(tvec)) for rvec, tvec in zip(rvecs, tvecs, strict=True))
        candidate_flags = [cv2.SOLVEPNP_SQPNP]
        if n_points >= 4:
            candidate_flags.append(cv2.SOLVEPNP_ITERATIVE)
    else:
        candidate_flags = [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]
        if n_points >= 6:
            candidate_flags.append(cv2.SOLVEPNP_ITERATIVE)

    for flags in dict.fromkeys(candidate_flags):
        try:
            success, rvec, tvec = cv2.solvePnP(
                solve_obj_points,
                undistorted_img_points,
                cameraMatrix=perfect_camera_matrix,
                distCoeffs=zero_distortion,
                flags=flags,
            )
        except cv2.error as exc:
            logger.debug("Skipping solvePnP candidate flags=%s for %s points: %s", flags, n_points, exc)
            continue
        if success:
            candidates.append((np.asarray(rvec), np.asarray(tvec)))

    best_valid_result: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    best_any_result: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    for rvec, tvec in candidates:
        rotation_obj_to_cam, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        translation_obj_to_cam = np.asarray(tvec, dtype=np.float64).reshape(3)

        if plane_to_object is not None and plane_origin is not None:
            rotation_obj_to_cam = rotation_obj_to_cam @ plane_to_object.T
            translation_obj_to_cam = translation_obj_to_cam - rotation_obj_to_cam @ plane_origin

        rvec_eval, _ = cv2.Rodrigues(rotation_obj_to_cam)
        projected, _ = cv2.projectPoints(
            np.asarray(obj_points, dtype=np.float32),
            rvec_eval,
            translation_obj_to_cam.reshape(3, 1),
            perfect_camera_matrix,
            zero_distortion,
        )
        rmse = float(
            np.sqrt(np.mean(np.sum((undistorted_img_points - projected.reshape(-1, 2)) ** 2, axis=1)))
        )
        if best_any_result is None or rmse < best_any_result[0]:
            best_any_result = (rmse, rotation_obj_to_cam, translation_obj_to_cam)

        faces_camera = _observed_faces_camera_side_valid(
            rotation_obj_to_cam,
            translation_obj_to_cam,
            point_ids,
            marker_face_geometry,
        )
        if faces_camera and (best_valid_result is None or rmse < best_valid_result[0]):
            best_valid_result = (rmse, rotation_obj_to_cam, translation_obj_to_cam)

    best_result = best_valid_result or best_any_result
    if best_result is None:
        return None
    return best_result[1], best_result[2], best_result[0]


def _marker_face_geometry_from_point_ids(
    point_id_to_obj: dict[int, NDArray[np.float64]],
) -> dict[int, MarkerFaceGeometry]:
    """Precompute face center, outward normal, and face key for each marker."""
    marker_to_corners: dict[int, list[tuple[int, NDArray[np.float64]]]] = {}
    for point_id, obj_point in point_id_to_obj.items():
        marker_to_corners.setdefault(int(point_id) // 10, []).append((int(point_id) % 10, obj_point))

    geometry: dict[int, MarkerFaceGeometry] = {}
    for marker_id, entries in marker_to_corners.items():
        if len(entries) != 4:
            continue
        ordered = [point for _, point in sorted(entries, key=lambda entry: entry[0])]
        corners = np.asarray(ordered, dtype=np.float64)
        face_center = corners.mean(axis=0)
        x_axis = corners[1] - corners[0]
        y_axis = corners[0] - corners[3]
        normal = np.cross(x_axis, y_axis)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-8:
            continue
        normal_unit = normal / normal_norm
        face_key = tuple(
            int(round(value / 1e-6))
            for value in (*normal_unit.tolist(), float(np.dot(normal_unit, face_center)))
        )
        geometry[marker_id] = MarkerFaceGeometry(
            face_center=face_center,
            normal=normal_unit,
            face_key=face_key,
        )
    return geometry


def _observed_faces_camera_side_valid(
    rotation_obj_to_cam: NDArray[np.float64],
    translation_obj_to_cam: NDArray[np.float64],
    point_ids: NDArray[np.int32] | None,
    marker_face_geometry: dict[int, MarkerFaceGeometry | tuple[NDArray[np.float64], NDArray[np.float64]]] | None,
) -> bool:
    """Check that observed cube faces point toward the camera.

    This is the rigid-target counterpart to ``ArucoCubeTracker`` face-normal
    validation. It helps reject the flipped solution that planar PnP can return
    for one-face cube observations.
    """
    if point_ids is None or marker_face_geometry is None:
        return True

    observed_marker_ids = sorted({int(point_id) // 10 for point_id in point_ids})
    for marker_id in observed_marker_ids:
        face_geometry = marker_face_geometry.get(marker_id)
        if face_geometry is None:
            continue
        normalized_geometry = _coerce_marker_face_geometry(face_geometry)
        if normalized_geometry is None:
            continue
        face_center_cam = rotation_obj_to_cam @ normalized_geometry.face_center + translation_obj_to_cam
        normal_cam = rotation_obj_to_cam @ normalized_geometry.normal
        if float(np.dot(normal_cam, -face_center_cam)) <= 0.0:
            return False

    return True


def world_points_from_target_poses(
    point_id_to_obj: dict[int, NDArray[np.float64]],
    world_pose_estimates_by_sync: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], float]],
) -> WorldPoints:
    """Expand per-sync target poses into world coordinates for every cube point."""
    if not point_id_to_obj or not world_pose_estimates_by_sync:
        return WorldPoints(pd.DataFrame(columns=list(WORLD_POINT_COLUMNS.keys())))

    all_point_ids = np.array(sorted(int(point_id) for point_id in point_id_to_obj.keys()), dtype=np.int64)
    all_obj_points = np.array([point_id_to_obj[pid] for pid in all_point_ids], dtype=np.float64)

    rows: list[dict[str, float | int]] = []
    for sync_index in sorted(world_pose_estimates_by_sync.keys()):
        rotation_obj_to_world, translation_obj_to_world, frame_time = world_pose_estimates_by_sync[sync_index]
        world_points = all_obj_points @ rotation_obj_to_world.T + translation_obj_to_world
        for point_id, world_point in zip(all_point_ids, world_points, strict=True):
            rows.append(
                {
                    "sync_index": int(sync_index),
                    "point_id": int(point_id),
                    "x_coord": float(world_point[0]),
                    "y_coord": float(world_point[1]),
                    "z_coord": float(world_point[2]),
                    "frame_time": frame_time,
                }
            )

    return WorldPoints(pd.DataFrame(rows, columns=list(WORLD_POINT_COLUMNS.keys())))


def estimate_target_pose_world_points_from_image_points(
    camera_array: CameraArray,
    image_points: ImagePoints,
    point_id_to_obj: dict[int, NDArray[np.float64]],
    *,
    min_points: int = 4,
) -> WorldPoints:
    """Estimate cube world points directly from image observations.

    This preview/bootstrap path produces a complete rigid target point cloud per
    usable sync, even when cameras see different cube faces and triangulation
    cannot match the same marker corners across cameras.
    """
    world_pose_estimates_by_sync = estimate_target_pose_world_poses(
        camera_array,
        image_points,
        point_id_to_obj,
        min_points=min_points,
    )
    return world_points_from_target_poses(point_id_to_obj, world_pose_estimates_by_sync)


def _rigid_target_bundle_residuals(
    params: NDArray[np.float64],
    camera_array: CameraArray,
    camera_indices: CameraIndices,
    pose_indices: NDArray[np.int32],
    object_coords: NDArray[np.float64],
    observed_normalized: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Residual vector for joint camera-extrinsic and cube-pose optimization."""
    n_cams = len(camera_array.posed_cameras)
    camera_params = params[: n_cams * 6].reshape((n_cams, 6))
    pose_params = params[n_cams * 6 :].reshape((-1, 6))

    world_coords = np.zeros_like(object_coords, dtype=np.float64)
    for pose_index in range(len(pose_params)):
        mask = pose_indices == pose_index
        if not np.any(mask):
            continue
        rvec = pose_params[pose_index, :3]
        translation = pose_params[pose_index, 3:6]
        rotation, _ = cv2.Rodrigues(rvec)
        world_coords[mask] = object_coords[mask] @ rotation.T + translation

    errors_xy = np.zeros_like(observed_normalized, dtype=np.float64)
    for cam_id, camera_data in camera_array.posed_cameras.items():
        camera_index = camera_array.posed_cam_id_to_index[cam_id]
        mask = camera_indices == camera_index
        if not np.any(mask):
            continue

        projected = project_world_to_image(
            camera_data,
            world_coords[mask],
            rvec=camera_params[camera_index, :3],
            tvec=camera_params[camera_index, 3:6],
            use_normalized=True,
        )
        errors_xy[mask] = projected - observed_normalized[mask]

    return errors_xy.ravel()


def _normalized_robust_f_scale(camera_array: CameraArray, robust_loss_pixels: float) -> float:
    """Convert a pixel-domain robust loss scale into normalized coordinates."""
    focal_lengths: list[float] = []
    for camera in camera_array.posed_cameras.values():
        if camera.matrix is None:
            continue
        focal_lengths.extend([float(camera.matrix[0, 0]), float(camera.matrix[1, 1])])

    positive_focals = [focal for focal in focal_lengths if focal > 0.0]
    if not positive_focals:
        return max(float(robust_loss_pixels) / 1000.0, 1e-6)

    return max(float(robust_loss_pixels) / float(np.median(positive_focals)), 1e-6)


def _rigid_target_sparsity_pattern(
    n_observations: int,
    n_cameras: int,
    n_poses: int,
    camera_indices: CameraIndices,
    pose_indices: NDArray[np.int32],
) -> lil_matrix:
    """Build Jacobian sparsity for camera params plus per-sync cube poses."""
    n_residuals = n_observations * 2
    n_params = n_cameras * 6 + n_poses * 6
    sparsity = lil_matrix((n_residuals, n_params), dtype=int)
    obs_idx = np.arange(n_observations)

    for cam_param in range(6):
        param_col = camera_indices * 6 + cam_param
        sparsity[2 * obs_idx, param_col] = 1
        sparsity[2 * obs_idx + 1, param_col] = 1

    pose_offset = n_cameras * 6
    for pose_param in range(6):
        param_col = pose_offset + pose_indices * 6 + pose_param
        sparsity[2 * obs_idx, param_col] = 1
        sparsity[2 * obs_idx + 1, param_col] = 1

    return sparsity


@dataclass(frozen=True)
class OptimizationStatus:
    """Result metadata from bundle adjustment optimization.

    Populated by optimize(), cleared by filter methods.
    """

    converged: bool
    termination_reason: str  # "converged_gtol", "max_evaluations", etc.
    iterations: int  # nfev from scipy
    final_cost: float


# Mapping from scipy least_squares status codes to human-readable reasons
_SCIPY_STATUS_REASONS: dict[int, str] = {
    -1: "improper_input",
    0: "max_evaluations",
    1: "converged_gtol",
    2: "converged_ftol",
    3: "converged_xtol",
    4: "converged_small_step",
}


@dataclass(frozen=True)
class CaptureVolume:
    camera_array: CameraArray
    image_points: ImagePoints
    world_points: WorldPoints
    # Computed field: maps each image observation to its world point index (-1 if unmatched)
    img_to_obj_map: np.ndarray = field(init=False)
    # Optimization metadata: None if capture volume hasn't been optimized or was filtered post-optimization
    _optimization_status: OptimizationStatus | None = field(default=None, compare=False)

    @property
    def optimization_status(self) -> OptimizationStatus | None:
        """Optimization result metadata, or None if not from optimize() call."""
        return self._optimization_status

    @cached_property
    def rigid_target_object_points(self) -> dict[int, NDArray[np.float64]]:
        """Cached rigid-target object coordinates keyed by point_id."""
        return rigid_target_point_id_to_obj(self.image_points)

    @cached_property
    def rigid_target_pose_candidates_by_sync(self) -> dict[int, list[RigidPoseEstimate]]:
        """Cached per-camera rigid-target pose candidates keyed by sync_index."""
        if len(self.rigid_target_object_points) < 4:
            return {}
        return _estimate_target_pose_world_pose_candidates(
            self.camera_array,
            self.image_points,
            self.rigid_target_object_points,
            min_points=4,
        )

    def __post_init__(self):
        """Compute mapping and validate geometry."""
        object.__setattr__(self, "img_to_obj_map", self._compute_img_to_obj_map())
        self._validate_geometry()

    def estimate_target_pose_world_points(
        self,
        point_id_to_obj: dict[int, NDArray[np.float64]],
        *,
        min_points: int = 4,
        image_points: ImagePoints | None = None,
    ) -> WorldPoints:
        """Estimate full rigid-target world points from per-frame target poses.

        This is primarily useful for non-planar rigid targets such as an ArUco
        cube, where different cameras may see different faces at the same sync
        index. Triangulation remains sparse in that situation because it still
        requires shared ``point_id`` values across cameras, but a valid target
        pose can still be estimated from any single camera with enough corners.
        """
        source_image_points = self.image_points if image_points is None else image_points
        return estimate_target_pose_world_points_from_image_points(
            self.camera_array,
            source_image_points,
            point_id_to_obj,
            min_points=min_points,
        )

    def _validate_geometry(self):
        """Ensure data counts make geometric sense."""
        n_img = len(self.image_points.df)
        n_world = len(self.world_points.df)
        n_cams = len(self.camera_array.posed_cameras)

        if n_img == 0:
            raise ValueError("No image observations provided")
        if n_world == 0:
            raise ValueError("No world points provided")
        if n_cams == 0:
            raise ValueError("No posed cameras in array")

        # Check that we have at least some matched observations
        n_matched = np.sum(self.img_to_obj_map >= 0)
        if n_matched == 0:
            raise ValueError("No image observations have corresponding world points")

        if not supports_rigid_target_optimization(self.image_points) and n_matched < n_world * 2:
            logger.warning(
                f"Suspicious geometry: {n_matched} matched observations for {n_world} world points. "
                f"Expected at least {n_world * 2} for multi-view geometry."
            )
        # Validate indices are in bounds
        valid_indices = self.img_to_obj_map[self.img_to_obj_map >= 0]
        if valid_indices.size > 0 and valid_indices.max() >= n_world:
            raise ValueError(f"obj_indices contains out-of-bounds index: {valid_indices.max()} >= {n_world}")

    def _compute_img_to_obj_map(self) -> np.ndarray:
        """Map each image observation to its world point index. Returns -1 for unmatched."""
        world_df = self.world_points.df.reset_index().rename(columns={"index": "world_idx"})
        mapping = world_df.set_index(["sync_index", "point_id"])["world_idx"].to_dict()

        img_df = self.image_points.df
        keys = list(zip(img_df["sync_index"], img_df["point_id"]))
        img_to_obj_map = np.array([mapping.get(key, -1) for key in keys], dtype=np.int32)

        n_unmatched = np.sum(img_to_obj_map == -1)
        if n_unmatched > 0:
            logger.info(f"{n_unmatched} of {len(img_to_obj_map)} image observations have no world point")

        return img_to_obj_map

    @cached_property
    def reprojection_report(self) -> ReprojectionReport:
        """
        Generate comprehensive reprojection error report in pixel units.
        Cached automatically since capture volume data is immutable.
        """
        # 1. Filter to matched observations from posed cameras only
        matched_mask = self.img_to_obj_map >= 0
        posed_cam_ids = set(self.camera_array.posed_cam_id_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
        combined_mask = matched_mask & posed_mask

        n_total = len(self.img_to_obj_map)
        n_matched = combined_mask.sum()
        n_unmatched = n_total - n_matched

        if n_matched == 0:
            raise ValueError("No matched observations for reprojection error calculation")

        matched_img_df = self.image_points.df[combined_mask]
        matched_obj_indices = self.img_to_obj_map[combined_mask]

        # 2. Prepare arrays for core function
        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]], dtype=np.int16
        )
        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        world_coords: WorldCoords = self.world_points.points[matched_obj_indices]

        # 3. Compute reprojection errors
        errors_xy: ErrorsXY = reprojection_errors(
            self.camera_array, camera_indices, image_coords, world_coords, use_normalized=False
        )

        # 4. Build raw_errors DataFrame
        euclidean_error = np.sqrt(np.sum(errors_xy**2, axis=1))
        raw_errors = pd.DataFrame(
            {
                "sync_index": matched_img_df["sync_index"].values,
                "cam_id": matched_img_df["cam_id"].values,
                "point_id": matched_img_df["point_id"].values,
                "error_x": errors_xy[:, 0],
                "error_y": errors_xy[:, 1],
                "euclidean_error": euclidean_error,
            }
        )

        # 5. Aggregate metrics
        overall_rmse = float(np.sqrt(np.mean(euclidean_error**2)))

        by_camera = {}
        for cam_id in self.camera_array.posed_cameras.keys():
            cam_errors = euclidean_error[matched_img_df["cam_id"] == cam_id]
            by_camera[cam_id] = float(np.sqrt(np.mean(cam_errors**2))) if len(cam_errors) > 0 else 0.0

        by_point_id = {}
        for point_id in np.unique(matched_img_df["point_id"]):
            point_errors = euclidean_error[matched_img_df["point_id"] == point_id]
            by_point_id[point_id] = float(np.sqrt(np.mean(point_errors**2)))

        # 6. Count unmatched by camera (only count for posed cameras)
        unmatched_by_camera = {}
        for cam_id in self.camera_array.cameras.keys():
            cam_total = (self.image_points.df["cam_id"] == cam_id).sum()
            cam_matched = ((self.image_points.df["cam_id"] == cam_id) & combined_mask).sum()
            unmatched_by_camera[cam_id] = int(cam_total - cam_matched)

        # 7. Create and cache report
        report = ReprojectionReport(
            overall_rmse=overall_rmse,
            by_camera=by_camera,
            by_point_id=by_point_id,
            n_unmatched_observations=int(n_unmatched),
            unmatched_rate=n_unmatched / n_total if n_total > 0 else 0.0,
            unmatched_by_camera=unmatched_by_camera,
            raw_errors=raw_errors,
            n_observations_matched=int(n_matched),
            n_observations_total=int(n_total),
            n_cameras=len(self.camera_array.posed_cameras),
            n_points=len(self.world_points.points),
        )

        return report

    def save(self, directory: Path | str) -> None:
        """Save capture volume to a directory.

        Writes camera_array.toml, image_points.csv, world_points.csv.
        Note: optimization_status is not persisted.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.camera_array.to_toml(directory / "camera_array.toml")
        self.image_points.to_csv(directory / "image_points.csv")
        self.world_points.to_csv(directory / "world_points.csv")

    @classmethod
    def load(cls, directory: Path | str) -> CaptureVolume:
        """Load capture volume from a directory.

        Expects: camera_array.toml, image_points.csv, world_points.csv.
        """
        directory = Path(directory)
        camera_array = CameraArray.from_toml(directory / "camera_array.toml")
        image_points = ImagePoints.from_csv(directory / "image_points.csv")
        world_points = WorldPoints.from_csv(directory / "world_points.csv")
        return cls(camera_array=camera_array, image_points=image_points, world_points=world_points)

    @classmethod
    def bootstrap(
        cls,
        image_points: ImagePoints,
        camera_array: CameraArray,
    ) -> CaptureVolume:
        """Bootstrap extrinsic calibration from 2D observations.

        Pipeline: deepcopy cameras → build pose network → apply poses → triangulate.
        Does NOT auto-optimize. Call .optimize() on the result.
        The input CameraArray is not modified.

        Raises:
            CalibrationError: If cameras lack intrinsics, cam_ids mismatch,
                or insufficient stereo pairs exist.
        """
        from caliscope.exceptions import CalibrationError
        from caliscope.core.bootstrap_pose.build_paired_pose_network import (
            build_paired_pose_network,
        )

        # Validate: cam_id mismatch
        point_cam_ids = set(image_points.df["cam_id"].unique())
        array_cam_ids = set(camera_array.cameras.keys())
        missing_cameras = point_cam_ids - array_cam_ids
        if missing_cameras:
            raise CalibrationError(f"ImagePoints reference cameras {missing_cameras} not in the CameraArray.")

        # Validate: intrinsics
        uncalibrated = [
            cam_id for cam_id, cam in camera_array.cameras.items() if cam.matrix is None or cam.distortions is None
        ]
        if uncalibrated:
            raise CalibrationError(
                f"Cannot run extrinsic calibration -- cameras {uncalibrated} have "
                f"no intrinsic calibration.\n\n"
                f"Run calibrate_intrinsics() for each camera first:\n"
                f"    output = calibrate_intrinsics(points, cameras[{uncalibrated[0]}])\n"
                f"    cameras[{uncalibrated[0]}] = output.camera"
            )

        # Validate: obj_loc presence
        obj_cols = image_points.df[["obj_loc_x", "obj_loc_y", "obj_loc_z"]]
        if obj_cols.isna().all().all():
            raise CalibrationError(
                "ImagePoints contain no object location data (obj_loc columns are all NaN). "
                "Extrinsic calibration requires a tracker that provides known 3D positions "
                "(e.g., CharucoTracker)."
            )

        cameras = deepcopy(camera_array)
        pose_network = build_paired_pose_network(image_points, cameras)
        pose_network.apply_to(cameras)
        world_points = image_points.triangulate(cameras)

        return cls(camera_array=cameras, image_points=image_points, world_points=world_points)

    def optimize(
        self,
        ftol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 0,
        strict: bool = True,
        progress_callback: OptimizationProgressCallback | None = None,
    ) -> CaptureVolume:
        """
        Perform bundle adjustment optimization on this CaptureVolume.

        Returns a NEW CaptureVolume with optimized camera parameters and 3D points.
        The original remains unchanged (immutable pattern).
        """
        # Extract static data once - filter to matched observations from posed cameras
        matched_mask = self.img_to_obj_map >= 0
        posed_cam_ids = set(self.camera_array.posed_cam_id_to_index.keys())
        posed_mask: np.ndarray = self.image_points.df["cam_id"].isin(posed_cam_ids).to_numpy()
        combined_mask = matched_mask & posed_mask

        matched_img_df = self.image_points.df[combined_mask]

        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_cam_id_to_index[cam_id] for cam_id in matched_img_df["cam_id"]], dtype=np.int16
        )

        image_coords: ImageCoords = matched_img_df[["img_loc_x", "img_loc_y"]].values
        image_to_world_indices = self.img_to_obj_map[combined_mask]

        # Initial parameters from current state
        initial_params = self._get_vectorized_params()

        # Get sparsity pattern for Jacobian
        sparsity_pattern = self._get_sparsity_pattern(camera_indices, image_to_world_indices)

        # Perform optimization
        logger.info(f"Beginning bundle adjustment on {len(image_coords)} observations")

        def _solver_callback(intermediate_result) -> None:
            if progress_callback is None:
                return
            progress_callback(
                int(getattr(intermediate_result, "nfev", 0)),
                float(getattr(intermediate_result, "cost", 0.0)),
            )

        result = least_squares(
            bundle_residuals,
            initial_params,
            args=(self.camera_array, camera_indices, image_coords, image_to_world_indices, True),
            jac_sparsity=sparsity_pattern,  # Now using sparse Jacobian
            verbose=verbose,
            x_scale="jac",
            loss="linear",
            ftol=ftol,
            max_nfev=max_nfev,
            method="trf",
            callback=_solver_callback if progress_callback is not None else None,
        )

        # Capture optimization status
        termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
        converged = result.status in (1, 2, 3, 4)  # Any gtol/ftol/xtol convergence

        optimization_status = OptimizationStatus(
            converged=converged,
            termination_reason=termination_reason,
            iterations=result.nfev,
            final_cost=float(result.cost),
        )

        if strict and not converged:
            from caliscope.exceptions import CalibrationError

            raise CalibrationError(
                f"Bundle adjustment did not converge: {termination_reason}\n"
                f"Pass strict=False to suppress this error and inspect the result."
            )

        # Create new capture volume with optimized parameters
        new_camera_array = deepcopy(self.camera_array)
        new_camera_array.update_extrinsic_params(result.x)

        # Extract optimized 3D points
        n_cams = len(self.camera_array.posed_cameras)
        n_cam_params = 6
        optimized_points = result.x[n_cams * n_cam_params :].reshape((-1, 3))

        # Create new world points with optimized coordinates
        new_world_df = self.world_points.df.copy()
        matched_obj_unique = np.unique(image_to_world_indices)
        new_world_df.loc[matched_obj_unique, ["x_coord", "y_coord", "z_coord"]] = optimized_points

        new_world_points = WorldPoints(new_world_df)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            _optimization_status=optimization_status,
        )

    def optimize_rigid_target(
        self,
        ftol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 0,
        strict: bool = True,
        progress_callback: OptimizationProgressCallback | None = None,
        stage_progress_callback: StageProgressCallback | None = None,
        max_pose_disagreement_translation_fraction: float = 0.45,
        min_pose_disagreement_translation_m: float = 0.03,
        max_pose_disagreement_rotation_deg: float = 35.0,
        allow_single_candidate_fallback: bool = True,
        max_selected_syncs: int | None = 250,
        quality_pool_multiplier: int = 4,
        pose_diversity_rotation_scale_deg: float = 30.0,
        robust_loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1",
        robust_loss_pixels: float = 2.5,
    ) -> CaptureVolume:
        """Optimize camera extrinsics and one rigid target pose per selected sync index.

        Cube syncs are selected in two stages before optimization: weak frames
        are removed by quality rank, then the remaining pool is capped by greedy
        translation/rotation diversity in target-pose space.

        The optimized parameter vector contains all posed camera extrinsics
        followed by one six-parameter cube pose for each selected sync. Object
        point coordinates remain fixed in the cube frame; every residual is the
        difference between observed normalized image coordinates and projected
        cube corners transformed through the current sync pose and camera pose.
        """
        def _report_stage_progress(fraction: float, message: str) -> None:
            if stage_progress_callback is None:
                return
            stage_progress_callback(max(0.0, min(1.0, float(fraction))), message)

        _report_stage_progress(0.02, "Preparing rigid target object points")
        point_id_to_obj = self.rigid_target_object_points
        if len(point_id_to_obj) < 4:
            raise ValueError("Rigid-target optimization requires at least 4 points with object coordinates")

        _report_stage_progress(0.08, "Estimating cube poses from camera observations")
        pose_candidates_by_sync = self.rigid_target_pose_candidates_by_sync
        if not pose_candidates_by_sync:
            raise ValueError("Could not initialize rigid target poses from image observations")

        _report_stage_progress(0.20, f"Filtering {len(pose_candidates_by_sync)} syncs by camera agreement")
        filtered_candidates_by_sync = _filter_pose_estimates_by_cross_camera_agreement(
            pose_candidates_by_sync,
            point_id_to_obj,
            max_translation_fraction=max_pose_disagreement_translation_fraction,
            min_translation_m=min_pose_disagreement_translation_m,
            max_rotation_deg=max_pose_disagreement_rotation_deg,
            allow_single_candidate_fallback=allow_single_candidate_fallback,
        )
        if not filtered_candidates_by_sync:
            raise ValueError("Could not initialize rigid target poses from image observations")

        def _selection_progress(fraction: float, message: str) -> None:
            _report_stage_progress(0.35 + 0.35 * max(0.0, min(1.0, float(fraction))), message)

        _report_stage_progress(0.35, "Selecting high-quality diverse cube frames")
        selected_syncs, selection_reasons = _select_rigid_target_syncs_by_quality_and_pose_diversity(
            self.image_points,
            filtered_candidates_by_sync,
            point_id_to_obj,
            max_syncs=max_selected_syncs,
            quality_pool_multiplier=quality_pool_multiplier,
            rotation_scale_deg=pose_diversity_rotation_scale_deg,
            progress_callback=_selection_progress if stage_progress_callback is not None else None,
        )
        if not selected_syncs:
            raise ValueError("Rigid-target frame selection removed all usable cube syncs")

        _report_stage_progress(0.72, f"Fusing poses for {len(selected_syncs)} selected cube frames")
        selected_candidates_by_sync = {
            sync_index: estimates
            for sync_index, estimates in filtered_candidates_by_sync.items()
            if int(sync_index) in selected_syncs
        }
        logger.info(
            "Cube Capture Volume diagnostics: raw_candidate_syncs=%s, agreement_usable_syncs=%s, "
            "selected_syncs=%s, dropped_by_agreement=%s, dropped_by_quality=%s, dropped_by_pose_diversity=%s",
            len(pose_candidates_by_sync),
            len(filtered_candidates_by_sync),
            len(selected_candidates_by_sync),
            len(pose_candidates_by_sync) - len(filtered_candidates_by_sync),
            sum(1 for reason in selection_reasons.values() if reason == "quality_filtered"),
            sum(1 for reason in selection_reasons.values() if reason == "pose_diversity_rejected"),
        )
        pose_estimates = _fuse_pose_candidates_to_world_poses(selected_candidates_by_sync)

        _report_stage_progress(0.80, "Preparing cube observations")
        usable_sync_indices = np.array(sorted(pose_estimates.keys()), dtype=np.int64)
        posed_cam_ids = set(self.camera_array.posed_cam_id_to_index.keys())
        image_df = self.image_points.df.copy()
        image_df = image_df[image_df["sync_index"].isin(usable_sync_indices)].copy()
        image_df = image_df[image_df["cam_id"].isin(posed_cam_ids)].copy()
        image_df = image_df[image_df["point_id"].isin(point_id_to_obj.keys())].copy()

        if image_df.empty:
            raise ValueError("No observations remained for rigid-target optimization")

        _report_stage_progress(0.86, "Undistorting cube observations")
        normalized_chunks: list[pd.DataFrame] = []
        for cam_id in sorted(int(cam_id) for cam_id in image_df["cam_id"].unique()):
            camera = self.camera_array.cameras[cam_id]
            cam_df = image_df[image_df["cam_id"] == cam_id].copy()
            undistorted = camera.undistort_points(
                cam_df[["img_loc_x", "img_loc_y"]].to_numpy(dtype=np.float32),
                output="normalized",
            )
            cam_df[["norm_x", "norm_y"]] = undistorted
            normalized_chunks.append(cam_df)

        optimized_image_df = (
            pd.concat(normalized_chunks)
            .sort_values(["sync_index", "cam_id", "point_id"])
            .reset_index(drop=True)
        )
        optimized_image_points = ImagePoints(optimized_image_df.drop(columns=["norm_x", "norm_y"]))

        sync_to_pose_index = {sync_index: index for index, sync_index in enumerate(sorted(pose_estimates.keys()))}
        pose_index_to_sync = {index: sync_index for sync_index, index in sync_to_pose_index.items()}

        camera_indices: CameraIndices = np.array(
            [self.camera_array.posed_cam_id_to_index[int(cam_id)] for cam_id in optimized_image_df["cam_id"]],
            dtype=np.int16,
        )
        pose_indices = np.array(
            [sync_to_pose_index[int(sync_index)] for sync_index in optimized_image_df["sync_index"]],
            dtype=np.int32,
        )
        object_coords = np.array(
            [point_id_to_obj[int(point_id)] for point_id in optimized_image_df["point_id"]],
            dtype=np.float64,
        )
        observed_normalized = optimized_image_df[["norm_x", "norm_y"]].to_numpy(dtype=np.float64)

        initial_camera_params = self.camera_array.get_extrinsic_params()
        if initial_camera_params is None:
            raise ValueError("Camera extrinsic parameters not initialized")

        initial_pose_params = []
        frame_times: dict[int, float] = {}
        for sync_index in sorted(pose_estimates.keys()):
            rotation_obj_to_world, translation_obj_to_world, frame_time = pose_estimates[sync_index]
            rvec, _ = cv2.Rodrigues(rotation_obj_to_world)
            initial_pose_params.append(np.hstack([rvec.ravel(), translation_obj_to_world]))
            frame_times[sync_index] = frame_time

        initial_params = np.concatenate(
            [initial_camera_params.ravel(), np.array(initial_pose_params, dtype=np.float64).ravel()]
        )
        _report_stage_progress(0.94, "Building sparse cube optimization problem")
        sparsity_pattern = _rigid_target_sparsity_pattern(
            len(optimized_image_df),
            len(self.camera_array.posed_cameras),
            len(initial_pose_params),
            camera_indices,
            pose_indices,
        )

        logger.info(
            "Beginning rigid-target optimization on %s observations across %s syncs",
            len(optimized_image_df),
            len(initial_pose_params),
        )
        _report_stage_progress(1.0, f"Running optimizer on {len(optimized_image_df)} cube observations")

        def _solver_callback(intermediate_result) -> None:
            if progress_callback is None:
                return
            progress_callback(
                int(getattr(intermediate_result, "nfev", 0)),
                float(getattr(intermediate_result, "cost", 0.0)),
            )

        f_scale = _normalized_robust_f_scale(self.camera_array, robust_loss_pixels)
        logger.info(
            "Rigid-target optimizer using loss=%s (f_scale=%.6f normalized, %.2f px)",
            robust_loss,
            f_scale,
            robust_loss_pixels,
        )

        result = least_squares(
            _rigid_target_bundle_residuals,
            initial_params,
            args=(self.camera_array, camera_indices, pose_indices, object_coords, observed_normalized),
            jac_sparsity=sparsity_pattern,
            verbose=verbose,
            x_scale="jac",
            loss=robust_loss,
            f_scale=f_scale,
            ftol=ftol,
            max_nfev=max_nfev,
            method="trf",
            callback=_solver_callback if progress_callback is not None else None,
        )

        termination_reason = _SCIPY_STATUS_REASONS.get(result.status, f"unknown_{result.status}")
        converged = result.status in (1, 2, 3, 4)
        optimization_status = OptimizationStatus(
            converged=converged,
            termination_reason=termination_reason,
            iterations=result.nfev,
            final_cost=float(result.cost),
        )

        if strict and not converged:
            from caliscope.exceptions import CalibrationError

            raise CalibrationError(
                f"Rigid-target optimization did not converge: {termination_reason}\n"
                f"Pass strict=False to suppress this error and inspect the result."
            )

        new_camera_array = deepcopy(self.camera_array)
        new_camera_array.update_extrinsic_params(result.x)

        n_cameras = len(self.camera_array.posed_cameras)
        optimized_pose_params = result.x[n_cameras * 6 :].reshape((-1, 6))
        optimized_pose_estimates = {}
        for pose_index, pose_param in enumerate(optimized_pose_params):
            rotation_obj_to_world, _ = cv2.Rodrigues(pose_param[:3])
            translation_obj_to_world = pose_param[3:6]
            sync_index = pose_index_to_sync[pose_index]
            optimized_pose_estimates[sync_index] = (
                rotation_obj_to_world,
                translation_obj_to_world,
                frame_times[sync_index],
            )

        new_world_points = world_points_from_target_poses(point_id_to_obj, optimized_pose_estimates)

        optimized_capture_volume = CaptureVolume(
            camera_array=new_camera_array,
            image_points=optimized_image_points,
            world_points=new_world_points,
            _optimization_status=optimization_status,
        )
        return optimized_capture_volume

    def _get_sparsity_pattern(
        self,
        camera_indices: NDArray[np.int16],
        obj_indices: NDArray[np.int32],
    ) -> lil_matrix:
        """
        Generate sparsity pattern for Jacobian matrix.

        Each observation contributes 2 residuals (x_error, y_error).
        Each residual depends on:
        - 6 camera parameters (rotation + translation)
        - 3 point parameters (x, y, z)

        Args:
            camera_indices: (n_observations,) array mapping observations to cameras
            obj_indices: (n_observations,) array mapping observations to 3D points

        Returns:
            sparsity: lil_matrix of shape (n_residuals, n_params)
        """
        n_observations = len(camera_indices)
        n_cameras = len(self.camera_array.posed_cameras)
        n_points = len(self.world_points.points)

        # Jacobian dimensions: 2 residuals per observation
        n_residuals = n_observations * 2
        n_params = n_cameras * 6 + n_points * 3

        sparsity = lil_matrix((n_residuals, n_params), dtype=int)

        # Observation indices (0 to n_observations-1)
        obs_idx = np.arange(n_observations)

        # Camera parameter dependencies (first 6 params per camera)
        for cam_param in range(6):
            param_col = camera_indices * 6 + cam_param
            sparsity[2 * obs_idx, param_col] = 1  # x residual depends on camera param
            sparsity[2 * obs_idx + 1, param_col] = 1  # y residual depends on camera param

        # Point parameter dependencies (3 params per point, after camera params)
        for point_param in range(3):
            param_col = n_cameras * 6 + obj_indices * 3 + point_param
            sparsity[2 * obs_idx, param_col] = 1  # x residual depends on point param
            sparsity[2 * obs_idx + 1, param_col] = 1  # y residual depends on point param

        return sparsity

    def _get_vectorized_params(self) -> NDArray[np.float64]:
        """
        Convert camera extrinsics and 3D points to a flattened optimization vector.
        Shape: (n_cameras*6 + n_points*3,)
        """
        camera_params = self.camera_array.get_extrinsic_params()  # (n_cams, 6)
        if camera_params is None:
            raise ValueError("Camera extrinsic parameters not initialized")
        points_3d = self.world_points.points  # (n_points, 3)

        return np.concatenate([camera_params.ravel(), points_3d.ravel()])

    def _filter_by_reprojection_thresholds(self, thresholds: dict[int, float], min_per_camera: int) -> CaptureVolume:
        """
        Internal: Filter observations using per-camera error thresholds with safety enforcement.

        Args:
            thresholds: dict mapping camera cam_id -> max_error_pixels for that camera
            min_per_camera: minimum observations to preserve per camera

        Returns:
            New CaptureVolume with filtered observations
        """
        # Get reprojection data (cached)
        report = self.reprojection_report
        raw_errors = report.raw_errors

        # Build initial keep mask: error <= threshold for that camera's cam_id
        threshold_series = raw_errors["cam_id"].map(thresholds)
        keep_mask = (raw_errors["euclidean_error"] <= threshold_series).copy()

        # Apply safety: ensure each camera keeps at least min_per_camera observations
        for cam_id in raw_errors["cam_id"].unique():
            camera_idx = raw_errors["cam_id"] == cam_id
            n_keep = keep_mask[camera_idx].sum()
            n_total = camera_idx.sum()

            # If below minimum and we can add more
            if n_keep < min_per_camera and n_keep < n_total:
                # How many we need to add (capped at total available)
                n_needed = min(min_per_camera, n_total) - n_keep

                # Get errors for observations that would be filtered out
                # pandas stubs don't narrow .loc with boolean mask to Series
                filtered_errors = raw_errors.loc[camera_idx & ~keep_mask, "euclidean_error"]

                if len(filtered_errors) >= n_needed:  # type: ignore[arg-type]
                    # Find the error threshold that would keep exactly n_needed more observations
                    threshold_to_add: float = filtered_errors.nsmallest(n_needed).iloc[-1]  # type: ignore[union-attr]

                    # Update mask to keep observations with error <= threshold_to_add
                    keep_mask[camera_idx] = raw_errors.loc[camera_idx, "euclidean_error"] <= threshold_to_add  # type: ignore[index, operator]

        # Get keys of observations to keep
        keep_keys = raw_errors[keep_mask][["sync_index", "cam_id", "point_id"]]

        # Filter image points by merging with keep keys
        filtered_img_df = self.image_points.df.merge(keep_keys, on=["sync_index", "cam_id", "point_id"], how="inner")
        filtered_image_points = ImagePoints(filtered_img_df)

        # Prune orphaned world points (3D points with no observations)
        remaining_3d_keys = filtered_img_df[["sync_index", "point_id"]].drop_duplicates()
        filtered_world_df = self.world_points.df.merge(remaining_3d_keys, on=["sync_index", "point_id"], how="inner")

        filtered_world_points = WorldPoints(filtered_world_df)

        return CaptureVolume(
            camera_array=self.camera_array,
            image_points=filtered_image_points,
            world_points=filtered_world_points,
        )

    def filter_by_absolute_error(self, max_pixels: float, min_per_camera: int = 10) -> CaptureVolume:
        """
        Remove observations with reprojection error > max_pixels.

        Safety: Ensures each camera keeps at least min_per_camera observations.
        If a camera would drop below this threshold, the lowest-error observations
        are restored until the threshold is met.

        Args:
            max_pixels: Maximum reprojection error (pixels) to keep
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New CaptureVolume with filtered observations
        """
        if max_pixels <= 0:
            raise ValueError(f"max_pixels must be positive, got {max_pixels}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        # Build uniform thresholds for all posed cameras
        thresholds = {cam_id: max_pixels for cam_id in self.camera_array.posed_cameras.keys()}

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def filter_by_percentile_error(
        self, percentile: float, scope: Literal["per_camera", "overall"] = "per_camera", min_per_camera: int = 10
    ) -> CaptureVolume:
        """
        Remove worst N% of observations based on reprojection error.

        Args:
            percentile: Percentage of worst observations to remove (0-100)
            scope: "per_camera" computes percentile per camera, "overall" uses global percentile
            min_per_camera: Minimum observations per camera (safety floor)

        Returns:
            New CaptureVolume with filtered observations
        """
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

        if min_per_camera < 1:
            raise ValueError(f"min_per_camera must be >= 1, got {min_per_camera}")

        report = self.reprojection_report
        raw_errors = report.raw_errors

        if scope == "per_camera":
            # Compute (100 - percentile)th percentile per camera
            thresholds: dict[int, float] = {}
            for cam_id in self.camera_array.posed_cameras.keys():
                camera_errors = raw_errors[raw_errors["cam_id"] == cam_id]["euclidean_error"]
                if len(camera_errors) > 0:
                    # Keep the best (100 - percentile) percent
                    keep_percentile = 100 - percentile
                    thresholds[cam_id] = float(np.percentile(camera_errors, keep_percentile))
                else:
                    thresholds[cam_id] = float(np.inf)  # No observations, keep nothing

        elif scope == "overall":
            # Compute global (100 - percentile)th percentile
            keep_percentile = 100 - percentile
            global_threshold = float(np.percentile(raw_errors["euclidean_error"], keep_percentile))
            thresholds = {cam_id: global_threshold for cam_id in self.camera_array.posed_cameras.keys()}

        else:
            raise ValueError(f"scope must be 'per_camera' or 'overall', got {scope}")

        return self._filter_by_reprojection_thresholds(thresholds, min_per_camera)

    def compute_volumetric_scale_accuracy(
        self,
        world_points: WorldPoints | None = None,
        image_points: ImagePoints | None = None,
    ) -> VolumetricScaleReport:
        """Compute multi-frame scale accuracy across the capture volume.

        Compares triangulated world points to their corresponding ground truth
        object positions (from obj_loc columns) at all frames where >=4 corners
        are visible. Uses ALL pairwise distances at each frame for robust
        statistical measurement.

        Returns:
            VolumetricScaleReport containing per-frame errors and aggregate metrics.
            Returns empty report if no valid frames exist (normal pre-alignment state).
        """
        source_image_points = self.image_points if image_points is None else image_points
        img_df = source_image_points.df
        world_df = self.world_points.df if world_points is None else world_points.df

        # Find sync_indices where obj_loc data exists
        # Check if obj_loc columns are present
        obj_loc_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        if not all(col in img_df.columns for col in obj_loc_cols):
            return VolumetricScaleReport.empty()

        # Filter to rows that have obj_loc x/y data (z may be NaN for planar boards)
        obj_loc_mask = ~img_df[["obj_loc_x", "obj_loc_y"]].isna().any(axis=1)
        frames_with_obj_loc = img_df[obj_loc_mask]["sync_index"].unique()

        if len(frames_with_obj_loc) == 0:
            return VolumetricScaleReport.empty()

        frame_errors: list[FrameScaleError] = []

        # Process each frame
        for sync_index in frames_with_obj_loc:
            img_subset = img_df[img_df["sync_index"] == sync_index]
            world_subset = world_df[world_df["sync_index"] == sync_index]

            if img_subset.empty or world_subset.empty:
                continue

            # Get unique point_ids with obj_loc data (drop duplicates from multi-camera observations)
            obj_points_df = img_subset[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]].drop_duplicates(
                subset=["point_id"]
            )

            # Merge world points with object locations by point_id
            merged = world_subset.merge(obj_points_df, on="point_id", how="inner")

            # Handle planar objects (z=0 or NaN)
            if merged["obj_loc_z"].isna().all():
                merged = merged.copy()
                merged["obj_loc_z"] = 0.0

            # Filter out any remaining NaN values
            valid_mask = ~merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].isna().any(axis=1)
            merged = merged[valid_mask]

            # Skip frames with <4 corners (spec's minimum threshold)
            if len(merged) < 4:
                continue

            # Count cameras contributing at this frame
            n_cameras_contributing = int(
                img_subset[img_subset["point_id"].isin(merged["point_id"])]["cam_id"].nunique()
            )

            # Extract arrays for scale accuracy computation
            world_points = merged[["x_coord", "y_coord", "z_coord"]].to_numpy()
            object_points = merged[["obj_loc_x", "obj_loc_y", "obj_loc_z"]].to_numpy()

            try:
                frame_error = compute_frame_scale_error(
                    world_points=world_points,
                    object_points=object_points,
                    sync_index=int(sync_index),
                    n_cameras_contributing=n_cameras_contributing,
                )
                frame_errors.append(frame_error)
            except ValueError as e:
                # Log but don't fail — some frames may have degenerate geometry
                logger.debug(f"Skipping sync_index {sync_index} due to error: {e}")
                continue

        # Return report (empty if no valid frames)
        return VolumetricScaleReport(frame_errors=tuple(frame_errors))

    def align_to_object(
        self,
        sync_index: int,
        world_points: WorldPoints | None = None,
        image_points: ImagePoints | None = None,
    ) -> "CaptureVolume":
        """
        Align the capture volume to real-world units using object point correspondences.

        Uses the 3D points triangulated at the given sync_index and their
        corresponding ground truth object positions (from obj_loc columns) to
        estimate a similarity transform that scales the reconstruction to real-world units.

        Note:
            Object coordinates (obj_loc_*) must be in real-world units (typically meters).
            For Charuco boards, this requires defining the board with square_length in meters.

        For planar Charuco boards, obj_loc_z may be missing and will be treated as 0.
        The obj_loc coordinates must be in the target units (typically meters).

        Args:
            sync_index: Frame index where object is visible and has obj_loc data

        Returns:
            New CaptureVolume with cameras and world points in object coordinate units

        Raises:
            ValueError: If insufficient valid correspondences (< 3 points) or missing data
        """
        # Extract data at sync_index
        source_image_points = self.image_points if image_points is None else image_points
        img_df = source_image_points.df
        source_world_points = self.world_points if world_points is None else world_points
        world_df = source_world_points.df

        img_subset = img_df[img_df["sync_index"] == sync_index]
        world_subset = world_df[world_df["sync_index"] == sync_index]

        if img_subset.empty:
            raise ValueError(f"No image observations at sync_index {sync_index}")
        if world_subset.empty:
            raise ValueError(f"No world points at sync_index {sync_index}")

        # Merge on point_id to find correspondences
        merged = pd.merge(
            world_subset[["point_id", "x_coord", "y_coord", "z_coord"]],
            img_subset[["point_id", "obj_loc_x", "obj_loc_y", "obj_loc_z"]],
            on="point_id",
            how="inner",
        )

        if len(merged) < 3:
            raise ValueError(f"Need at least 3 point correspondences at sync_index {sync_index}, got {len(merged)}")

        # Handle missing obj_loc_z (planar boards)
        if merged["obj_loc_z"].isna().all():
            logger.info("obj_loc_z is all NaN, assuming planar board with z=0")
            merged["obj_loc_z"] = 0.0

        # Filter out any rows with NaN object coordinates
        obj_cols = ["obj_loc_x", "obj_loc_y", "obj_loc_z"]
        valid_mask = ~merged[obj_cols].isna().any(axis=1)
        merged = merged[valid_mask]

        if len(merged) < 3:
            raise ValueError(
                f"Need at least 3 valid point correspondences at sync_index {sync_index}, "
                f"got {len(merged)} after filtering NaN values"
            )

        # Prepare source (triangulated) and target (object) points
        source_points = merged[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)
        target_points = merged[obj_cols].values.astype(np.float64)

        # Estimate and apply transform
        transform = estimate_similarity_transform(source_points, target_points)

        logger.info(
            f"Estimated alignment: scale={transform.scale:.6f}, "
            f"translation={transform.translation}, rotation_det={np.linalg.det(transform.rotation):.6f}"
        )

        new_camera_array, new_world_points = apply_similarity_transform(
            self.camera_array, source_world_points, transform
        )

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            _optimization_status=self._optimization_status,
        )

    @property
    def unique_sync_indices(self) -> np.ndarray:
        """
        Return sorted array of unique sync_index values present in world_points.

        Used for slider range in visualization widgets.
        """
        indices = self.world_points.df["sync_index"].unique()
        return np.sort(indices)

    def rotate(self, axis: Literal["x", "y", "z"], angle_degrees: float) -> "CaptureVolume":
        """
        Rotate the coordinate system around the specified axis.

        Uses right-hand rule: positive angle = counter-clockwise rotation
        when looking down the positive axis toward the origin.

        Transforms both camera extrinsics and world points, returning a new
        immutable CaptureVolume. The original remains unchanged.

        Args:
            axis: The axis to rotate around ("x", "y", or "z")
            angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            New CaptureVolume with rotated coordinate system.
        """
        angle_rad = np.radians(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Standard rotation matrices following right-hand rule
        if axis == "x":
            rotation = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ],
                dtype=np.float64,
            )
        elif axis == "y":
            rotation = np.array(
                [
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c],
                ],
                dtype=np.float64,
            )
        elif axis == "z":
            rotation = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")

        transform = SimilarityTransform(
            rotation=rotation,
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
        )

        new_camera_array, new_world_points = apply_similarity_transform(self.camera_array, self.world_points, transform)

        return CaptureVolume(
            camera_array=new_camera_array,
            image_points=self.image_points,
            world_points=new_world_points,
            _optimization_status=self._optimization_status,
        )


if __name__ == "__main__":
    from pathlib import Path
    from caliscope import __root__
    from caliscope.core.point_data import ImagePoints
    from caliscope.core.capture_volume import CaptureVolume
    from caliscope.cameras.camera_array import CameraArray

    # Load test data
    session_path = Path(__root__, "tests", "sessions", "larger_calibration_post_monocal")
    xy_path = session_path / "calibration" / "extrinsic" / "CHARUCO" / "xy_CHARUCO.csv"
    array_path = session_path / "camera_array.toml"

    image_points = ImagePoints.from_csv(xy_path)
    camera_array = CameraArray.from_toml(array_path)
    world_points = image_points.triangulate(camera_array)

    capture_volume = CaptureVolume(camera_array, image_points, world_points)

    # Inspect the reprojection report
    report = capture_volume.reprojection_report
