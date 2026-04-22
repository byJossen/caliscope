from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from caliscope.core.capture_volume import CaptureVolume
from caliscope.core.point_data import ImagePoints, WorldPoints
from caliscope.core.reprojection import project_world_to_image
from caliscope.recording.frame_source import FrameSource
from caliscope.tracker import Tracker
from caliscope.trackers.aruco_tracker import ArucoTracker

logger = logging.getLogger(__name__)


def export_reprojection_overlay_videos(
    recording_dir: Path,
    output_dir: Path,
    capture_volume: CaptureVolume,
    tracker: Tracker,
    *,
    image_points: ImagePoints | None = None,
    world_points: WorldPoints | None = None,
    progress_callback=None,
) -> None:
    """Export per-camera MP4s with raw detections and reprojected 3D target overlays."""
    source_image_points = capture_volume.image_points if image_points is None else image_points
    source_world_points = capture_volume.world_points if world_points is None else world_points

    if len(source_image_points.df) == 0 or len(source_world_points.df) == 0:
        logger.info("Skipping reprojection overlay video export: no image/world points available")
        return

    frame_sources = _open_frame_sources(recording_dir, capture_volume)
    if not frame_sources:
        logger.warning("Skipping reprojection overlay video export: no source videos available")
        return

    sync_indices = np.sort(source_world_points.df["sync_index"].unique().astype(np.int64))
    if len(sync_indices) == 0:
        logger.info("Skipping reprojection overlay video export: no sync indices available")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = _create_video_writers(output_dir, frame_sources, sync_indices)

    try:
        world_groups = {
            int(sync_index): group.copy()
            for sync_index, group in source_world_points.df.groupby("sync_index")
        }
        raw_groups = {
            (int(sync_index), int(cam_id)): group.copy()
            for (sync_index, cam_id), group in source_image_points.df.groupby(["sync_index", "cam_id"])
        }

        total_steps = max(len(sync_indices) * max(len(capture_volume.camera_array.posed_cameras), 1), 1)
        completed_steps = 0

        for sync_index in sync_indices:
            sync_world = world_groups.get(int(sync_index))
            if sync_world is None or len(sync_world) == 0:
                continue

            for cam_id, camera in capture_volume.camera_array.posed_cameras.items():
                completed_steps += 1
                if progress_callback is not None:
                    progress_callback(completed_steps, total_steps)

                frame_source = frame_sources.get(cam_id)
                writer = writers.get(cam_id)
                if frame_source is None or writer is None:
                    continue

                raw_group = raw_groups.get((int(sync_index), int(cam_id)))
                frame_index = _frame_index_for_sync(sync_index, raw_group)
                if frame_index is None:
                    continue

                frame = frame_source.read_frame_at(frame_index)
                if frame is None:
                    continue

                projected = _project_world_points(sync_world, camera)
                annotated = _draw_reprojection_overlay(
                    frame=frame,
                    tracker=tracker,
                    sync_index=int(sync_index),
                    frame_index=frame_index,
                    raw_points=raw_group,
                    projected_points=projected,
                )
                writer.write(annotated)
    finally:
        for writer in writers.values():
            writer.release()
        for frame_source in frame_sources.values():
            frame_source.close()


def _open_frame_sources(recording_dir: Path, capture_volume: CaptureVolume) -> dict[int, FrameSource]:
    frame_sources: dict[int, FrameSource] = {}
    for cam_id in capture_volume.camera_array.posed_cameras:
        try:
            frame_sources[cam_id] = FrameSource(recording_dir, cam_id)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"Could not open extrinsic video for cam_id {cam_id}: {exc}")
    return frame_sources


def _create_video_writers(
    output_dir: Path,
    frame_sources: dict[int, FrameSource],
    sync_indices: np.ndarray,
) -> dict[int, cv2.VideoWriter]:
    writers: dict[int, cv2.VideoWriter] = {}
    sync_step = _estimate_sync_step(sync_indices)

    for cam_id, frame_source in frame_sources.items():
        path = output_dir / f"cam_{cam_id}_reprojection.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # type: ignore[attr-defined]
        fps = max(frame_source.fps / max(sync_step, 1), 1.0)
        writers[cam_id] = cv2.VideoWriter(str(path), fourcc, fps, frame_source.size)
        logger.info(f"Writing reprojection overlay video for cam_id {cam_id} to {path}")

    return writers


def _estimate_sync_step(sync_indices: np.ndarray) -> int:
    if len(sync_indices) < 2:
        return 1
    diffs = np.diff(sync_indices)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1
    return int(max(1, round(float(np.median(positive)))))


def _frame_index_for_sync(sync_index: int, raw_group: pd.DataFrame | None) -> int | None:
    if raw_group is None or len(raw_group) == 0:
        return None
    if "frame_index" in raw_group.columns and not raw_group["frame_index"].isna().all():
        return int(raw_group["frame_index"].iloc[0])
    return int(sync_index)


def _project_world_points(sync_world: pd.DataFrame, camera) -> pd.DataFrame:
    if camera.rotation is None or camera.translation is None or camera.matrix is None or camera.distortions is None:
        return pd.DataFrame(columns=["point_id", "img_loc_x", "img_loc_y"])

    world_coords = sync_world[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)
    rvec, _ = cv2.Rodrigues(camera.rotation)
    projected = project_world_to_image(
        camera,
        world_coords,
        rvec=rvec.ravel(),
        tvec=camera.translation,
    )
    projected_df = pd.DataFrame(
        {
            "point_id": sync_world["point_id"].to_numpy(dtype=np.int64),
            "img_loc_x": projected[:, 0],
            "img_loc_y": projected[:, 1],
        }
    )
    finite_mask = np.isfinite(projected_df["img_loc_x"].to_numpy(dtype=np.float64)) & np.isfinite(
        projected_df["img_loc_y"].to_numpy(dtype=np.float64)
    )
    return projected_df.loc[finite_mask].reset_index(drop=True)


def _draw_reprojection_overlay(
    frame: np.ndarray,
    tracker: Tracker,
    sync_index: int,
    frame_index: int,
    raw_points: pd.DataFrame | None,
    projected_points: pd.DataFrame,
) -> np.ndarray:
    annotated = frame.copy()

    if raw_points is not None and len(raw_points) > 0:
        _draw_points(
            annotated,
            tracker,
            raw_points,
            edge_color_mode="marker",
            point_color_override=None,
            label_suffix="raw",
        )

    if len(projected_points) > 0:
        _draw_points(
            annotated,
            tracker,
            projected_points,
            edge_color_mode="projected",
            point_color_override=(0, 165, 255),
            label_suffix="proj",
        )

    _draw_legend(annotated)
    _draw_header(annotated, sync_index, frame_index)
    return annotated


def _draw_points(
    frame: np.ndarray,
    tracker: Tracker,
    points_df: pd.DataFrame,
    *,
    edge_color_mode: str,
    point_color_override: tuple[int, int, int] | None,
    label_suffix: str,
) -> None:
    coords_by_id = {
        int(row.point_id): (float(row.img_loc_x), float(row.img_loc_y))
        for row in points_df.itertuples(index=False)
        if np.isfinite(float(row.img_loc_x)) and np.isfinite(float(row.img_loc_y))
    }

    for point_a, point_b in tracker.get_connected_points():
        if point_a not in coords_by_id or point_b not in coords_by_id:
            continue
        color = _edge_color(point_a, point_b, projected=edge_color_mode == "projected")
        start = _pixel_point(coords_by_id[point_a])
        end = _pixel_point(coords_by_id[point_b])
        if start is None or end is None:
            continue
        try:
            cv2.line(frame, start, end, color, 2, lineType=cv2.LINE_AA)
        except cv2.error as exc:
            logger.debug(
                "Skipping invalid overlay line for points %s-%s (%s -> %s): %s",
                point_a,
                point_b,
                start,
                end,
                exc,
            )

    for point_id, (x, y) in coords_by_id.items():
        params = tracker.scatter_draw_instructions(point_id)
        color = point_color_override if point_color_override is not None else params["color"]
        center = _pixel_point((x, y))
        if center is None:
            continue
        try:
            cv2.circle(
                frame,
                center,
                int(params["radius"]),
                color,
                int(params["thickness"]),
                lineType=cv2.LINE_AA,
            )
        except cv2.error as exc:
            logger.debug("Skipping invalid overlay point %s at %s: %s", point_id, center, exc)

    if isinstance(tracker, ArucoTracker):
        _draw_marker_labels(frame, points_df, tracker, projected=edge_color_mode == "projected", label_suffix=label_suffix)


def _draw_marker_labels(
    frame: np.ndarray,
    points_df: pd.DataFrame,
    tracker: ArucoTracker,
    *,
    projected: bool,
    label_suffix: str,
) -> None:
    marker_points: dict[int, list[tuple[float, float]]] = {}
    for row in points_df.itertuples(index=False):
        x = float(row.img_loc_x)
        y = float(row.img_loc_y)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        marker_points.setdefault(int(row.point_id) // 10, []).append((x, y))

    for marker_id, coords in marker_points.items():
        if not coords:
            continue
        centroid = np.mean(np.asarray(coords, dtype=np.float32), axis=0)
        base_label = tracker.aruco_target.get_marker_label(marker_id) if tracker.aruco_target is not None else f"id_{marker_id}"
        prefix = "proj" if projected else "raw"
        label = f"{prefix}:{base_label}"
        anchor = _pixel_point((float(centroid[0]), float(centroid[1])))
        if anchor is None:
            continue
        origin = (anchor[0] + 6, anchor[1] - 6)
        color = (0, 165, 255) if projected else _marker_color(marker_id)
        try:
            cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)
        except cv2.error as exc:
            logger.debug("Skipping invalid overlay label %s at %s: %s", label, origin, exc)


def _pixel_point(coord: tuple[float, float]) -> tuple[int, int] | None:
    """Convert floating-point image coordinates to finite integer pixel positions."""
    x, y = coord
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if abs(x) > 10_000_000 or abs(y) > 10_000_000:
        return None
    return (int(round(float(x))), int(round(float(y))))


def _draw_legend(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (12, 44), (260, 108), (20, 20, 20), -1)
    cv2.rectangle(frame, (12, 44), (260, 108), (90, 90, 90), 1)
    cv2.line(frame, (24, 68), (52, 68), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "raw detections", (64, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (24, 92), (52, 92), (0, 165, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "reprojected 3D target", (64, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, lineType=cv2.LINE_AA)


def _draw_header(frame: np.ndarray, sync_index: int, frame_index: int) -> None:
    text = f"sync {sync_index} | frame {frame_index}"
    cv2.putText(frame, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, lineType=cv2.LINE_AA)


def _edge_color(point_a: int, point_b: int, *, projected: bool) -> tuple[int, int, int]:
    if projected:
        return (0, 165, 255)
    if point_a // 10 == point_b // 10:
        return _marker_color(point_a // 10)
    return (0, 255, 255)


def _marker_color(marker_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 255, 0),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    )
    return palette[marker_id % len(palette)]
