import numpy as np

from caliscope.cameras.camera_array import CameraData
from caliscope.gui.lens_model_visualizer import LensModelVisualizer


def _make_test_frame(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx = np.tile(x, (height, 1))
    yy = np.tile(y[:, None], (1, width))
    return np.dstack([xx, yy, np.full((height, width), 64, dtype=np.uint8)])


def test_fisheye_visualizer_matches_camera_native_undistortion() -> None:
    camera = CameraData(
        cam_id=0,
        size=(640, 480),
        matrix=np.array([[320.0, 0.0, 320.0], [0.0, 322.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        distortions=np.array([0.12, -0.18, 0.09, -0.03], dtype=np.float64),
        fisheye=True,
    )
    frame = _make_test_frame(*camera.size)

    visualizer = LensModelVisualizer(camera)

    assert visualizer.is_ready
    assert not visualizer.content_expands_beyond_frame
    assert np.array_equal(visualizer.undistort(frame), camera.undistort_frame(frame))


def test_standard_visualizer_matches_camera_native_undistortion() -> None:
    camera = CameraData(
        cam_id=0,
        size=(640, 480),
        matrix=np.array([[500.0, 0.0, 320.0], [0.0, 505.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        distortions=np.array([0.08, -0.04, 0.001, -0.002, 0.01], dtype=np.float64),
        fisheye=False,
    )
    frame = _make_test_frame(*camera.size)

    visualizer = LensModelVisualizer(camera)

    assert visualizer.is_ready
    assert not visualizer.content_expands_beyond_frame
    assert np.array_equal(visualizer.undistort(frame), camera.undistort_frame(frame))
