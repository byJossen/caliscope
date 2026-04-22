from caliscope.cameras.camera_array import CameraArray, CameraData


def test_camera_data_defaults_to_standard_lens_model() -> None:
    camera = CameraData(cam_id=0, size=(640, 480))

    assert camera.fisheye is False


def test_legacy_camera_toml_without_fisheye_defaults_to_standard(tmp_path) -> None:
    camera_array_path = tmp_path / "camera_array.toml"
    camera_array_path.write_text(
        """
[cameras.0]
cam_id = 0
size = [640, 480]
""".strip()
    )

    camera_array = CameraArray.from_toml(camera_array_path)

    assert camera_array.cameras[0].fisheye is False
