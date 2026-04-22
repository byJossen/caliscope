from caliscope.gui.presenters.intrinsic_calibration_presenter import _sample_collection_indices


def test_sample_collection_indices_honors_frame_skip_when_uncapped() -> None:
    assert _sample_collection_indices(20, frame_skip=5, max_frame_count=10) == [0, 5, 10, 15, 20]


def test_sample_collection_indices_caps_uniformly_over_full_video() -> None:
    indices = _sample_collection_indices(999, frame_skip=1, max_frame_count=5)

    assert len(indices) == 5
    assert indices[0] == 0
    assert indices[-1] == 999
    assert indices == sorted(indices)


def test_sample_collection_indices_allows_uncapped_collection() -> None:
    assert _sample_collection_indices(12, frame_skip=3, max_frame_count=None) == [0, 3, 6, 9, 12]
