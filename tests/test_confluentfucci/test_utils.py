from __future__ import annotations

from pathlib import Path

import pytest

from confluentfucci import utils

tests_folder_path = Path(__file__).parent.parent


@pytest.fixture()
def test_stack():
    path = tests_folder_path / "resources/test_red_stack.tiff"

    yield path

    (path.parent / "test_red_stack_segmented.h5").unlink(missing_ok=True)
    (path.parent / "test_red_stack_segmented.tiff").unlink(missing_ok=True)


@pytest.fixture()
def test_segmented_stack():
    path = tests_folder_path / "resources/test_segmented_stack.tiff"

    yield path

    (path.parent / "test_red_stack_segmented.tiff.xml").unlink(missing_ok=True)


def test_read_stack(test_stack) -> None:
    assert utils.read_stack(test_stack).shape == (3, 342, 1212)


def test_segmentation(test_stack) -> None:
    model = tests_folder_path.parent / "models/cellpose/nuclei_red_v2"

    utils.segment_stack(path=test_stack, model=model)

    segmented_path = test_stack.parent / "test_red_stack_segmented.tiff"
    segmented = utils.read_stack(segmented_path)

    assert segmented.shape == (3, 342, 1212)
    assert segmented[0].max() == 41
    assert segmented[1].max() == 40
    assert segmented[2].max() == 38


@pytest.mark.skip("Currently not testable in cicd")
def test_trackmate(test_segmented_stack) -> None:
    settings = test_segmented_stack.parent.parent.parent / "models/trackmate/basic_settings.xml"
    utils.run_trackmate(settings_path=settings, data_path=test_segmented_stack)

    result_path = test_segmented_stack.parent / "test_segmented_stack.tiff.xml"
    assert result_path.exists()
