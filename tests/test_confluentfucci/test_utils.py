from __future__ import annotations

from pathlib import Path

import pytest

from confluentfucci import utils


@pytest.fixture()
def test_stack():
    path = Path('../resources/test_red_stack.tiff')

    yield path

    (path.parent / 'test_red_stack_segmented.h5').unlink(missing_ok=True)
    (path.parent / 'test_red_stack_segmented.tiff').unlink(missing_ok=True)
    (path.parent / 'test_red_stack_segmented.tiff.xml').unlink(missing_ok=True)


def test_read_stack(test_stack) -> None:
    assert utils.read_stack(test_stack).shape == (3, 342, 1212)


def test_analysis_pipeline(test_stack) -> None:
    model = Path('../../models/cellpose/nuclei_red_v2')

    utils.segment_stack(path=test_stack, model=model)

    segmented_path = test_stack.parent / 'test_red_stack_segmented.tiff'
    segmented = utils.read_stack(segmented_path)

    assert segmented.shape == (3, 342, 1212)
    assert segmented[0].max() == 41
    assert segmented[1].max() == 40
    assert segmented[2].max() == 39

    settings = Path('../../models/trackmate/basic_settings.xml')
    utils.run_trackmate(settings_path=settings, data_path=segmented_path)

    result_path = segmented_path.parent / 'test_red_stack_segmented.tiff.xml'
    assert result_path.exists()
