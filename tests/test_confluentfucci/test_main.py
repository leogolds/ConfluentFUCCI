from __future__ import annotations

from confluentfucci.main import get_hello


def it_prints_hi_to_the_project_author() -> None:
    expected = 'Hello, Leo Goldstien!'
    actual = get_hello('Leo Goldstien')
    assert actual == expected
