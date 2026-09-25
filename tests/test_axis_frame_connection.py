"""A neighboring panel border must not become a zero-height axis frame."""

import cv2
import numpy as np
import pytest

from src.tools.extract import find_frame


@pytest.mark.parametrize("boxed", [False, True])
def test_lower_panel_border_at_same_x_is_not_bottom_axis(boxed):
    dark = np.zeros((290, 320), np.uint8)
    cv2.line(dark, (70, 40), (70, 215), 1, 1)
    cv2.line(dark, (70, 215), (290, 215), 1, 1)
    if boxed:
        cv2.line(dark, (70, 40), (290, 40), 1, 1)
        cv2.line(dark, (290, 40), (290, 215), 1, 1)
    # The next panel begins below a white label gutter at the identical x.
    cv2.line(dark, (70, 270), (290, 270), 1, 1)
    cv2.line(dark, (70, 270), (70, 289), 1, 1)
    assert find_frame(dark) == (70, 215, 40, 290)


def test_inset_border_without_horizontal_connection_is_not_axis():
    dark = np.zeros((260, 320), np.uint8)
    cv2.line(dark, (60, 30), (60, 210), 1, 1)
    cv2.line(dark, (60, 210), (295, 210), 1, 1)
    cv2.rectangle(dark, (115, 75), (275, 175), 1, 1)
    assert find_frame(dark) == (60, 210, 30, 295)
