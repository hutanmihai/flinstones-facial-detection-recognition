import os
from pathlib import Path

import cv2 as cv
import numpy as np

########################################################################################################################
# CORE SHOW IMAGE FUNCTION
########################################################################################################################


def show_image(image: np.ndarray, title: str = "image") -> None:
    """
    Shows the image.
    :param title: the title of the window, default is "image"
    :param image: the image to show
    :return:
    """
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def check_if_dirs_exist(paths: list[Path]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou


def compute_average_precision(recall, precision):
    m_rec = np.concatenate(([0], recall, [1]))
    m_pre = np.concatenate(([0], precision, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision
