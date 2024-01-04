import cv2 as cv
import numpy as np

from src.constants import (
    COLOR_CHARACTER_MAPPING,
    SOLUTION_DETECTIONS_PATH,
    SOLUTION_SCORES_PATH,
    SOLUTION_FILE_NAMES_PATH,
    VALIDATION_NUMPY_PATH,
    VALIDATION_ANNOTATIONS_PATH,
)
from src.utils.helpers import show_image
from src.utils.readers import get_annotations


def visualize_images_with_boxes(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
):
    for image_index in annotations.keys():
        image: np.ndarray = images[int(image_index)]
        for coordinates, character in annotations[image_index]:
            xmin, ymin, xmax, ymax = coordinates
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_CHARACTER_MAPPING[character], 1)
        show_image(image, image_index)


def visualize_images_with_boxes_and_detections():
    detections = np.load(SOLUTION_DETECTIONS_PATH)
    scores = np.load(SOLUTION_SCORES_PATH)
    file_names = np.load(SOLUTION_FILE_NAMES_PATH)

    images = np.load(VALIDATION_NUMPY_PATH)
    annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)

    for k, v in annotations.items():
        for bbox, _ in v:
            cv.rectangle(images[int(k.split(".")[0])], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    for file_name, detection, score in zip(file_names, detections, scores):
        file_name = file_name.split(".")[0]
        image = images[int(file_name)]
        cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 255, 0), 2)
        cv.putText(
            image,
            "score:" + str(score)[:4],
            (detection[0], detection[1]),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        show_image(image)
