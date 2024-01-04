from pathlib import Path
from random import randint

import cv2 as cv
import numpy as np

from src.constants import (
    NEGATIVES_PATH,
    NEGATIVES_VALIDATION_PATH,
    POSITIVES_PATH,
    POSITIVES_VALIDATION_PATH,
    DIM_HOG_WINDOW,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    COLLAPSED_NUMPY_PATH,
    COLLAPSED_ANNOTATIONS_PATH,
    VALIDATION_NUMPY_PATH,
    VALIDATION_ANNOTATIONS_PATH,
)
from src.utils.helpers import check_if_dirs_exist
from src.utils.readers import get_annotations


def check_overlap(box, coordinates):
    xmin, ymin, xmax, ymax = box
    for coord in coordinates:
        cxmin, cymin, cxmax, cymax = coord
        if not (xmax < cxmin or xmin > cxmax or ymax < cymin or ymin > cymax):
            return False
    return True


def extract_positives(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]], path: Path
) -> None:
    for image_index in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_index)]
        for coordinates, character in annotations[image_index]:
            xmin, ymin, xmax, ymax = coordinates
            box = cv.resize(image[ymin:ymax, xmin:xmax], (DIM_HOG_WINDOW, DIM_HOG_WINDOW))
            cv.imwrite(f"{path}/{str(image_index).zfill(4)}_{counter}.jpg", box)
            counter += 1
            flipped_box = cv.flip(box, 1)
            cv.imwrite(f"{path}/{str(image_index).zfill(4)}_{counter}.jpg", flipped_box)
            counter += 1


def extract_negatives(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]], path: Path
) -> None:
    # TODO: REFACTOR THIS, IT IS ERROR PRONE TO RANGE RANDOM ERROR
    for image_index in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_index)]
        coordinates = [coord for coord, _ in annotations[image_index]]

        while counter < len(coordinates) * 2:
            x = randint(0, IMAGE_WIDTH - DIM_HOG_WINDOW)
            y = randint(0, IMAGE_HEIGHT - DIM_HOG_WINDOW)
            box = (x, y, x + DIM_HOG_WINDOW, y + DIM_HOG_WINDOW)
            if check_overlap(box, coordinates):
                cv.imwrite(
                    f"{path}/{str(image_index).zfill(4)}_{counter}.jpg".zfill(5),
                    image[y : y + DIM_HOG_WINDOW, x : x + DIM_HOG_WINDOW],
                )
                counter += 1


def extract_positives_and_negatives(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist([POSITIVES_PATH, NEGATIVES_PATH])
    extract_positives(images, annotations, POSITIVES_PATH)
    extract_negatives(images, annotations, NEGATIVES_PATH)


def extract_positives_and_negatives_validation(
    images: np.ndarray, annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist([POSITIVES_VALIDATION_PATH, NEGATIVES_VALIDATION_PATH])
    extract_positives(images, annotations, POSITIVES_VALIDATION_PATH)
    extract_negatives(images, annotations, NEGATIVES_VALIDATION_PATH)


def extract_cnn_images():
    train_images = np.load(COLLAPSED_NUMPY_PATH)
    annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)
    validation_images = np.load(VALIDATION_NUMPY_PATH)
    validation_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)
    extract_positives_and_negatives(train_images, annotations)
    extract_positives_and_negatives_validation(validation_images, validation_annotations)
