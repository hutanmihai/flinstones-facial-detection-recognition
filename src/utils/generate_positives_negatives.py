import timeit
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
    COLLAPSED_ANNOTATIONS_PATH,
    VALIDATION_ANNOTATIONS_PATH,
    VALIDATION_IMAGES_PATH,
    COLLAPSED_IMAGES_PATH,
)
from src.utils.helpers import check_if_dirs_exist
from src.utils.readers import get_annotations, get_images


def check_overlap(box, coordinates):
    xmin, ymin, xmax, ymax = box
    for coord in coordinates:
        cxmin, cymin, cxmax, cymax = coord
        if not (xmax < cxmin or xmin > cxmax or ymax < cymin or ymin > cymax):
            return False
    return True


def extract_positives(
    images: list[np.ndarray], annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]], path: Path
) -> None:
    for image_name in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_name.split(".")[0]) - 1]
        for coordinates, character in annotations[image_name]:
            xmin, ymin, xmax, ymax = coordinates
            box = cv.resize(image[ymin:ymax, xmin:xmax], (DIM_HOG_WINDOW, DIM_HOG_WINDOW))
            cv.imwrite(f"{path}/{image_name.rstrip('.jpg')}_{counter}.jpg", box)
            counter += 1
            flipped_box = cv.flip(box, 1)
            cv.imwrite(f"{path}/{image_name.rstrip('.jpg')}_{counter}.jpg", flipped_box)
            counter += 1


def extract_negatives(
    images: list[np.ndarray], annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]], path: Path
) -> None:
    # TODO: REFACTOR THIS, IT IS ERROR PRONE TO RANGE RANDOM ERROR
    for image_name in annotations.keys():
        counter = 0
        image: np.ndarray = images[int(image_name.split(".")[0]) - 1]
        coordinates = [coord for coord, _ in annotations[image_name]]

        while counter < len(coordinates) * 2:
            x = randint(0, IMAGE_WIDTH - DIM_HOG_WINDOW)
            y = randint(0, IMAGE_HEIGHT - DIM_HOG_WINDOW)
            box = (x, y, x + DIM_HOG_WINDOW, y + DIM_HOG_WINDOW)
            if check_overlap(box, coordinates):
                cv.imwrite(
                    f"{path}/{image_name.rstrip('.jpg')}_{counter}.jpg".zfill(5),
                    image[y : y + DIM_HOG_WINDOW, x : x + DIM_HOG_WINDOW],
                )
                counter += 1


def extract_positives_and_negatives(
    images: list[np.ndarray], annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist([POSITIVES_PATH, NEGATIVES_PATH])
    extract_positives(images, annotations, POSITIVES_PATH)
    extract_negatives(images, annotations, NEGATIVES_PATH)


def extract_positives_and_negatives_validation(
    images: list[np.ndarray], annotations: dict[str, list[tuple[tuple[int, int, int, int], str]]]
) -> None:
    check_if_dirs_exist([POSITIVES_VALIDATION_PATH, NEGATIVES_VALIDATION_PATH])
    extract_positives(images, annotations, POSITIVES_VALIDATION_PATH)
    extract_negatives(images, annotations, NEGATIVES_VALIDATION_PATH)


def extract_train_and_validation_patches():
    print("Extracting positives and negatives for training and validation...")
    start_time = timeit.default_timer()
    train_images = get_images(COLLAPSED_IMAGES_PATH)
    annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)
    validation_images = get_images(VALIDATION_IMAGES_PATH)
    validation_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)
    extract_positives_and_negatives(train_images, annotations)
    extract_positives_and_negatives_validation(validation_images, validation_annotations)
    print("-" * 50)
    print("Successfully extracted positives and negatives for training and validation!")
    print(f"Extraction took {timeit.default_timer() - start_time} seconds.\n")
