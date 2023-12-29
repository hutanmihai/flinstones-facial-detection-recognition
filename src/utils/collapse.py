import numpy as np

from src.constants import (
    COLLAPSED_ANNOTATIONS_PATH,
    COLLAPSED_NUMPY_PATH,
    NUMPY_PATH,
    TRAIN_ANNOTATIONS_PATH,
    TRAIN_IMAGES,
    VALIDATION_IMAGES,
    VALIDATION_NUMPY_PATH,
)
from src.utils.helpers import check_if_dirs_exist
from src.utils.readers import get_annotations, get_images


def collapse():
    with open(COLLAPSED_ANNOTATIONS_PATH, "w") as f:
        for i, annotation_path in enumerate(TRAIN_ANNOTATIONS_PATH):
            annotations = get_annotations(annotation_path)
            new_annotations = {}
            for image_name, annotation in annotations.items():
                image_name_int = int(image_name.split(".")[0]) + 1
                image_name_int += i * 1000
                new_annotations[image_name_int] = annotation

            for image_index, annotation in new_annotations.items():
                image_index = str(image_index).zfill(4) + ".jpg"
                for box, character in annotation:
                    f.write(f"{image_index} {box[0]} {box[1]} {box[2]} {box[3]} {character}\n")


def save_train_images_numpy():
    check_if_dirs_exist([NUMPY_PATH])
    images = get_images(TRAIN_IMAGES)
    np.save(COLLAPSED_NUMPY_PATH, images)


def save_validation_images_numpy():
    check_if_dirs_exist([NUMPY_PATH])
    images = get_images(VALIDATION_IMAGES)
    np.save(VALIDATION_NUMPY_PATH, images)
