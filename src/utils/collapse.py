import os

import numpy as np

from src.utils.readers import get_images, get_annotations
from src.constants import (
    TRAIN_IMAGES,
    BARNEY_ANNOTATIONS_PATH,
    BETTY_ANNOTATIONS_PATH,
    FRED_ANNOTATIONS_PATH,
    WILMA_ANNOTATIONS_PATH,
    COLLAPSED_NUMPY_PATH,
    COLLAPSED_ANNOTATIONS_PATH,
    TRAIN_DATA_PATH,
)


def collapse():
    if not os.path.exists(TRAIN_DATA_PATH / "collapsed"):
        os.makedirs(TRAIN_DATA_PATH / "collapsed")

    images = get_images(TRAIN_IMAGES)
    np.save(COLLAPSED_NUMPY_PATH, images)

    with open(COLLAPSED_ANNOTATIONS_PATH, "w") as f:
        for i, annotation_path in enumerate(
            [BARNEY_ANNOTATIONS_PATH, BETTY_ANNOTATIONS_PATH, FRED_ANNOTATIONS_PATH, WILMA_ANNOTATIONS_PATH]
        ):
            annotations = get_annotations(annotation_path)
            new_annotations = {}
            for image_name, annotation in annotations.items():
                image_name_int = int(image_name.split(".")[0])
                image_name_int += i * 1000 - 1  # -1 because we start from 0 in the numpy array
                new_annotations[image_name_int] = annotation

            for image_index, annotation in new_annotations.items():
                for box, character in annotation:
                    f.write(f"{image_index} {box[0]} {box[1]} {box[2]} {box[3]} {character}\n")
