import pickle
import timeit

import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage.transform import resize

from src.constants import (
    PIXELS_PER_CELL,
    CELLS_PER_BLOCK,
    ORIENTATIONS,
    DIM_HOG_CELL,
    DIM_HOG_WINDOW,
    MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SOLUTION_DETECTIONS_PATH,
    SOLUTION_SCORES_PATH,
    SOLUTION_FILE_NAMES_PATH,
    VALIDATION_IMAGES_PATH,
)
from src.utils.helpers import non_maximal_suppression, write_solution
from src.utils.readers import get_images


def run_task1_hog():
    big_start_time = timeit.default_timer()
    # Initialize the scales that we will use to resize the image
    SCALES = [0.5, 0.4, 0.3, 0.2]

    # Load the classifier
    model = pickle.load(open(MODEL_PATH / "model.pkl", "rb"))
    weights = model.coef_.T
    bias = model.intercept_[0]

    # Load the validation images
    validation_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in get_images(VALIDATION_IMAGES_PATH)]

    # Initialize the detections, scores and file names
    detections = None
    scores = np.array([])
    file_names = np.array([])

    for i, image in enumerate(validation_images):
        start_time = timeit.default_timer()
        print(f"Processing image {i}/{len(validation_images)}...")
        image_scores = []
        image_detections = []
        image_name = str(i + 1).zfill(4) + ".jpg"

        for scale in SCALES:
            if scale * IMAGE_HEIGHT < DIM_HOG_WINDOW:
                break

            # Resize the image
            resized_image = resize(image, [IMAGE_HEIGHT * scale, IMAGE_WIDTH * scale])

            hog_descriptors = hog(
                resized_image,
                pixels_per_cell=PIXELS_PER_CELL,
                cells_per_block=CELLS_PER_BLOCK,
                orientations=ORIENTATIONS,
                feature_vector=False,
            )

            NUM_COLS = resized_image.shape[1] // DIM_HOG_CELL - 1
            NUM_ROWS = resized_image.shape[0] // DIM_HOG_CELL - 1
            NUM_CELL_IN_TEMPLATE = DIM_HOG_WINDOW // DIM_HOG_CELL - 1

            for y in range(0, NUM_ROWS - NUM_CELL_IN_TEMPLATE):
                for x in range(0, NUM_COLS - NUM_CELL_IN_TEMPLATE):
                    descr = hog_descriptors[y : y + NUM_CELL_IN_TEMPLATE, x : x + NUM_CELL_IN_TEMPLATE].flatten()
                    score = np.dot(descr, weights)[0] + bias
                    if score > 1:
                        x_min = int(x * DIM_HOG_CELL / scale)
                        y_min = int(y * DIM_HOG_CELL / scale)
                        x_max = int((x * DIM_HOG_CELL + DIM_HOG_WINDOW) / scale)
                        y_max = int((y * DIM_HOG_CELL + DIM_HOG_WINDOW) / scale)
                        image_detections.append([x_min, y_min, x_max, y_max])
                        image_scores.append(score)
        if len(image_scores) > 0:
            image_detections, image_scores = non_maximal_suppression(np.array(image_detections), np.array(image_scores))
        if len(image_scores) > 0:
            if detections is None:
                detections = image_detections
            else:
                detections = np.concatenate((detections, image_detections))
            scores = np.append(scores, image_scores)
            file_names = np.append(file_names, np.repeat(image_name, len(image_scores)))

        end_time = timeit.default_timer()
        print(f"Process time for {i}/{len(validation_images)} was {end_time - start_time} seconds.")

    write_solution(
        detections,
        SOLUTION_DETECTIONS_PATH,
        scores,
        SOLUTION_SCORES_PATH,
        file_names,
        SOLUTION_FILE_NAMES_PATH,
    )

    print(f"Total time: {timeit.default_timer() - big_start_time} seconds.")
