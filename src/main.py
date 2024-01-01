import pickle
import timeit

import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage.transform import resize

from src.constants import (
    VALIDATION_ANNOTATIONS_PATH,
    VALIDATION_NUMPY_PATH,
    PIXELS_PER_CELL,
    CELLS_PER_BLOCK,
    ORIENTATIONS,
    THRESHOLD,
    DIM_HOG_CELL,
    DIM_HOG_WINDOW,
    MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SOLUTION_DETECTIONS_PATH,
    SOLUTION_SCORES_PATH,
    SOLUTION_FILE_NAMES_PATH,
)
from src.utils.helpers import show_image, intersection_over_union
from src.utils.readers import get_annotations


def non_maximal_suppression(image_detections, image_scores, image_size):
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True:
                    if intersection_over_union(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if (
                            sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2]
                            and sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]
                        ):
                            is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]


def run():
    # Initialize the scales that we will use to resize the image
    SCALES = [1, 0.5, 0.3, 0.2]

    # Load the classifier
    model = pickle.load(open(MODEL_PATH / "model.pkl", "rb"))
    weights = model.coef_.T
    bias = model.intercept_[0]

    # Load the validation images
    validation_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in np.load(VALIDATION_NUMPY_PATH)]

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
                    if score > THRESHOLD:
                        x_min = int(x * DIM_HOG_CELL / scale)
                        y_min = int(y * DIM_HOG_CELL / scale)
                        x_max = int((x * DIM_HOG_CELL + DIM_HOG_WINDOW) / scale)
                        y_max = int((y * DIM_HOG_CELL + DIM_HOG_WINDOW) / scale)
                        image_detections.append([x_min, y_min, x_max, y_max])
                        image_scores.append(score)
        if len(image_scores) > 0:
            image_detections, image_scores = non_maximal_suppression(
                np.array(image_detections), np.array(image_scores), image.shape
            )
        if len(image_scores) > 0:
            if detections is None:
                detections = image_detections
            else:
                detections = np.concatenate((detections, image_detections))
            scores = np.append(scores, image_scores)
            file_names = np.append(file_names, np.repeat(image_name, len(image_scores)))

        end_time = timeit.default_timer()
        print(f"Process time for {i}/{len(validation_images)} was {end_time - start_time} seconds.")

    return detections, scores, file_names


if __name__ == "__main__":
    # Collapse the images and annotations, save numpys
    # save_train_images_numpy()
    # save_validation_images_numpy()
    # collapse()

    # Initialize the annotations and images
    # train_images = np.load(COLLAPSED_NUMPY_PATH)
    # annotations = get_annotations(COLLAPSED_ANNOTATIONS_PATH)
    # validation_images = np.load(VALIDATION_NUMPY_PATH)
    # validation_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)

    # Generate the positives and negatives
    # extract_positives_and_negatives(train_images, annotations)
    # extract_positives_and_negatives_validation(validation_images, validation_annotations)

    # RUN
    ddetections, sscores, ffile_names = run()

    np.save(SOLUTION_DETECTIONS_PATH, ddetections)
    np.save(SOLUTION_SCORES_PATH, sscores)
    np.save(SOLUTION_FILE_NAMES_PATH, ffile_names)

    # images = np.load(VALIDATION_NUMPY_PATH)
    # annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)
    #
    # for ffile_name, ddetection, score in zip(ffile_names, ddetections, sscores):
    #     ffile_name = ffile_name.split(".")[0]
    #     image = images[int(ffile_name)]
    #     try:
    #         cv.rectangle(image, (ddetection[0], ddetection[1]), (ddetection[2], ddetection[3]), (0, 255, 0), 2)
    #         cv.putText(
    #             image,
    #             "score:" + str(score)[:4],
    #             (ddetection[0], ddetection[1]),
    #             cv.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0, 255, 0),
    #             1,
    #         )
    #     except:
    #         print(ddetections)
    #     show_image(image)

    # val_positives = get_images(POSITIVES_VALIDATION_GLOB)
    # val_negatives = get_images(NEGATIVES_VALIDATION_GLOB)
    # val_positives = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in val_positives]
    # val_negatives = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in val_negatives]
    # val_pos_features = [
    #     hog(image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS)
    #     for image in val_positives
    # ]
    # val_neg_features = [
    #     hog(image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS)
    #     for image in val_negatives
    # ]
    # val_pos_features = np.array(val_pos_features)
    # val_neg_features = np.array(val_neg_features)
    # val_examples = np.concatenate((np.squeeze(val_pos_features), np.squeeze(val_neg_features)), axis=0)
    # val_labels = np.concatenate((np.ones(len(val_pos_features)), np.zeros(len(val_neg_features))))
    # predictions = model.predict(val_examples)
    # print(f"Accuracy: {accuracy_score(val_labels ,predictions)}")
