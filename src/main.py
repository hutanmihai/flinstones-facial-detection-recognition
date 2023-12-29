import os
import timeit

import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from src.constants import (
    COLLAPSED_ANNOTATIONS_PATH,
    COLLAPSED_NUMPY_PATH,
    NEGATIVES_GLOB,
    POSITIVES_GLOB,
    VALIDATION_ANNOTATIONS_PATH,
    VALIDATION_NUMPY_PATH,
    POSITIVES_VALIDATION_GLOB,
    NEGATIVES_VALIDATION_GLOB,
    VALIDATION_DATA_PATH,
    PIXELS_PER_CELL,
    CELLS_PER_BLOCK,
    ORIENTATIONS,
    THRESHOLD,
    DIM_HOG_CELL,
    DIM_HOG_WINDOW,
    NUM_ROWS,
    NUM_CELL_IN_TEMPLATE,
    NUM_COLS,
)
from src.utils.readers import get_annotations, get_images


BEST_MODEL = None


def get_positive_descriptors():
    images = get_images(POSITIVES_GLOB)
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(
            image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS
        )
        descriptors.append(features)

        features = hog(
            np.fliplr(image),
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            orientations=ORIENTATIONS,
        )
        descriptors.append(features)

    descriptors = np.array(descriptors)
    return descriptors


def get_negatives_descriptors():
    images = get_images(NEGATIVES_GLOB)
    # Convert to grayscale
    images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in images]
    descriptors = []
    for image in images:
        features = hog(
            image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS
        )
        descriptors.append(features)

    descriptors = np.array(descriptors)
    return descriptors


def train_classifier():
    positive_features = get_positive_descriptors()
    negative_features = get_negatives_descriptors()
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    model = LinearSVC(dual=True)
    model.fit(training_examples, train_labels)
    acc = model.score(training_examples, train_labels)
    print(f"Accuracy: {acc}")
    return model


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


def non_maximal_suppression(image_detections, image_scores, image_size):
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if (
                    is_maximal[j] == True
                ):  # don't change to 'is True' because is a numpy True and is not a python True :)
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
    validation_images = np.load(VALIDATION_NUMPY_PATH)
    validation_images = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in validation_images]
    validation_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)
    detections = None
    scores = np.array([])  # array cu toate scorurile pe care le obtinem
    file_names = np.array(
        []
    )  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
    # detectie din imagine, numele imaginii va aparea in aceasta lista
    w = BEST_MODEL.coef_.T
    bias = BEST_MODEL.intercept_[0]
    for i, image in enumerate(validation_images):
        start_time = timeit.default_timer()
        print("Procesam imaginea de testare %d/%d.." % (i, len(validation_images)))
        image_scores = []
        image_detections = []
        hog_descriptors = hog(
            image,
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            orientations=ORIENTATIONS,
            feature_vector=False,
        )

        for y in range(0, NUM_ROWS - NUM_CELL_IN_TEMPLATE):
            for x in range(0, NUM_COLS - NUM_CELL_IN_TEMPLATE):
                descr = hog_descriptors[y : y + NUM_CELL_IN_TEMPLATE, x : x + NUM_CELL_IN_TEMPLATE].flatten()
                score = np.dot(descr, w)[0] + bias
                if score > THRESHOLD:
                    x_min = int(x * DIM_HOG_CELL)
                    y_min = int(y * DIM_HOG_CELL)
                    x_max = int(x * DIM_HOG_CELL + DIM_HOG_WINDOW)
                    y_max = int(y * DIM_HOG_CELL + DIM_HOG_WINDOW)
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
            short_name = "random"
            image_names = [short_name for ww in range(len(image_scores))]
            file_names = np.append(file_names, image_names)

        end_time = timeit.default_timer()
        print(
            "Timpul de procesarea al imaginii de testare %d/%d este %f sec."
            % (i, len(validation_images), end_time - start_time)
        )

    return detections, scores, file_names


def compute_average_precision(rec, prec):
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision


def eval_detections(detections, scores, file_names):
    gt_annotations = get_annotations(VALIDATION_ANNOTATIONS_PATH)
    ground_truth_file_names = np.array(gt_annotations.keys())
    ground_truth_detections = []
    for file_name in gt_annotations.keys():
        for bbox, character in gt_annotations[file_name]:
            ground_truth_detections.append(bbox)
    ground_truth_detections = np.array(ground_truth_detections)

    num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
    gt_exists_detection = np.zeros(num_gt_detections)
    # sorteazam detectiile dupa scorul lor
    sorted_indices = np.argsort(scores)[::-1]
    file_names = file_names[sorted_indices]
    scores = scores[sorted_indices]
    detections = detections[sorted_indices]

    num_detections = len(detections)
    true_positive = np.zeros(num_detections)
    false_positive = np.zeros(num_detections)
    duplicated_detections = np.zeros(num_detections)

    for detection_idx in range(num_detections):
        indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

        gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
        bbox = detections[detection_idx]
        max_overlap = -1
        index_max_overlap_bbox = -1
        for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
            overlap = intersection_over_union(bbox, gt_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                index_max_overlap_bbox = indices_detections_on_image[gt_idx]

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if max_overlap >= 0.3:
            if gt_exists_detection[index_max_overlap_bbox] == 0:
                true_positive[detection_idx] = 1
                gt_exists_detection[index_max_overlap_bbox] = 1
            else:
                false_positive[detection_idx] = 1
                duplicated_detections[detection_idx] = 1
        else:
            false_positive[detection_idx] = 1

    cum_false_positive = np.cumsum(false_positive)
    cum_true_positive = np.cumsum(true_positive)

    rec = cum_true_positive / num_gt_detections
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    print("Average precision: %.3f" % average_precision)
    plt.plot(rec, prec, "-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Average precision %.3f" % average_precision)
    plt.savefig(os.path.join(VALIDATION_DATA_PATH, "precizie_medie.png"))
    plt.show()


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

    # Train the classifier
    model = train_classifier()
    BEST_MODEL = model
    ddetections, sscores, ffile_names = run()

    eval_detections(ddetections, sscores, ffile_names)
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

    # Visualize data
    # visualize_images_with_boxes(train_images, annotations)
    # visualize_images_with_boxes(validation_images, validation_annotations)
