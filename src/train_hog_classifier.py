import pickle
import timeit

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from src.constants import (
    PIXELS_PER_CELL,
    CELLS_PER_BLOCK,
    ORIENTATIONS,
    MODEL_PATH,
    POSITIVES_PATH,
    NEGATIVES_PATH,
    POSITIVES_VALIDATION_PATH,
    NEGATIVES_VALIDATION_PATH,
)
from src.utils.helpers import check_if_dirs_exist
from src.utils.readers import get_images


def get_positive_descriptors():
    images = get_images(POSITIVES_PATH)
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


def get_negatives_descriptors():
    images = get_images(NEGATIVES_PATH)
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
    print("Training classifier...")
    start_time = timeit.default_timer()
    positive_features = get_positive_descriptors()
    negative_features = get_negatives_descriptors()
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    model = LinearSVC(
        dual=True,
        C=0.01,
    )
    model.fit(training_examples, train_labels)
    acc = model.score(training_examples, train_labels)
    print(f"Accuracy: {acc}")
    print(f"Training took {timeit.default_timer() - start_time} seconds.\n")
    check_if_dirs_exist([MODEL_PATH])
    pickle.dump(model, open(MODEL_PATH / "model.pkl", "wb"))
    scores = model.decision_function(training_examples)

    # Visualize how are the scores distributed for positive and negative examples
    positive_scores = scores[train_labels > 0]
    negative_scores = scores[train_labels <= 0]

    plt.figure(figsize=(8, 6))
    plt.hist(positive_scores, bins=50, alpha=0.5, label="Positive Scores", color="blue")
    plt.hist(negative_scores, bins=50, alpha=0.5, label="Negative Scores", color="red")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Distribution of Scores for Positive and Negative Examples")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_classifier():
    print("Testing classifier...")
    start_time = timeit.default_timer()

    val_positives = get_images(POSITIVES_VALIDATION_PATH)
    val_negatives = get_images(NEGATIVES_VALIDATION_PATH)
    val_positives = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in val_positives]
    val_negatives = [cv.cvtColor(image, cv.COLOR_BGR2GRAY) for image in val_negatives]
    val_pos_features = [
        hog(image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS)
        for image in val_positives
    ]
    val_neg_features = [
        hog(image, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS)
        for image in val_negatives
    ]
    val_pos_features = np.array(val_pos_features)
    val_neg_features = np.array(val_neg_features)
    val_examples = np.concatenate((np.squeeze(val_pos_features), np.squeeze(val_neg_features)), axis=0)
    val_labels = np.concatenate((np.ones(len(val_pos_features)), np.zeros(len(val_neg_features))))

    model = pickle.load(open(MODEL_PATH / "model.pkl", "rb"))
    predictions = model.predict(val_examples)
    print(f"Validation Accuracy: {accuracy_score(val_labels ,predictions)}")

    print(f"Testing took {timeit.default_timer() - start_time} seconds.\n")


def train_hog_classifier():
    train_classifier()
    test_classifier()
