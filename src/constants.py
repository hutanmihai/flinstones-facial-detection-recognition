from glob import glob
from pathlib import Path

# Main paths
TRAIN_DATA_PATH = Path("../data/train")
VALIDATION_DATA_PATH = Path("../data/validation")
TEST_DATA_PATH = Path("../data/test")

# Model paths
MODEL_PATH = Path("../models")

# Images paths
BARNEY_IMAGES_PATH = TRAIN_DATA_PATH / "barney"
BETTY_IMAGES_PATH = TRAIN_DATA_PATH / "betty"
FRED_IMAGES_PATH = TRAIN_DATA_PATH / "fred"
WILMA_IMAGES_PATH = TRAIN_DATA_PATH / "wilma"
VALIDATION_IMAGES_PATH = VALIDATION_DATA_PATH / "images"
NUMPY_PATH = Path("../data/numpy")

# Sorted Images globs
BARNEY_IMAGES = sorted(glob(str(BARNEY_IMAGES_PATH / "*.jpg")))
BETTY_IMAGES = sorted(glob(str(BETTY_IMAGES_PATH / "*.jpg")))
FRED_IMAGES = sorted(glob(str(FRED_IMAGES_PATH / "*.jpg")))
WILMA_IMAGES = sorted(glob(str(WILMA_IMAGES_PATH / "*.jpg")))

TRAIN_IMAGES = BARNEY_IMAGES + BETTY_IMAGES + FRED_IMAGES + WILMA_IMAGES
VALIDATION_IMAGES = sorted(glob(str(VALIDATION_IMAGES_PATH / "*.jpg")))

# Solution paths
SOLUTION_PATH = Path("../solution/")
SOLUTION_TASK1_PATH = SOLUTION_PATH / "task1"

SOLUTION_DETECTIONS_PATH = SOLUTION_TASK1_PATH / "detections_all_faces.npy"
SOLUTION_SCORES_PATH = SOLUTION_TASK1_PATH / "scores_all_faces.npy"
SOLUTION_FILE_NAMES_PATH = SOLUTION_TASK1_PATH / "file_names_all_faces.npy"

# Ground truth paths
TASK1_GT_PATH = Path("../data/validation/task1_gt_validation.txt")
TASK2_GT_BARNEY_PATH = Path("../data/validation/task2_barney_gt_validation.txt")
TASK2_GT_BETTY_PATH = Path("../data/validation/task2_betty_gt_validation.txt")
TASK2_GT_FRED_PATH = Path("../data/validation/task2_fred_gt_validation.txt")

# Annotations paths
BARNEY_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "barney_annotations.txt"
BETTY_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "betty_annotations.txt"
FRED_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "fred_annotations.txt"
WILMA_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "wilma_annotations.txt"

TRAIN_ANNOTATIONS_PATH = (
    BARNEY_ANNOTATIONS_PATH,
    BETTY_ANNOTATIONS_PATH,
    FRED_ANNOTATIONS_PATH,
    WILMA_ANNOTATIONS_PATH,
)
VALIDATION_ANNOTATIONS_PATH = VALIDATION_DATA_PATH / "validations_annotations.txt"

# Positives and negatives paths
CNN_TRAIN_IMAGES_PATH = Path("../data/train_images")
POSITIVES_PATH = Path("../data/train_images/positives")
NEGATIVES_PATH = Path("../data/train_images/negatives")

CNN_VALIDATION_IMAGES_PATH = Path("../data/validation_images")
POSITIVES_VALIDATION_PATH = Path("../data/validation_images/positives")
NEGATIVES_VALIDATION_PATH = Path("../data/validation_images/negatives")

# Positives and negatives globs
POSITIVES_GLOB = sorted(glob(str(POSITIVES_PATH / "*.jpg")))
NEGATIVES_GLOB = sorted(glob(str(NEGATIVES_PATH / "*.jpg")))

POSITIVES_VALIDATION_GLOB = sorted(glob(str(POSITIVES_VALIDATION_PATH / "*.jpg")))
NEGATIVES_VALIDATION_GLOB = sorted(glob(str(NEGATIVES_VALIDATION_PATH / "*.jpg")))

# Collapsed paths
COLLAPSED_NUMPY_PATH = Path("../data/numpy/train_images.npy")
COLLAPSED_ANNOTATIONS_PATH = Path("../data/train/collapsed_annotations.txt")

VALIDATION_NUMPY_PATH = Path("../data/numpy/validation_images.npy")

# Color - character mapping
COLOR_CHARACTER_MAPPING = {
    "barney": (204, 255, 204),
    "betty": (204, 0, 102),
    "fred": (255, 178, 102),
    "wilma": (106, 228, 242),
    "unknown": (0, 0, 255),
}

# Hog dimensions
DIM_HOG_CELL = 3
DIM_HOG_WINDOW = 40

# Hog parameters
PIXELS_PER_CELL = (DIM_HOG_CELL, DIM_HOG_CELL)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 20

# Detection score threshold
THRESHOLD = 0.75

# Train, validation and test images dimensions
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360
