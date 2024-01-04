from pathlib import Path

# Hog dimensions
DIM_HOG_CELL = 3
WINDOW_SIZE = 40

# Hog parameters
PIXELS_PER_CELL = (DIM_HOG_CELL, DIM_HOG_CELL)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 20

# Detection score threshold
THRESHOLD = 0.5
# Intersection over union threshold
IOU_THRESHOLD = 0.1

# Train, validation and test images dimensions
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360

# Color - character mapping
COLOR_CHARACTER_MAPPING = {
    "barney": (204, 255, 204),
    "betty": (204, 0, 102),
    "fred": (255, 178, 102),
    "wilma": (106, 228, 242),
    "unknown": (0, 0, 255),
}

# Main paths
TRAIN_DATA_PATH = Path("../data/train")
VALIDATION_DATA_PATH = Path("../data/validation")
TEST_DATA_PATH = Path("../data/test")
MODEL_PATH = Path("../models")

# Images paths
BARNEY_IMAGES_PATH = TRAIN_DATA_PATH / "barney"
BETTY_IMAGES_PATH = TRAIN_DATA_PATH / "betty"
FRED_IMAGES_PATH = TRAIN_DATA_PATH / "fred"
WILMA_IMAGES_PATH = TRAIN_DATA_PATH / "wilma"

TRAIN_IMAGES_PATHS = (BARNEY_IMAGES_PATH, BETTY_IMAGES_PATH, FRED_IMAGES_PATH, WILMA_IMAGES_PATH)
VALIDATION_IMAGES_PATH = VALIDATION_DATA_PATH / "images"
COLLAPSED_IMAGES_PATH = TRAIN_DATA_PATH / "collapsed"

# Annotations paths
BARNEY_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "barney_annotations.txt"
BETTY_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "betty_annotations.txt"
FRED_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "fred_annotations.txt"
WILMA_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "wilma_annotations.txt"

TRAIN_ANNOTATIONS_PATHS = (
    BARNEY_ANNOTATIONS_PATH,
    BETTY_ANNOTATIONS_PATH,
    FRED_ANNOTATIONS_PATH,
    WILMA_ANNOTATIONS_PATH,
)
VALIDATION_ANNOTATIONS_PATH = VALIDATION_DATA_PATH / "validations_annotations.txt"
COLLAPSED_ANNOTATIONS_PATH = TRAIN_DATA_PATH / "collapsed_annotations.txt"

# Ground truth paths
TASK1_GT_PATH = Path("../data/validation/task1_gt_validation.txt")
TASK2_GT_BARNEY_PATH = Path("../data/validation/task2_barney_gt_validation.txt")
TASK2_GT_BETTY_PATH = Path("../data/validation/task2_betty_gt_validation.txt")
TASK2_GT_FRED_PATH = Path("../data/validation/task2_fred_gt_validation.txt")
TASK2_GT_WILMA_PATH = Path("../data/validation/task2_wilma_gt_validation.txt")

# Positives and negatives paths
TRAIN_PATCHES_PATH = Path("../data/train_images")
POSITIVES_PATH = Path("../data/train_images/positives")
NEGATIVES_PATH = Path("../data/train_images/negatives")

VALIDATION_PATCHES_PATH = Path("../data/validation_images")
POSITIVES_VALIDATION_PATH = Path("../data/validation_images/positives")
NEGATIVES_VALIDATION_PATH = Path("../data/validation_images/negatives")

# Solution paths
SOLUTION_PATH = Path("../solution/")
SOLUTION_TASK1_PATH = SOLUTION_PATH / "task1"
SOLUTION_TASK2_PATH = SOLUTION_PATH / "task2"

SOLUTION_DETECTIONS_PATH = SOLUTION_TASK1_PATH / "detections_all_faces.npy"
SOLUTION_SCORES_PATH = SOLUTION_TASK1_PATH / "scores_all_faces.npy"
SOLUTION_FILE_NAMES_PATH = SOLUTION_TASK1_PATH / "file_names_all_faces.npy"

SOLUTION_DETECTIONS_BARNEY_PATH = SOLUTION_TASK2_PATH / "detections_barney.npy"
SOLUTION_SCORES_BARNEY_PATH = SOLUTION_TASK2_PATH / "scores_barney.npy"
SOLUTION_FILE_NAMES_BARNEY_PATH = SOLUTION_TASK2_PATH / "file_names_barney.npy"

SOLUTION_DETECTIONS_BETTY_PATH = SOLUTION_TASK2_PATH / "detections_betty.npy"
SOLUTION_SCORES_BETTY_PATH = SOLUTION_TASK2_PATH / "scores_betty.npy"
SOLUTION_FILE_NAMES_BETTY_PATH = SOLUTION_TASK2_PATH / "file_names_betty.npy"

SOLUTION_DETECTIONS_FRED_PATH = SOLUTION_TASK2_PATH / "detections_fred.npy"
SOLUTION_SCORES_FRED_PATH = SOLUTION_TASK2_PATH / "scores_fred.npy"
SOLUTION_FILE_NAMES_FRED_PATH = SOLUTION_TASK2_PATH / "file_names_fred.npy"

SOLUTION_DETECTIONS_WILMA_PATH = SOLUTION_TASK2_PATH / "detections_wilma.npy"
SOLUTION_SCORES_WILMA_PATH = SOLUTION_TASK2_PATH / "scores_wilma.npy"
SOLUTION_FILE_NAMES_WILMA_PATH = SOLUTION_TASK2_PATH / "file_names_wilma.npy"
