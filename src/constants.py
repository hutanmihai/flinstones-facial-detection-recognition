from pathlib import Path
from glob import glob

# Main paths
TRAIN_DATA_PATH = Path("../data/train")
VALIDATION_DATA_PATH = Path("../data/validation")
TEST_DATA_PATH = Path("../data/test")

# Images paths
BARNEY_IMAGES_PATH = TRAIN_DATA_PATH / "barney"
BETTY_IMAGES_PATH = TRAIN_DATA_PATH / "betty"
FRED_IMAGES_PATH = TRAIN_DATA_PATH / "fred"
WILMA_IMAGES_PATH = TRAIN_DATA_PATH / "wilma"
VALIDATION_IMAGES_PATH = VALIDATION_DATA_PATH / "images"

# Sorted Images globs
BARNEY_IMAGES = sorted(glob(str(BARNEY_IMAGES_PATH / "*.jpg")))
BETTY_IMAGES = sorted(glob(str(BETTY_IMAGES_PATH / "*.jpg")))
FRED_IMAGES = sorted(glob(str(FRED_IMAGES_PATH / "*.jpg")))
WILMA_IMAGES = sorted(glob(str(WILMA_IMAGES_PATH / "*.jpg")))

TRAIN_IMAGES = BARNEY_IMAGES + BETTY_IMAGES + FRED_IMAGES + WILMA_IMAGES
VALIDATION_IMAGES = sorted(glob(str(VALIDATION_IMAGES_PATH / "*.jpg")))

# Solution paths
TRAIN_SOLUTION_PATH = Path("../data/train_solution.txt")
TEST_SOLUTION_PATH = Path("../data/test_solution.txt")

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
VALIDATION_ANOTATIONS_PATH = VALIDATION_DATA_PATH / "validation_annotations.txt"

# Positives and negatives paths
POSITIVES_PATH = Path("../data/positives")
NEGATIVES_PATH = Path("../data/negatives")

# Positives and negatives globs
POSITIVES_GLOB = sorted(glob(str(POSITIVES_PATH / "*.jpg")))
NEGATIVES_GLOB = sorted(glob(str(NEGATIVES_PATH / "*.jpg")))

# Collapsed paths
COLLAPSED_NUMPY_PATH = Path("../data/train/collapsed/train_images.npy")
COLLAPSED_ANNOTATIONS_PATH = Path("../data/train/collapsed_annotations.txt")

# Color - character mapping
COLOR_CHARACTER_MAPPING = {
    "barney": (204, 255, 204),
    "betty": (204, 0, 102),
    "fred": (255, 178, 102),
    "wilma": (106, 228, 242),
    "unknown": (0, 0, 255),
}

# Image size
MAX_WIDTH = 480
MAX_HEIGHT = 360

# Positives Negatives constants
POS_NEG_WIDTH = 40
POS_NEG_HEIGHT = 40
