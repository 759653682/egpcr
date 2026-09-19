from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "custom_dataset_artifactsV2"

SCALER_FILENAME = "scaler_custom_dataset.joblib"
MODEL_FILENAME = "model_custom_dataset.pth"

TRAIN_DATA_FILENAME = "X_norm1.txt"
TEST_DATA_FILENAME = "X_fault2.txt"
COLUMNS_TO_DELETE_ORIG = []
FAULT_IDS = [2]
MAX_TEST_SAMPLES = 300
N_NORMAL_SAMPLES = 105
NUM_FEATURES = 12

EXPERIMENT_MODE = "denoising"
UNCERTAIN_VARIABLES_INDICES = [1, 3, 5, 8, 10]
UNCERTAINTY_LEVELS = [0.02, 0.03, 0.05, 0.02, 0.04]

BAYESIAN_EPSILON = 1e-12
GLOBAL_SEED = 20240521
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
