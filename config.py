import os
import torch
import numpy as np
from typing import List, Dict, Any

# ==============================================================================
# ENVIRONMENT SELECTION
# ==============================================================================
# l2rpn_case14_sandbox or l2rpn_icaps_2021_small
#ACTIVE_ENV: str = "l2rpn_case14_sandbox"
ACTIVE_ENV: str = "l2rpn_icaps_2021_small"
# ==============================================================================
# EXECUTION MODES
# ==============================================================================
# 1. Train models from scratch
TRAIN_MODE: bool = True

# 2. Test a single full episode simulation
TEST_SINGLE_EPISODE: bool = False

# 3. Predict failure probability for ALL lines given ONE specific observation
# (Does NOT simulate disconnection, purely inference)
PREDICT_PROBA_MODE: bool = False

# If using PREDICT_PROBA_MODE with the example script, define which episode/step
# to fetch a sample observation from (or you can feed your own object).
PROBA_TEST_EPISODE_ID: int = 50
PROBA_TEST_STEP: int = 50

# ==============================================================================
# GLOBAL PATHS & SETTINGS
# ==============================================================================
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
MODELS_FORECASTER: str = os.path.join(BASE_DIR, "forecasts")
AGENT_DIR: str = os.path.join(BASE_DIR, "agents")
ENN_DATA_DIR: str = os.path.join(MODELS_DIR, "enn_data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_FORECASTER, exist_ok=True)
os.makedirs(AGENT_DIR, exist_ok=True)
os.makedirs(ENN_DATA_DIR, exist_ok=True)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# NETWORK CONFIGURATIONS
# ==============================================================================

DEFAULT_ENN_TRAIN = {
    'num_episodes': 250,
    'max_steps': 1000,
    'batch_size': 128,
    'epochs': 500,
    'max_lr': 3e-4,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'edl_lambda': 0.2,
    'kl_annealing_steps': 1000
}

class Config:
    ENV_NAME: str
    NO_LOADS: int
    NO_GENS: int
    NO_LINES: int
    GEN_MAX: np.ndarray
    GEN_MIN: np.ndarray
    LINES_TO_TEST: List[str]
    ENN_INPUT_DIM: int
    ENN_NUM_CLASSES: int
    ENN_HIDDEN_DIM: int
    MODEL_MEAN_PATH: str
    MODEL_ALEATORIC_PATH: str
    MODEL_ENN_PATH: str
    MODEL_CLASSIFIER_PATH: str
    X_TRAIN_PATH: str
    Y_TRAIN_PATH: str
    CSV_OUTPUT_PATH: str = os.path.join(DATA_DIR, "uncertainty_disconnection_analysis.csv")

class Config14(Config):
    ENV_NAME = "l2rpn_case14_sandbox"
    NO_LOADS = 11
    NO_GENS = 6
    NO_LINES = 20
    ENN_INPUT_DIM = 467
    ENN_NUM_CLASSES = 111
    ENN_HIDDEN_DIM = 467
    GEN_MAX = np.array([100.0, 140.0, 100.0, 100.0, 100.0, 100.0])
    GEN_MIN = np.zeros(6)
    LINES_TO_TEST = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
    MODEL_MEAN_PATH = os.path.join(MODELS_FORECASTER, "HBGB_14.pkl")
    MODEL_ALEATORIC_PATH = os.path.join(MODELS_DIR, "HBGB_14_aleatoric.pkl")
    MODEL_ENN_PATH = os.path.join(MODELS_DIR, "enn_14.pth")
    MODEL_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "final_classifier_14.pkl")
    AGENT_PATH = os.path.join(AGENT_DIR, "./network14")
    X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_14.npy")
    Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_14.npy")
    ENN_TRAIN_CONFIG = DEFAULT_ENN_TRAIN.copy()

class Config36(Config):
    ENV_NAME = "l2rpn_icaps_2021_small"
    NO_LOADS = 37
    NO_GENS = 22
    NO_LINES = 59
    ENN_INPUT_DIM = 1363
    ENN_NUM_CLASSES = 250
    ENN_HIDDEN_DIM = 1363
    GEN_MAX = np.array([
        50.0, 67.2, 50.0, 250.0, 50.0, 33.6, 37.3, 37.3, 33.6, 74.7, 100.0,
        37.3, 37.3, 100.0, 74.7, 74.7, 150.0, 67.2, 74.7, 400.0, 300.0, 350.0
    ])
    GEN_MIN = np.zeros(22)
    LINES_TO_TEST = ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131", "39_41_121", "43_44_125", "44_45_126", "34_35_110", "54_58_154"]
    MODEL_MEAN_PATH = os.path.join(MODELS_FORECASTER, "HBGB_36.pkl")
    MODEL_ALEATORIC_PATH = os.path.join(MODELS_DIR, "HBGB_36_aleatoric.pkl")
    MODEL_ENN_PATH = os.path.join(MODELS_DIR, "enn_36.pth")
    MODEL_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "final_classifier_36.pkl")
    AGENT_PATH = os.path.join(AGENT_DIR, "./network36")
    X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_36.npy")
    Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train_36.npy")
    ENN_TRAIN_CONFIG = DEFAULT_ENN_TRAIN.copy()

if ACTIVE_ENV == "l2rpn_case14_sandbox":
    CFG = Config14
elif ACTIVE_ENV == "l2rpn_icaps_2021_small":
    CFG = Config36
else:
    raise ValueError(f"Environment {ACTIVE_ENV} not supported.")

ENN_PARAMS: Dict[str, int] = {
    'input_dim': CFG.ENN_INPUT_DIM,
    'num_classes': CFG.ENN_NUM_CLASSES,
    'hidden_dim': CFG.ENN_HIDDEN_DIM,
}
ENN_DROPOUT: float = 0.1