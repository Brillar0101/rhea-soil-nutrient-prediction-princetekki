"""
Configuration for Rhea Soil Nutrient Prediction Challenge
"""
import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
SUBMISSION_DIR = os.path.join(ROOT_DIR, "submissions")

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, "Train.csv")
TEST_FILE = os.path.join(DATA_DIR, "TestSet.csv")
SAMPLE_DATES_FILE = os.path.join(DATA_DIR, "Sample_Collection_Dates.csv")
DATA_DICT_FILE = os.path.join(DATA_DIR, "data_dictionary.csv")
TARGET_KEEP_FILE = os.path.join(DATA_DIR, "TargetPred_To_Keep.csv")
SAMPLE_SUB_FILE = os.path.join(DATA_DIR, "SampleSubmission.csv")

# Target columns (13 nutrients) - names as in Train.csv
TARGETS = ["Al", "B", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "N", "Na", "P", "S", "Zn"]

# Submission target column names (with Target_ prefix)
SUB_TARGETS = [f"Target_{t}" for t in TARGETS]

# Extra columns in train that can be used as auxiliary features/targets
TRAIN_EXTRA_COLS = ["C_organic", "C_total", "electrical_conductivity", "ph"]

# Sparse targets (many entries in TargetPred_To_Keep are 0)
SPARSE_TARGETS = ["B", "Na", "P", "S", "Zn"]
DENSE_TARGETS = ["Al", "Ca", "Cu", "Fe", "K", "Mg", "Mn", "N"]

# Random seed for reproducibility
SEED = 42

# CV folds
N_FOLDS = 10

# GPU settings
USE_GPU = False  # Set True when running on GPU cluster
