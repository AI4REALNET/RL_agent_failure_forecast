import os
import sys
import subprocess
import time

# BUG FIX: Added TEST_SINGLE_EPISODE to the import statement to prevent NameError
from src.config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE, TEST_SINGLE_EPISODE


def execute_module(module_path: str) -> None:
    """
    Executes a Python module as a subprocess, ensuring the project root
    is correctly appended to the PYTHONPATH to prevent module resolution errors.

    Args:
        module_path (str): The relative path to the Python script to execute.

    Raises:
        SystemExit: If the subprocess fails, it exits with the same error code.
    """
    print(f"\n{'=' * 60}")
    print(f" EXECUTING MODULE: {module_path}")
    print(f"{'=' * 60}\n")

    # Set up the environment variables to ensure proper module resolution
    env = os.environ.copy()
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    # Prepend project directories to PYTHONPATH for reliable absolute imports
    python_path = f"{current_directory}{os.pathsep}{parent_directory}"
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")

    start_time = time.time()

    # Building the command as a list avoids errors if paths contain spaces
    command = [sys.executable, module_path]

    try:
        # check=True automatically raises CalledProcessError if returncode != 0
        subprocess.run(command, check=True, env=env)
        elapsed_time = time.time() - start_time
        print(f"\n SUCCESS: {module_path} completed in {elapsed_time:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n ERROR: {module_path} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


def run_training_pipeline() -> None:
    """
    Manages the execution flow of the full training pipeline, checking
    for existing artifacts to avoid redundant and expensive computations.
    """
    print("\n[MODE] INITIALIZING FULL TRAINING PIPELINE")

    # Step 1: Train Forecasters
    if not os.path.exists(CFG.MODEL_MEAN_PATH):
        execute_module("src/train_forecast.py")
    else:
        print(f" SKIP: Forecast model already exists at {CFG.MODEL_MEAN_PATH}")

    # Step 2: Train ENN (Episode Neural Network)
    if not os.path.exists(CFG.MODEL_ENN_PATH):
        execute_module("src/training_enn.py")
    else:
        print(f" SKIP: ENN model already exists at {CFG.MODEL_ENN_PATH}")

    # Step 3: Collect Data for the Classifier
    if not os.path.exists(CFG.CSV_OUTPUT_PATH):
        execute_module("src/collect_data.py")
    else:
        print(f" SKIP: Data already collected at {CFG.CSV_OUTPUT_PATH}")

    # Step 4: Train Classifier (Always runs in this pipeline based on original logic)
    execute_module("src/train_classifier.py")


def main() -> None:
    """
    Main entry point for the SEST orchestrator. Routes execution based on
    the configuration flags defined in src.config.
    """
    # Defensive programming: provide a fallback if ENV_NAME is missing
    environment_name = getattr(CFG, 'ENV_NAME', 'UNKNOWN')
    print(f" CONFIGURATION DETECTED: ENV={environment_name}")

    # Route execution based on configured mode
    if TRAIN_MODE:
        run_training_pipeline()
    elif TEST_SINGLE_EPISODE:
        execute_module("src/collect_data_1.py")
    elif PREDICT_PROBA_MODE:
        execute_module("src/train_classifier.py")
    else:
        print("\n WARNING: No active execution mode selected in src/config.py.")


if __name__ == "__main__":
    main()