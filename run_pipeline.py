import os
import sys
import subprocess
import time

from src.config import CFG, TRAIN_MODE, PREDICT_PROBA_MODE, TEST_SINGLE_EPISODE

# LLM_RULE_MODE is the new flag for symbolic rule inference.
# If it does not yet exist in config.py, it defaults to False.
try:
    from src.config import LLM_RULE_MODE
except ImportError:
    LLM_RULE_MODE = False


# =============================================================================
# Subprocess executor (unchanged from original)
# =============================================================================

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

    env = os.environ.copy()
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    python_path       = f"{current_directory}{os.pathsep}{parent_directory}"
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")

    start_time = time.time()
    command    = [sys.executable, module_path]

    try:
        subprocess.run(command, check=True, env=env)
        elapsed = time.time() - start_time
        print(f"\n  SUCCESS: {module_path} completed in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: {module_path} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)


# =============================================================================
# Training pipeline (unchanged from original)
# =============================================================================

def run_training_pipeline() -> None:
    """
    Manages the execution flow of the full training pipeline, checking
    for existing artifacts to avoid redundant and expensive computations.
    """
    print("\n[MODE] INITIALIZING FULL TRAINING PIPELINE")

    if not os.path.exists(CFG.MODEL_MEAN_PATH):
        execute_module("src/train_forecast.py")
    else:
        print(f"  SKIP: Forecast model already exists at {CFG.MODEL_MEAN_PATH}")

    if not os.path.exists(CFG.MODEL_ENN_PATH):
        execute_module("src/training_enn.py")
    else:
        print(f"  SKIP: ENN model already exists at {CFG.MODEL_ENN_PATH}")

    if not os.path.exists(CFG.CSV_OUTPUT_PATH):
        execute_module("src/collect_data.py")
    else:
        print(f"  SKIP: Data already collected at {CFG.CSV_OUTPUT_PATH}")

    execute_module("src/train_classifier.py")


# =============================================================================
# LLM Rule Inference mode  ← NEW
# =============================================================================

def run_llm_rule_inference() -> None:
    """
    Loads the trained models and the LLM symbolic rules, then runs one
    simulation episode. For each monitored line, at every analysis step it:

      1. Runs the forecast pipeline (t+12) — same logic as collect_data.py.
      2. Applies the symbolic rule for that line.
      3. Prints the binary prediction (0 = OK, 1 = predicted failure) and
         the natural-language explanation sentence (LaTeX-ready for the paper).

    Configuration in src/config.py:
      LLM_RULES_DIR      : path to the folder with line_*/best_rule.py files
                           e.g. "llm_rules_results/temp_0.5"
      LLM_RULES_EPISODE  : episode seed to simulate (default: CFG.PROBA_TEST_EPISODE_ID)

    The sentences for all available lines are also printed at startup so you
    can use them directly in the paper table.
    """
    import joblib
    import numpy as np
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Reward import L2RPNReward
    from curriculumagent.baseline.baseline import CurriculumAgent

    # Local imports (src/ is on PYTHONPATH when run via run_pipeline.py)
    from src.training_enn import load_trained_enn, _scaler_path, get_uncertainty
    from src.utils import compute_grid_stats
    from src.collect_data import get_features_with_history
    from src.rule_predictor import RulePredictor

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    rules_dir      = getattr(CFG, "LLM_RULES_DIR",     "llm_rules_results/temp_0.5")
    episode_seed   = getattr(CFG, "LLM_RULES_EPISODE",  getattr(CFG, "PROBA_TEST_EPISODE_ID", 50))
    analysis_every = getattr(CFG, "ANALYSIS_STEP",       20)   # analyse every N steps

    print(f"\n[MODE] LLM RULE INFERENCE")
    print(f"  Rules directory : {rules_dir}")
    print(f"  Episode seed    : {episode_seed}")
    print(f"  Analysis every  : {analysis_every} steps")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\n  Loading models...")
    try:
        model_predict   = joblib.load(CFG.MODEL_MEAN_PATH)
        model_aleatoric = joblib.load(CFG.MODEL_ALEATORIC_PATH)
        model_enn       = load_trained_enn()
        print("  All models loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Could not load models: {e}")
        print("  Run the training pipeline first (TRAIN_MODE=True).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Instantiate RulePredictor  (once — reused across all steps)
    # ------------------------------------------------------------------
    observations_array = []   # accumulated observations for the episode

    rule_predictor = RulePredictor(
        rules_dir=rules_dir,
        model_predict=model_predict,
        model_aleatoric=model_aleatoric,
        model_enn=model_enn,
        # The functions and CFG are imported automatically inside rule_predictor.py
        # from the project modules — no need to pass them explicitly.
        observations_array=observations_array,
    )

    # Print all rule sentences at startup (useful for the paper table)
    sentences = rule_predictor.all_sentences()
    if sentences:
        print(f"\n{'=' * 70}")
        print("  SYMBOLIC RULE SENTENCES (all available lines)")
        print(f"{'=' * 70}")
        for line_name, sentence in sentences.items():
            print(f"\n  [{line_name}]\n  {sentence}")
        print(f"\n{'=' * 70}\n")
    else:
        print(f"  [WARN] No rules found in '{rules_dir}'. Check the path.")

    # ------------------------------------------------------------------
    # Simulation episode
    # ------------------------------------------------------------------
    env   = grid2op.make(CFG.ENV_NAME, reward_class=L2RPNReward, backend=LightSimBackend())
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("  [WARN] Agent could not be loaded — using do-nothing fallback.")

    obs  = env.reset(seed=episode_seed)
    done = False
    reward = env.reward_range[0]

    print(f"  Starting episode (seed={episode_seed})...\n")

    while not done:
        observations_array.append(obs)
        # observations_array is the same list object referenced by rule_predictor,
        # so the predictor always sees the latest history automatically.

        if obs.current_step > 12 and obs.current_step % analysis_every == 0:
            print(f"  --- Step {obs.current_step} ---")

            for line_name in CFG.LINES_TO_TEST:
                # Normalise: LINES_TO_TEST may contain int IDs or string names
                if isinstance(line_name, int):
                    line_str = _line_id_to_name(line_name, env)
                else:
                    line_str = str(line_name)

                result = rule_predictor.predict(obs=obs, line_name=line_str)

                status = "FAILURE PREDICTED" if result["prediction"] == 1 else "OK"
                print(f"    Line {result['line_name']:15s} -> {status}")
                if result["prediction"] == 1:
                    print(f"      {result['sentence']}")

        try:
            action = agent.act(obs, reward, done)
        except Exception:
            action = env.action_space({})

        obs, reward, done, _ = env.step(action)

    env.close()
    print("\n  Episode finished.")


def _line_id_to_name(line_id: int, env) -> str:
    """Converts an integer line ID to its string name using env.name_line."""
    try:
        return str(env.name_line[line_id])
    except Exception:
        return str(line_id)


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """
    Routes execution based on the configuration flags in src/config.py:

      TRAIN_MODE=True      -> Full training pipeline (forecasters + ENN + data + classifier)
      TEST_SINGLE_EPISODE  -> Single episode simulation and data collection
      PREDICT_PROBA_MODE   -> Probabilistic inference on a single observation
      LLM_RULE_MODE=True   -> LLM symbolic rule inference: runs a simulation episode
                              and reports per-line failure predictions with
                              natural-language explanations (LaTeX-ready for the paper)

    To activate LLM rule inference, set in src/config.py:
        TRAIN_MODE          = False
        TEST_SINGLE_EPISODE = False
        PREDICT_PROBA_MODE  = False
        LLM_RULE_MODE       = True
    """
    environment_name = getattr(CFG, "ENV_NAME", "UNKNOWN")
    print(f"  CONFIGURATION: ENV={environment_name}")

    if TRAIN_MODE:
        run_training_pipeline()
    elif TEST_SINGLE_EPISODE:
        execute_module("src/collect_data.py")
    elif PREDICT_PROBA_MODE:
        execute_module("src/train_classifier.py")
    elif LLM_RULE_MODE:
        run_llm_rule_inference()
    else:
        print("\n  WARNING: No active execution mode selected in src/config.py.")
        print("  Set one of: TRAIN_MODE, TEST_SINGLE_EPISODE, PREDICT_PROBA_MODE, LLM_RULE_MODE = True")


if __name__ == "__main__":
    main()