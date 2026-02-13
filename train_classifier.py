import os
import joblib
import optuna
import pandas as pd
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, f1_score, accuracy_score
from typing import Tuple, List, Any, Dict

# Local imports
from config import CFG, DEVICE, ENN_PARAMS, ENN_DROPOUT, TRAIN_MODE
from enn_models import EvidentialNetwork, calculate_epistemic_uncertainty
from utils import get_features, compute_grid_stats


# We don't need the network36 for pure inference, just environment structure

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_and_prep_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    df["line_id_encoded"] = df["line_disconnected"].astype("category").cat.codes
    # Add epsilon to avoid division by zero
    df["load_gen_ratio"] = df["sum_load_p"] / (df["sum_gen_p"] + 1e-6)
    df["label"] = df["failed"].astype(int)

    for col in ["epistemic_before"]:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    denom_fa, denom_ovr = (tp + fp), (tp + fn)
    fa_rate = (fp / denom_fa * 100) if denom_fa > 0 else 0.0
    ovr_rate = (fn / denom_ovr * 100) if denom_ovr > 0 else 0.0
    return fa_rate, ovr_rate, cm


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cat_indices: List[int]) -> float:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 255),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced'])
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = HistGradientBoostingClassifier(categorical_features=cat_indices, random_state=42, **params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores.append(fbeta_score(y.iloc[val_idx], preds, beta=2))
    return float(np.mean(scores))


# ==============================================================================
# NEW FUNCTION: PREDICT PROBABILITY FOR SINGLE OBSERVATION (NO SIMULATION)
# ==============================================================================
def predict_failure_probability_single_obs(
        env: grid2op.Environment,
        obs: Any,
        models: Dict[str, Any]
) -> pd.DataFrame:
    """
    Given a single observation, calculates the failure probability
    for each disconnected line defined in config WITHOUT running simulations.
    It generates the features based on the current state and model predictions.
    """
    # Unpack models
    model_aleatoric = models['aleatoric']
    model_enn = models['enn']
    classifier = models['classifier']

    # 1. Aleatoric Uncertainty (Forecasting)
    # Ideally we use history, but for a single detached observation, we use the current obs as dummy history
    # This assumes the state is static for feature generation purposes if history isn't provided
    dummy_history = [obs]

    # We predict 12 steps ahead using the forecaster logic to get the sigma
    x_forecast, _ = get_features(dummy_history, obs, 12)

    y_var = model_aleatoric.predict(x_forecast)[0]
    y_sigma = np.sqrt(np.abs(y_var))

    aleatoric_load_p = np.mean(y_sigma[:CFG.NO_LOADS])
    aleatoric_load_q = np.mean(y_sigma[CFG.NO_LOADS: CFG.NO_LOADS * 2])
    aleatoric_gen_p = np.mean(y_sigma[CFG.NO_LOADS * 2:])

    # 2. Epistemic Uncertainty (ENN)
    try:
        obs_tensor = torch.tensor(obs.to_vect(), dtype=torch.float32).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            epistemic_before = calculate_epistemic_uncertainty(model_enn(obs_tensor))
    except:
        epistemic_before = 0.0

    # 3. Grid Stats (Current State)
    grid_stats = compute_grid_stats(obs)
    load_gen_ratio = grid_stats["sum_load_p"] / (grid_stats["sum_gen_p"] + 1e-6)

    results = []

    # 4. Infer probability for each line
    for line_name in CFG.LINES_TO_TEST:
        # Resolve Line ID
        line_id = -1
        if hasattr(env, "name_line"):
            try:
                line_id = list(env.name_line).index(line_name)
            except ValueError:
                # If name doesn't exist (e.g. wrong config), skip
                continue

        # Build Feature Vector matching training order
        feature_row = {
            "line_id_encoded": line_id,
            "sum_load_p": grid_stats["sum_load_p"],
            "sum_load_q": grid_stats["sum_load_q"],
            "sum_gen_p": grid_stats["sum_gen_p"],
            "sum_gen_q": grid_stats["sum_gen_q"],
            "var_line_rho": grid_stats["var_line_rho"],
            "avg_line_rho": grid_stats["avg_line_rho"],
            "max_line_rho": grid_stats["max_line_rho"],
            "nb_rho_ge_0.95": grid_stats["nb_rho_ge_0.95"],
            "aleatoric_load_p_mean": aleatoric_load_p,
            "aleatoric_load_q_mean": aleatoric_load_q,
            "aleatoric_gen_p_mean": aleatoric_gen_p,
            "load_gen_ratio": load_gen_ratio,
            "epistemic_before": epistemic_before,
        }

        # Ensure column order matches training exactly
        ordered_cols = [
            "line_id_encoded",
            "sum_load_p", "sum_load_q", "sum_gen_p", "sum_gen_q",
            "var_line_rho", "avg_line_rho", "max_line_rho", "nb_rho_ge_0.95",
            "aleatoric_load_p_mean", "aleatoric_load_q_mean", "aleatoric_gen_p_mean",
            "load_gen_ratio", "epistemic_before"
        ]

        X_input = pd.DataFrame([feature_row])[ordered_cols]

        # Predict Probability (Class 1 = Failure)
        proba_failure = classifier.predict_proba(X_input)[0][1]

        results.append({
            "line_name": line_name,
            "line_id": line_id,
            "failure_probability": proba_failure
        })

    return pd.DataFrame(results).sort_values(by="failure_probability", ascending=False)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------
    # MODE 1: TRAIN CLASSIFIER
    # -----------------------------------------------------------
    if TRAIN_MODE:
        baseline_features = [
            "sum_load_p", "sum_load_q", "sum_gen_p", "sum_gen_q",
            "var_line_rho", "avg_line_rho", "max_line_rho", "nb_rho_ge_0.95",
            "aleatoric_load_p_mean", "aleatoric_load_q_mean", "aleatoric_gen_p_mean", "load_gen_ratio"
        ]
        features_to_use = ["line_id_encoded"] + baseline_features + ["epistemic_before"]
        cat_indices = [0]

        try:
            df = load_and_prep_data(CFG.CSV_OUTPUT_PATH)
        except FileNotFoundError:
            print(f"Data file not found at {CFG.CSV_OUTPUT_PATH}. Run collect_data.py first.")
            exit(1)

        X = df[features_to_use]
        y = df["label"]

        print("[INFO] TRAIN_MODE=True. Training Classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_train, y_train, cat_indices), n_trials=20)

        final_model = HistGradientBoostingClassifier(categorical_features=cat_indices, random_state=42,
                                                     **study.best_params)
        final_model.fit(X_train, y_train)

        joblib.dump(final_model, CFG.MODEL_CLASSIFIER_PATH)
        print(f"[SAVE] Classifier saved to {CFG.MODEL_CLASSIFIER_PATH}")

        y_pred = final_model.predict(X_test)
        fa, ovr, cm = calculate_metrics(y_test, y_pred)
        print(f"Oversight Rate: {ovr:.2f}% | False Alarm: {fa:.2f}%")

    # -----------------------------------------------------------
    # MODE 2: PREDICT PROBABILITY (SINGLE OBSERVATION)
    # -----------------------------------------------------------
    elif CFG.PREDICT_PROBA_MODE:
        print("[INFO] PREDICT_PROBA_MODE=True. Loading Models...")

        if not os.path.exists(CFG.MODEL_CLASSIFIER_PATH):
            print("Classifier not found. Please train first.")
            exit(1)

        # Load all models
        try:
            model_aleatoric = joblib.load(CFG.MODEL_ALEATORIC_PATH)
            model_enn = EvidentialNetwork(ENN_PARAMS['input_dim'], ENN_PARAMS['num_classes'], ENN_PARAMS['hidden_dim'],
                                          ENN_DROPOUT).to(DEVICE)
            # Try loading weights, silent fail if random init (for testing code flow)
            try:
                model_enn.load_state_dict(torch.load(CFG.MODEL_ENN_PATH, map_location=DEVICE))
            except:
                pass
            model_enn.eval()
            classifier = joblib.load(CFG.MODEL_CLASSIFIER_PATH)
        except Exception as e:
            print(f"Error loading models: {e}")
            exit(1)

        models_dict = {
            'aleatoric': model_aleatoric,
            'enn': model_enn,
            'classifier': classifier
        }

        # Initialize Environment just to get structure/names (or use a dummy)
        env = grid2op.make(CFG.ENV_NAME, backend=LightSimBackend())

        # --- GET A SAMPLE OBSERVATION ---
        # In a real scenario, you would pass your own 'obs' object here.
        # For this script, we fetch one from the environment.
        print(f"[INFO] Fetching sample observation (Ep {CFG.PROBA_TEST_EPISODE_ID}, Step {CFG.PROBA_TEST_STEP})...")
        env.seed(CFG.PROBA_TEST_EPISODE_ID)
        obs = env.reset()
        done = False
        while not done:
            if obs.current_step == CFG.PROBA_TEST_STEP:
                break
            obs, _, done, _ = env.step(env.action_space())
            if done:
                obs = env.reset()  # Reset if episode ends early
                break

        # --- RUN INFERENCE ---
        # This is the function you want: inputs (Env, Obs, Models) -> Output (Probabilities)
        probs_df = predict_failure_probability_single_obs(env, obs, models_dict)

        print("\n[RESULTS] Failure Probabilities by Line:")
        print(probs_df)

        # Optional: Save to CSV
        save_path = os.path.join(os.path.dirname(CFG.CSV_OUTPUT_PATH), "inference_probs.csv")
        probs_df.to_csv(save_path, index=False)
        print(f"[SAVE] Results saved to {save_path}")

    else:
        print("Please enable TRAIN_MODE or PREDICT_PROBA_MODE in config.py")