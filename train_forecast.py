from __future__ import annotations

import os
import joblib
import optuna
import numpy as np
import grid2op
from typing import Any, Dict, List, Optional, Sequence, Tuple

from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Local imports
from config import CFG, TRAIN_MODE
from utils import get_features, append_to_npy
from curriculumagent.baseline.baseline import CurriculumAgent


# =============================================================================
# Utility metrics
# =============================================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) in percentage.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth, shape (n_samples, n_dims) or (n_dims,).
    y_pred : np.ndarray
        Predictions, same shape as y_true.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        sMAPE values (%), same shape as inputs (broadcasted).
        If inputs are (n_samples, n_dims), output is (n_samples, n_dims).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return 200.0 * np.abs(y_pred - y_true) / denom


# =============================================================================
# Data collection
# =============================================================================

def collect_training_data(env: grid2op.Environment, agent: CurriculumAgent, episode_seeds: Sequence[int]) -> None:
    """
    Run episodes to collect supervised pairs:
      X(t)  = features from get_features(history up to t, obs_t, horizon=0)
      y(t)  = concat([obs_t.load_p, obs_t.load_q, obs_t.gen_p])

    This is "instantaneous" mapping at the same timestamp (horizon=0),
    and later we evaluate multi-horizon forecasts by calling get_features(..., horizon=h)
    at evaluation time.

    Parameters
    ----------
    env : grid2op.Environment
        Grid2Op environment.
    agent : CurriculumAgent
        Acting agent for stepping the environment.
    episode_seeds : Sequence[int]
        Episode seeds to run and collect data from.
    """
    print(f"[INFO] Collecting data for {CFG.ENV_NAME}...")

    # Remove old data files if present
    if os.path.exists(CFG.X_TRAIN_PATH):
        os.remove(CFG.X_TRAIN_PATH)
    if os.path.exists(CFG.Y_TRAIN_PATH):
        os.remove(CFG.Y_TRAIN_PATH)

    for idx, ep_seed in enumerate(episode_seeds):
        obs = env.reset(seed=int(ep_seed))
        done = False

        observations: List[Any] = []
        X_ep: List[np.ndarray] = []
        y_ep: List[np.ndarray] = []

        while not done:
            observations.append(obs)

            # horizon=0 means "features for the current obs"
            x, _ = get_features(observations, obs, 0)
            y = np.concatenate([obs.load_p, obs.load_q, obs.gen_p], axis=0)

            X_ep.append(np.asarray(x).flatten())
            y_ep.append(np.asarray(y))

            action = agent.act(obs, env.reward_range[0], done)
            obs, _, done, _ = env.step(action)

        append_to_npy(CFG.X_TRAIN_PATH, np.asarray(X_ep))
        append_to_npy(CFG.Y_TRAIN_PATH, np.asarray(y_ep))

        if (idx + 1) % 50 == 0:
            print(f"[DATA] Saved episode {idx + 1}/{len(episode_seeds)}")


# =============================================================================
# Model training
# =============================================================================

def train_mean_model(n_trials: int = 20, max_subsample: int = 5000, random_state: int = 42) -> None:
    """
    Train the mean forecasting model using Optuna hyperparameter search.

    Model:
      MultiOutputRegressor(HistGradientBoostingRegressor)

    Objective:
      Minimize RMSE on a random subsample of training set.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials (requested: 20).
    max_subsample : int
        Maximum number of samples used inside each trial evaluation.
    random_state : int
        Seed for reproducible subsampling inside Optuna objective.
    """
    print("[TRAIN] Training Mean Model...")
    X_train = np.load(CFG.X_TRAIN_PATH, allow_pickle=True)
    y_train = np.load(CFG.Y_TRAIN_PATH, allow_pickle=True)

    rng = np.random.default_rng(random_state)

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
            "early_stopping": True,
        }

        n = len(X_train)
        sub_n = min(max_subsample, n)
        sub_idx = rng.choice(n, size=sub_n, replace=False)

        model = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
        model.fit(X_train[sub_idx], y_train[sub_idx])

        preds = model.predict(X_train[sub_idx])
        rmse = float(np.sqrt(mean_squared_error(y_train[sub_idx], preds)))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    # Ensure early_stopping present even if optuna didn't sample it
    best_params["early_stopping"] = True

    best_model = MultiOutputRegressor(HistGradientBoostingRegressor(**best_params))
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, CFG.MODEL_MEAN_PATH)
    print(f"[SAVE] Mean Model saved to {CFG.MODEL_MEAN_PATH}")
    print(f"[TRAIN] Best params: {best_params}")


def train_aleatoric_model() -> None:
    """
    Train an aleatoric (variance) model on the residuals of the mean model, robustly.

    Problem:
    - Using loss='poisson' on squared residuals can overflow due to exp(link) internally.
    Solution:
    - Fit in log-space: z = log1p(residual^2) using squared_error loss
    - At inference: var = expm1(z_pred), sigma = sqrt(var)

    Output model predicts z_pred (log-variance), not raw variance.
    """
    print("[TRAIN] Training Aleatoric Model (log-variance target)...")
    X_train = np.load(CFG.X_TRAIN_PATH, allow_pickle=True)
    y_train = np.load(CFG.Y_TRAIN_PATH, allow_pickle=True)

    mean_model = joblib.load(CFG.MODEL_MEAN_PATH)
    y_pred = mean_model.predict(X_train)

    squared_residuals = (y_train - y_pred) ** 2

    # Clip extreme residuals before log transform for stability
    upper = np.percentile(squared_residuals, 99.9, axis=0)
    squared_residuals = np.clip(squared_residuals, 0.0, upper)

    # Stable log-variance target
    z = np.log1p(squared_residuals)

    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        l2_regularization=2.0,
        early_stopping=True,
    )
    var_model = MultiOutputRegressor(base)
    var_model.fit(X_train, z)

    joblib.dump(var_model, CFG.MODEL_ALEATORIC_PATH)
    print(f"[SAVE] Aleatoric Model saved to {CFG.MODEL_ALEATORIC_PATH}")


# =============================================================================
# Evaluation
# =============================================================================

def collect_episode_observations(env: grid2op.Environment, agent: CurriculumAgent, seed: int) -> List[Any]:
    """
    Run one episode and return the full list of observations (one per environment step).

    Parameters
    ----------
    env : grid2op.Environment
        Environment instance.
    agent : CurriculumAgent
        Acting agent.
    seed : int
        Episode seed.

    Returns
    -------
    List[Any]
        List of Grid2Op observations for each step in the episode.
    """
    obs = env.reset(seed=int(seed))
    done = False
    obs_list: List[Any] = []

    while not done:
        obs_list.append(obs)
        action = agent.act(obs, env.reward_range[0], done)
        obs, _, done, _ = env.step(action)

    return obs_list


def evaluate_forecast_on_episodes(
    env: grid2op.Environment,
    agent: CurriculumAgent,
    model_mean: Any,
    episode_seeds: Sequence[int],
    max_horizon_steps: int = 12,
) -> Dict[str, Any]:
    """
    Evaluate direct multi-horizon forecasts (5..60 minutes) using get_features(..., horizon=h).

    For each episode and each time t, for each horizon h:
      - predict y_hat(t+h) from features at time t with horizon=h
      - compare with true values at time t+h

    Returns:
      - Global RMSE, R2 across ALL horizons pooled together
      - Global sMAPE per group: load_p, load_q, gen_p pooled together
      - sMAPE by horizon and by dimension (each load/gen index separately)

    Parameters
    ----------
    env : grid2op.Environment
        Environment instance.
    agent : CurriculumAgent
        Acting agent.
    model_mean : Any
        Trained mean model with predict().
    episode_seeds : Sequence[int]
        Seeds defining which episodes to evaluate on.
    max_horizon_steps : int
        Number of forecast steps (12 => 5..60 min).

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - r2, rmse, smape_load_p, smape_load_q, smape_gen_p
        - smape_load_p_by_h_dim: (12, NO_LOADS)
        - smape_load_q_by_h_dim: (12, NO_LOADS)
        - smape_gen_p_by_h_dim:  (12, NO_GENS)
    """
    # Pooled arrays across all horizons
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    # Per horizon accumulators
    true_lp_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]
    pred_lp_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]
    true_lq_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]
    pred_lq_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]
    true_gp_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]
    pred_gp_by_h: List[List[np.ndarray]] = [[] for _ in range(max_horizon_steps)]

    for ep in episode_seeds:
        obs_list = collect_episode_observations(env, agent, seed=int(ep))

        for t in range(len(obs_list)):
            obs_t = obs_list[t]
            history = obs_list[: t + 1]  # history up to current time t

            for h in range(1, max_horizon_steps + 1):
                if t + h >= len(obs_list):
                    break

                # Features at time t for horizon h
                x, _ = get_features(history, obs_t, h)
                y_hat = np.asarray(model_mean.predict(x)[0])

                # Ground-truth at time t+h
                obs_th = obs_list[t + h]
                y = np.concatenate([obs_th.load_p, obs_th.load_q, obs_th.gen_p], axis=0)

                y_true_all.append(y)
                y_pred_all.append(y_hat)

                # Split into groups
                lp_true = y[:CFG.NO_LOADS]
                lq_true = y[CFG.NO_LOADS : CFG.NO_LOADS * 2]
                gp_true = y[CFG.NO_LOADS * 2 :]

                lp_pred = y_hat[:CFG.NO_LOADS]
                lq_pred = y_hat[CFG.NO_LOADS : CFG.NO_LOADS * 2]
                gp_pred = y_hat[CFG.NO_LOADS * 2 :]

                idx = h - 1
                true_lp_by_h[idx].append(lp_true)
                pred_lp_by_h[idx].append(lp_pred)
                true_lq_by_h[idx].append(lq_true)
                pred_lq_by_h[idx].append(lq_pred)
                true_gp_by_h[idx].append(gp_true)
                pred_gp_by_h[idx].append(gp_pred)

    y_true_all_arr = np.asarray(y_true_all)
    y_pred_all_arr = np.asarray(y_pred_all)

    rmse = float(np.sqrt(mean_squared_error(y_true_all_arr, y_pred_all_arr)))
    r2 = float(r2_score(y_true_all_arr, y_pred_all_arr, multioutput="uniform_average"))

    smape_lp_global = float(np.mean(smape(
        y_true_all_arr[:, :CFG.NO_LOADS],
        y_pred_all_arr[:, :CFG.NO_LOADS]
    )))
    smape_lq_global = float(np.mean(smape(
        y_true_all_arr[:, CFG.NO_LOADS : CFG.NO_LOADS * 2],
        y_pred_all_arr[:, CFG.NO_LOADS : CFG.NO_LOADS * 2]
    )))
    smape_gp_global = float(np.mean(smape(
        y_true_all_arr[:, CFG.NO_LOADS * 2 :],
        y_pred_all_arr[:, CFG.NO_LOADS * 2 :]
    )))

    # Per-horizon per-dimension matrices
    smape_lp_h_dim = np.full((max_horizon_steps, CFG.NO_LOADS), np.nan, dtype=np.float64)
    smape_lq_h_dim = np.full((max_horizon_steps, CFG.NO_LOADS), np.nan, dtype=np.float64)
    smape_gp_h_dim = np.full((max_horizon_steps, CFG.NO_GENS), np.nan, dtype=np.float64)

    for h in range(max_horizon_steps):
        if len(true_lp_by_h[h]) == 0:
            continue

        tlp = np.asarray(true_lp_by_h[h])
        plp = np.asarray(pred_lp_by_h[h])
        tlq = np.asarray(true_lq_by_h[h])
        plq = np.asarray(pred_lq_by_h[h])
        tgp = np.asarray(true_gp_by_h[h])
        pgp = np.asarray(pred_gp_by_h[h])

        # smape returns (n_samples, dim) -> average over samples yields per-dim vector
        smape_lp_h_dim[h, :] = np.mean(smape(tlp, plp), axis=0)
        smape_lq_h_dim[h, :] = np.mean(smape(tlq, plq), axis=0)
        smape_gp_h_dim[h, :] = np.mean(smape(tgp, pgp), axis=0)

    return {
        "r2": r2,
        "rmse": rmse,
        "smape_load_p": smape_lp_global,
        "smape_load_q": smape_lq_global,
        "smape_gen_p": smape_gp_global,
        "smape_load_p_by_h_dim": smape_lp_h_dim,
        "smape_load_q_by_h_dim": smape_lq_h_dim,
        "smape_gen_p_by_h_dim": smape_gp_h_dim,
    }


def save_metrics_txt(path: str, title: str, metrics: Dict[str, Any]) -> None:
    """
    Append metrics to a plain text file.

    It writes:
    - Global metrics: R2, RMSE, sMAPE group-level
    - Per-horizon (5..60min) sMAPE matrices, one row per horizon,
      columns are per-load or per-generator indices.

    Parameters
    ----------
    path : str
        Output text file path.
    title : str
        Title block.
    metrics : Dict[str, Any]
        Output dictionary from evaluate_forecast_on_episodes.
    """
    horizons_min = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 100 + "\n")
        f.write(f"R2:   {metrics['r2']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"sMAPE load_p (global): {metrics['smape_load_p']:.6f}\n")
        f.write(f"sMAPE load_q (global): {metrics['smape_load_q']:.6f}\n")
        f.write(f"sMAPE gen_p  (global): {metrics['smape_gen_p']:.6f}\n")

        f.write("\n--- sMAPE load_p per horizon (rows) and per load index (cols) ---\n")
        for i, m in enumerate(horizons_min):
            row = metrics["smape_load_p_by_h_dim"][i]
            f.write(f"{m:02d}min: " + ",".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in row) + "\n")

        f.write("\n--- sMAPE load_q per horizon (rows) and per load index (cols) ---\n")
        for i, m in enumerate(horizons_min):
            row = metrics["smape_load_q_by_h_dim"][i]
            f.write(f"{m:02d}min: " + ",".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in row) + "\n")

        f.write("\n--- sMAPE gen_p per horizon (rows) and per generator index (cols) ---\n")
        for i, m in enumerate(horizons_min):
            row = metrics["smape_gen_p_by_h_dim"][i]
            f.write(f"{m:02d}min: " + ",".join("nan" if not np.isfinite(v) else f"{v:.6f}" for v in row) + "\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not TRAIN_MODE:
        print("[INFO] TRAIN_MODE is False. Skipping training.")
        raise SystemExit(0)

    # -----------------------------
    # Environment + agent setup
    # -----------------------------
    env: grid2op.Environment = grid2op.make(
        CFG.ENV_NAME,
        backend=LightSimBackend(),
        reward_class=L2RPNReward,
    )

    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
        print("[INFO] CurriculumAgent loaded.")
    except Exception:
        print("[WARN] Agent could not be loaded. The run will still proceed, but behavior may differ.")

    # -----------------------------
    # Episode splits
    # -----------------------------
    # Training data episodes (used to collect X/Y AND for training metrics)
    train_seeds: List[int] = list(range(0, 100))

    # Disjoint evaluation episodes (test metrics)
    test_seeds: List[int] = list(range(100, 120))

    # Where to save text metrics
    # If CFG doesn't define FORECAST_METRICS_TXT, fallback to local file.
    metrics_txt_path: str = getattr(CFG, "FORECAST_METRICS_TXT", "forecast_metrics.txt")

    # Optional: clear old metrics file each run
    # Comment this out if you want to append over multiple runs.
    if os.path.exists(metrics_txt_path):
        os.remove(metrics_txt_path)

    # -----------------------------
    # 1) Collect training data
    # -----------------------------
    collect_training_data(env, agent, train_seeds)

    # -----------------------------
    # 2) Train models
    # -----------------------------
    train_mean_model(n_trials=20)
    train_aleatoric_model()

    # -----------------------------
    # 3) Evaluate mean model and write metrics
    # -----------------------------
    model_mean = joblib.load(CFG.MODEL_MEAN_PATH)

    train_metrics = evaluate_forecast_on_episodes(env, agent, model_mean, train_seeds, max_horizon_steps=12)
    save_metrics_txt(metrics_txt_path, "METRICS ON TRAIN EPISODES (0-99)", train_metrics)

    test_metrics = evaluate_forecast_on_episodes(env, agent, model_mean, test_seeds, max_horizon_steps=12)
    save_metrics_txt(metrics_txt_path, "METRICS ON TEST EPISODES (100-119)", test_metrics)

    print(f"[METRICS] Metrics saved to: {metrics_txt_path}")