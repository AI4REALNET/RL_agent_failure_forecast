import joblib
import pandas as pd
import numpy as np
import torch
import os
import grid2op
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import BaseActionBudget, RandomLineOpponent
from grid2op.Exceptions import Grid2OpException
from lightsim2grid import LightSimBackend
from typing import List, Any, Dict

# Local Imports
from config import CFG, DEVICE, ENN_PARAMS, ENN_DROPOUT, TRAIN_MODE, TEST_SINGLE_EPISODE
from enn_models import EvidentialNetwork, calculate_epistemic_uncertainty
from utils import get_features, compute_grid_stats
from curriculumagent.baseline import CurriculumAgent

import datetime



def save_incremental(df: pd.DataFrame, path: str) -> None:
    """Helper function to save data to a CSV file incrementally."""
    if df is None or df.empty:
        print("[DEBUG] Tentativa de salvar DataFrame vazio.")  # Adicione isto
        return
    if not os.path.exists(path):
        df.to_csv(path, index=False, mode='w')
    else:
        df.to_csv(path, index=False, mode='a', header=False)


def analyze_disconnection_effect(
        env,
        model_predict,
        model_aleatoric,
        model_enn,
        obs,
        observations_array,
        ep,
        agent
) -> pd.DataFrame:
    """
    Versão alinhada com model__predict.py:
    - Usa obs._forecasted_inj para suportar simulate() com forecasts
    - Agente atua a cada passo, e no último passo testa desconexões
    - Mantém as stats via compute_grid_stats(sim_obs)
    """

    results = []

    # -----------------------------
    # 1) Epistemic BEFORE (t=0)
    # -----------------------------
    try:
        obs_tensor = torch.tensor(obs.to_vect(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            alpha_before = model_enn(obs_tensor)
        unc_epistemic_before = float(calculate_epistemic_uncertainty(alpha_before))
    except Exception:
        unc_epistemic_before = -1.0

    # -----------------------------
    # 2) Forecast traj + timestamps
    # -----------------------------
    temp_obs_array = observations_array.copy()
    current_sim_obs = obs

    load_p_traj, load_q_traj, gen_p_traj = [], [], []
    timestamps = []

    final_sigma_load_p = np.zeros(CFG.NO_LOADS)
    final_sigma_load_q = np.zeros(CFG.NO_LOADS)
    final_sigma_gen_p  = np.zeros(CFG.NO_GENS)

    base_ts = obs.get_time_stamp()

    for step in range(1, 13):
        x, _ = get_features(temp_obs_array, current_sim_obs, step)
        # atenção: no teu treino usas model.predict(x) com x shape (1,dim)
        y_pred = model_predict.predict(x)[0]

        l_p = y_pred[:CFG.NO_LOADS]
        l_q = y_pred[CFG.NO_LOADS: CFG.NO_LOADS * 2]
        g_p = np.clip(y_pred[CFG.NO_LOADS * 2:], CFG.GEN_MIN, CFG.GEN_MAX)

        load_p_traj.append(l_p)
        load_q_traj.append(l_q)
        gen_p_traj.append(g_p)

        ts = base_ts + datetime.timedelta(minutes=step * 5)
        timestamps.append(ts)

        # aleatoric só no último passo
        if step == 12:
            y_var_pred = model_aleatoric.predict(x)[0]

            # se aplicares a correção do aleatoric por log (secção 2),
            # troca para: y_var = np.expm1(y_var_pred)
            y_var = np.abs(y_var_pred)

            y_sigma = np.sqrt(y_var + 1e-12)
            final_sigma_load_p = y_sigma[:CFG.NO_LOADS]
            final_sigma_load_q = y_sigma[CFG.NO_LOADS: CFG.NO_LOADS * 2]
            final_sigma_gen_p  = y_sigma[CFG.NO_LOADS * 2:]

    # -----------------------------
    # 3) Simulação com _forecasted_inj
    # -----------------------------
    sim_obs = obs.copy()
    done_sim = False

    for i in range(12):
        # define forecast injection (mesmo padrão do model__predict.py)
        now_ts = sim_obs.get_time_stamp()

        sim_obs._forecasted_inj = [
            (
                now_ts,
                {"injection": {
                    "load_p": sim_obs.load_p,
                    "load_q": sim_obs.load_q,
                    "prod_p": sim_obs.gen_p,
                    "prod_v": sim_obs.gen_v
                }}
            ),
            (
                timestamps[i],
                {"injection": {
                    "load_p": load_p_traj[i],
                    "load_q": load_q_traj[i],
                    "prod_p": gen_p_traj[i],
                    "prod_v": sim_obs.gen_v
                }}
            )
        ]

        # último passo: testar desconexões
        if i == 11:
            for line_target in CFG.LINES_TO_TEST:
                try:
                    line_id = line_target
                    if isinstance(line_target, str) and hasattr(env, "name_line"):
                        if line_target in env.name_line:
                            line_id = list(env.name_line).index(line_target)

                    action_disc = env.action_space({"set_line_status": [(line_id, -1)]})
                    obs_disc, _, failed, _ = sim_obs.simulate(action_disc)

                    if not failed:
                        try:
                            obs_tensor_after = torch.tensor(
                                obs_disc.to_vect(), dtype=torch.float32, device=DEVICE
                            ).unsqueeze(0)
                            with torch.no_grad():
                                alpha_after = model_enn(obs_tensor_after)
                            unc_epistemic_after = float(calculate_epistemic_uncertainty(alpha_after))
                        except Exception:
                            unc_epistemic_after = unc_epistemic_before
                    else:
                        unc_epistemic_after = -1.0

                    grid_stats = compute_grid_stats(sim_obs)

                    results.append({
                        "episode": ep,
                        "step": obs.current_step,
                        "line_disconnected": line_target,
                        "failed": 1 if failed else 0,
                        "aleatoric_load_p_mean": float(np.mean(final_sigma_load_p)),
                        "aleatoric_load_q_mean": float(np.mean(final_sigma_load_q)),
                        "aleatoric_gen_p_mean": float(np.mean(final_sigma_gen_p)),
                        "epistemic_before": float(unc_epistemic_before),
                        "epistemic_after": float(unc_epistemic_after),
                        "sum_load_p": grid_stats["sum_load_p"],
                        "sum_gen_p": grid_stats["sum_gen_p"],
                        "avg_line_rho": grid_stats["avg_line_rho"],
                        "max_line_rho": grid_stats["max_line_rho"],
                        "nb_rho_ge_0.95": grid_stats["nb_rho_ge_0.95"],
                    })

                except Exception:
                    continue

            break  # terminou

        # não é o último passo: agente atua
        try:
            action = agent.act(sim_obs, 0.0, False)
        except Exception:
            action = env.action_space()  # do nothing

        try:
            sim_obs, _, done_sim, _ = sim_obs.simulate(action)
        except Exception:
            break

        if done_sim:
            break

    return pd.DataFrame(results)


def run_single_episode_test(
        episode_id: int,
        model_predict: Any,
        model_aleatoric: Any,
        model_enn: Any
) -> None:
    """Runs prediction and analysis for a single specific episode."""
    print(f"\n[TEST] Running Single Episode Test. ID: {episode_id}")

    env = grid2op.make(CFG.ENV_NAME, backend=LightSimBackend())
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("[WARN] Agent could not be loaded, using random/do-nothing.")

    obs = env.reset(seed=episode_id)
    done = False
    observations_array = []
    results_list = []

    while not done:
        print(f"[TEST] Ep {episode_id} | Step {obs.current_step}")
        observations_array.append(obs)

        # Real-time Agent Action
        action = agent.act(obs, 0.0, done)

        # Analyze security every 20 steps (after initial forecasting lag)
        if obs.current_step > 12 and obs.current_step % 20 == 0:
            df = analyze_disconnection_effect(
                env, model_predict, model_aleatoric, model_enn,
                obs, observations_array, episode_id, agent
            )
            if not df.empty:
                df['phase'] = "SingleTest"
                results_list.append(df)

                # Immediate terminal feedback
                failures = df[df['failed'] == 1]
                if not failures.empty:
                    print(f"    -> WARNING: Potential failures: {failures['line_disconnected'].tolist()}")

        # Execute Real Environment Step
        obs, _, done, _ = env.step(action)

    if results_list:
        full_df = pd.concat(results_list)
        save_incremental(full_df, CFG.CSV_OUTPUT_PATH)
        print(f"[TEST] Episode {episode_id} complete. Results saved.")


def run_simulation_phase(
        phase_name: str,
        env_params: Dict[str, Any],
        ep_start: int,
        ep_end: int,
        model_predict: Any,
        model_aleatoric: Any,
        model_enn: Any
) -> None:
    """Runs a batch of episodes for training data collection."""
    print(f"\n>>> STARTING PHASE: {phase_name} (Eps {ep_start}-{ep_end})")

    env = grid2op.make(CFG.ENV_NAME, backend=LightSimBackend(), **env_params)
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("[WARN] Agent not loaded.")

    for ep in range(ep_end):
        if ep < ep_start:
            continue

        obs = env.reset(seed=ep)
        done = False
        observations_array = []

        while not done:
            observations_array.append(obs)

            # Real Agent Action
            action = agent.act(obs, 0.0, done)

            # Trigger analysis
            if obs.current_step > 12 and obs.current_step % 20 == 0:
                df = analyze_disconnection_effect(
                    env, model_predict, model_aleatoric, model_enn,
                    obs, observations_array, ep, agent
                )
                if not df.empty:
                    df['phase'] = phase_name
                    save_incremental(df, CFG.CSV_OUTPUT_PATH)

            # Advance Environment
            obs, _, done, _ = env.step(action)


if __name__ == "__main__":

    output_dir = os.path.dirname(CFG.CSV_OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório criado: {output_dir}")


    # Environment Setup
    if TRAIN_MODE:
        if os.path.exists(CFG.CSV_OUTPUT_PATH):
            os.remove(CFG.CSV_OUTPUT_PATH)

    # Load All Pre-trained Models
    print(">>> Loading Trained Models...")
    try:
        model_predict = joblib.load(CFG.MODEL_MEAN_PATH)
        model_aleatoric = joblib.load(CFG.MODEL_ALEATORIC_PATH)

        model_enn = EvidentialNetwork(
            ENN_PARAMS['input_dim'], ENN_PARAMS['num_classes'],
            ENN_PARAMS['hidden_dim'], ENN_DROPOUT
        ).to(DEVICE)

        try:
            model_enn.load_state_dict(torch.load(CFG.MODEL_ENN_PATH, map_location=DEVICE))
            print(">>> ENN Weights loaded successfully.")
        except Exception:
            print("[WARN] ENN weights not found (using random initialization).")

        model_enn.eval()
    except Exception as e:
        print(f"Error loading models: {e}. Ensure the training pipeline was completed.")
        exit(1)

    # Run execution based on active configuration mode
    if TEST_SINGLE_EPISODE:
        run_single_episode_test(
            CFG.EPISODE_ID_TO_TEST,
            model_predict, model_aleatoric, model_enn
        )
    else:
        # Standard Data Collection Phase
        run_simulation_phase("Standard", {}, 700, 730, model_predict, model_aleatoric, model_enn)

        # Optional Attack Scenarios
        opp_kwargs = {
            "opponent_action_class": PowerlineSetAction,
            "opponent_class": RandomLineOpponent,
            "opponent_budget_class": BaseActionBudget,
            "kwargs_opponent": {"lines_attacked": CFG.LINES_TO_TEST, "seed": 42}
        }
        run_simulation_phase("HeavyAttack_1", {**opp_kwargs, "opponent_attack_cooldown": 288}, 700, 750, model_predict, model_aleatoric, model_enn)
        run_simulation_phase("HeavyAttack_2", {**opp_kwargs, "opponent_attack_cooldown": 128}, 700, 750, model_predict, model_aleatoric, model_enn)