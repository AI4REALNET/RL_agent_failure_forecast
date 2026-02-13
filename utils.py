import os
import numpy as np
import datetime
from typing import Tuple, List, Dict, Any, Optional

# Import the active configuration
from config import CFG

def convert_to_cos_sin(value: int, max_value: int) -> Tuple[float, float]:
    """Converts cyclic variables (hours, days) into sin/cos features."""
    val_cos = np.cos(2 * np.pi * value / max_value)
    val_sin = np.sin(2 * np.pi * value / max_value)
    return val_cos, val_sin

def get_features(observations_array: List[Any], obs: Any, step: int) -> Tuple[np.ndarray, int]:
    """
    Constructs the feature vector for forecasting models.
    """
    current_index = obs.current_step + step

    def get_past(array: List[Any], idx: int, steps: int) -> Any:
        past_index = idx - steps - 1
        if 0 <= past_index < len(array):
            return array[past_index]
        return None

    def extract_features(past_obs: Any, attr: str, size: int) -> np.ndarray:
        if past_obs is not None:
            return getattr(past_obs, attr)
        return np.zeros(size)

    past_obs_hour = get_past(observations_array, current_index, 12)
    past_obs_day = get_past(observations_array, current_index, 288)
    past_obs_week = get_past(observations_array, current_index, 2016)

    feature_vec = np.concatenate([
        extract_features(past_obs_week, 'load_p', CFG.NO_LOADS),
        extract_features(past_obs_day, 'load_p', CFG.NO_LOADS),
        extract_features(past_obs_hour, 'load_p', CFG.NO_LOADS),
        extract_features(past_obs_week, 'load_q', CFG.NO_LOADS),
        extract_features(past_obs_day, 'load_q', CFG.NO_LOADS),
        extract_features(past_obs_hour, 'load_q', CFG.NO_LOADS),
        extract_features(past_obs_week, 'gen_p', CFG.NO_GENS),
        extract_features(past_obs_day, 'gen_p', CFG.NO_GENS),
        extract_features(past_obs_hour, 'gen_p', CFG.NO_GENS),
    ])

    temporal_features = []
    if step == 0:
        h_cos, h_sin = convert_to_cos_sin(obs.hour_of_day, 23)
        m_cos, m_sin = convert_to_cos_sin(obs.minute_of_hour, 59)
        d_cos, d_sin = convert_to_cos_sin(obs.day_of_week, 6)
        temporal_features = [obs.day, h_cos, h_sin, m_cos, m_sin, d_cos, d_sin]
    else:
        # Simulate timestamp for future steps
        atual_ts = obs.get_time_stamp()
        new_ts = atual_ts + datetime.timedelta(minutes=step * 5)
        temporal_features = [
            new_ts.day,
            *convert_to_cos_sin(new_ts.hour, 23),
            *convert_to_cos_sin(new_ts.minute, 59),
            *convert_to_cos_sin(new_ts.weekday(), 6)
        ]

    # Return shape (1, n_features) and dummy timestamp
    return np.concatenate([feature_vec, temporal_features]).reshape(1, -1).astype(float), 0

def compute_grid_stats(obs: Any) -> Dict[str, float]:
    """Calculates basic grid statistics from an observation."""
    rho = obs.rho
    stats = {
        "sum_load_p": float(np.sum(obs.load_p)),
        "sum_load_q": float(np.sum(obs.load_q)),
        "sum_gen_p": float(np.sum(obs.gen_p)),
        "sum_gen_q": float(np.sum(obs.gen_q)),
        "var_line_rho": float(np.var(rho)),
        "avg_line_rho": float(np.mean(rho)),
        "max_line_rho": float(np.max(rho)),
        "nb_rho_ge_0.95": int(np.sum(rho >= 0.95))
    }
    return stats

def append_to_npy(file_path: str, array: np.ndarray) -> None:
    """Appends data to an .npy file, creating it if it doesn't exist."""
    if os.path.exists(file_path):
        old_data = np.load(file_path, allow_pickle=True)
        new_data = np.concatenate([old_data, array], axis=0)
    else:
        new_data = array
    np.save(file_path, new_data)