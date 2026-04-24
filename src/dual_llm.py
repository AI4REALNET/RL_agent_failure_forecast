import os
import re
import json
import math
import time
import random
import traceback
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix
from config import CFG, OUTPUT_DIR_LLM

# ==========================================
# FEATURE DEFINITIONS
# ==========================================

FEATURES: List[str] = [
    "line_id_encoded",       # Integer encoding of the disconnected power line (from LINE_MAP)
    "sum_load_p",            # Total active (real) power load across all buses [MW]
    "sum_load_q",            # Total reactive power load [MVAr]
    "sum_gen_p",             # Total active generation [MW]
    "var_line_rho",          # Variance of line loading ratios (spread of congestion)
    "avg_line_rho",          # Mean line loading ratio across all monitored lines
    "max_line_rho",          # Maximum line loading ratio (worst-case congestion)
    "nb_rho_ge_0.95",        # Count of lines with rho >= 0.95 (near-overload count)
    "aleatoric_load_p_mean", # Mean aleatoric uncertainty of the active load forecast
    "aleatoric_load_q_mean", # Mean aleatoric uncertainty of the reactive load forecast
    "aleatoric_gen_p_mean",  # Mean aleatoric uncertainty of the generation forecast
    "load_gen_ratio",        # Active load / active generation ratio (computed in main)
    "epistemic_before",      # Epistemic (model) uncertainty at current timestep t
    "epistemic_after",       # Epistemic uncertainty at t+12 (before the disconnection event)
    # Forecasted grid state at t+12 (horizon before the disconnection)
    "fcast_sum_load_p",
    "fcast_sum_load_q",
    "fcast_sum_gen_p",
    "fcast_var_line_rho",
    "fcast_avg_line_rho",
    "fcast_max_line_rho",
    "fcast_nb_rho_ge_0.95",
]

# Features exposed to the LLM: exclude line_id_encoded because each rule is
# already generated per-line, so using it inside a rule is trivial and
# provides no explanatory value. Rules that reference line_id_encoded are
# also rejected by validate_rule_code.
FEATURES_FOR_LLM: List[str] = [f for f in FEATURES if f != "line_id_encoded"]

FEATURE_DESCRIPTIONS = {
    "line_id_encoded": "Integer encoding of the disconnected line used by the HGB teacher.",
    "sum_load_p": "Total active load.",
    "sum_load_q": "Total reactive load.",
    "sum_gen_p": "Total active generation.",
    "var_line_rho": "Variance of line loading rho values.",
    "avg_line_rho": "Average line loading rho.",
    "max_line_rho": "Maximum line loading rho.",
    "nb_rho_ge_0.95": "Number of lines with rho >= 0.95.",
    "aleatoric_load_p_mean": "Mean aleatoric uncertainty for active load.",
    "aleatoric_load_q_mean": "Mean aleatoric uncertainty for reactive load.",
    "aleatoric_gen_p_mean": "Mean aleatoric uncertainty for generation.",
    "load_gen_ratio": "Ratio between load and generation.",
    "epistemic_before": "Epistemic uncertainty at timestep t.",
    "epistemic_after": "Epistemic uncertainty at timestep t+12, before the disconnection.",
    # fcast_ features: grid state forecasted at t+12, before the disconnection
    "fcast_sum_load_p": "Forecasted total active load at t+12 (before disconnection).",
    "fcast_sum_load_q": "Forecasted total reactive load at t+12 (before disconnection).",
    "fcast_sum_gen_p": "Forecasted total active generation at t+12 (before disconnection).",
    "fcast_var_line_rho": "Forecasted variance of line loading rho at t+12 (before disconnection).",
    "fcast_avg_line_rho": "Forecasted average line loading rho at t+12 (before disconnection).",
    "fcast_max_line_rho": "Forecasted maximum line loading rho at t+12 (before disconnection).",
    "fcast_nb_rho_ge_0.95": "Forecasted number of lines with rho >= 0.95 at t+12 (before disconnection).",
}

# ==========================================
# LINE MAP
# ==========================================

LINE_MAP: Dict[str, int] = {
    "34_35_110": 0,
    "39_41_121": 1,
    "41_48_131": 2,
    "43_44_125": 3,
    "44_45_126": 4,
    "48_50_136": 5,
    "48_53_141": 6,
    "54_58_154": 7,
    "62_58_180": 8,
    "62_63_160": 9,
}
"""Maps power-line string labels (fromBus_toBus_lineIndex) to integer encodings
used by the HGB teacher model. Each key corresponds to one distillation experiment."""

# ==========================================
# EXPERIMENT HYPERPARAMETERS
# ==========================================

TEMPERATURES: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
"""LLM sampling temperatures swept across experiments.
Lower = more deterministic rules; higher = more structural exploration."""

ITERATIONS: int = 500
"""Number of Generator–Critic loop iterations per (line, temperature) run."""

# CHANGE 3: window parameters aligned with paper
WINDOW_SIZE: int = 120
"""Maximum total samples in the balanced context window shown to the LLM per iteration."""

MAX_POSITIVE_IN_WINDOW: int = 40
"""Maximum positive (failure) samples allowed in one context window.
Paper specifies 'up to 40 positive examples'."""

NEGATIVE_RATIO: float = 2.0
"""Target negative-to-positive ratio for context window sampling.
Paper specifies '2:1 ratio'; excess budget filled from remaining training data."""

# ==========================================
# SPLIT THRESHOLDS
# ==========================================

MIN_POS_TRAIN: int = 10
"""Minimum number of positive samples required in the training partition."""

MIN_POS_TEST: int = 5
"""Minimum number of positive samples required in the test partition."""

MIN_TOTAL_SAMPLES_PER_LINE: int = 20
"""Absolute minimum total rows for a line to be processed via standard split."""


MIN_POS_STANDARD: int = MIN_POS_TRAIN + MIN_POS_TEST   # = 15; threshold for standard split
MIN_POS_LOO_THRESHOLD: int = 4    # lines with fewer positives use Leave-One-Out
MIN_TOTAL_SAMPLES_SPARSE: int = 6 # absolute floor to attempt any experiment

# ==========================================
# PROMPT LENGTH LIMITS
# ==========================================

MAX_HISTORY_ITEMS_FOR_PROMPT: int = 6
"""Number of recent iterations included in the history table sent to both LLMs."""

MAX_WORST_CASES_FOR_PROMPT: int = 10
"""Maximum number of misclassified test samples shown to both LLMs."""

MAX_TRAINING_JSON_CHARS: int = 10000
"""Hard character cap on the JSON training window in the Generator prompt."""

MAX_FEEDBACK_CHARS: int = 4000
"""Hard character cap on the Critic feedback stored and forwarded to next iteration."""

MAX_RULE_CHARS: int = 12000
"""Hard character cap on rule code strings in all prompts."""

MAX_JUSTIFICATION_CHARS: int = 3000
"""Hard character cap on the Generator's justification section."""

MAX_CHANGES_CHARS: int = 2000
"""Hard character cap on the Generator's change-log section."""

# ==========================================
# API / RETRY CONFIGURATION
# ==========================================

API_TIMEOUT: int = 180
"""Seconds before a single LLM API call times out."""

API_RETRIES: int = 3
"""Number of retry attempts before raising a RuntimeError on API failure."""

SLEEP_BETWEEN_RETRIES: int = 3
"""Seconds to sleep between retry attempts (reduces load on the API endpoint)."""

# ==========================================
# REPRODUCIBILITY
# ==========================================

RANDOM_SEED: int = 42
"""Global seed applied to Python's random module and NumPy for reproducible sampling."""
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================
# TEACHER PERFORMANCE TARGETS (CHANGE 2)
# ==========================================

# CHANGE 2: Real HGB target metrics (measured on test set)
HGB_TARGET: Dict[str, float] = {
    "acc": 0.8951,   # Accuracy of the HGB teacher on the full test set
    "f2":  0.7643,   # F2 score (beta=2, recall-weighted) of the teacher
    "ovr": 0.0679,   # Omission rate (false negative rate) of the teacher
    "fa":  0.1084,   # False alarm rate (false positive rate) of the teacher
}
"""Performance benchmarks from the trained HGB teacher, measured on the held-out test set.
Injected into both Generator and Critic prompts as the reference targets to reach."""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Suppresses SSL warnings that arise when verify_ssl=False is set in the config
# (used in intranet environments with self-signed certificates).

# ==========================================
# CORE UTILITIES
# ==========================================

def ensure_dir(path: str) -> None:
    """Creates a directory (and all missing parents) if it does not already exist.
    Safe to call multiple times — will not raise if the directory is already present."""
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> float:
    """Converts an arbitrary value to float, returning ``float('nan')`` for None
    or any value that cannot be coerced (e.g. strings, objects).
    Used for robust handling of LLM-returned metric values that may be malformed."""
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def clip_text(text: Optional[str], max_chars: int) -> str:
    """Truncates a string to ``max_chars`` characters.
    Returns an empty string if ``text`` is None. Used to prevent prompt overflows
    when injecting long rule code or Critic feedback into LLM prompts."""
    if text is None:
        return ""
    text = str(text)
    return text[:max_chars]


def dict_to_pretty_json(d: Any) -> str:
    """Serialises a value to an indented JSON string for LLM prompt injection.
    Uses ``default=str`` so non-serialisable types (e.g. numpy ints) are handled gracefully."""
    return json.dumps(d, ensure_ascii=False, indent=2, default=str)


def load_llm_config() -> Dict[str, Any]:
    """Reads and returns the LLM API configuration from ``LLM_CONFIG_FILE_PATH``.

    Re-read on every call so credentials or model settings can be updated
    between iterations without restarting the process.

    Expected JSON keys:
        ``api_key``    – Bearer token for the API endpoint.
        ``api_url``    – Full URL of an OpenAI-compatible ``/chat/completions`` endpoint.
        ``model``      – Model identifier string.
        ``max_tokens`` – Maximum tokens in the model response (default: 4000).
        ``verify_ssl`` – Whether to verify SSL certificates (default: True).
        ``timeout``    – Per-request timeout in seconds (overrides ``API_TIMEOUT``).
    """
    with open(LLM_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# METRICS & SCORING
# ==========================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes all evaluation metrics for a binary prediction array.
    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dictionary with keys: acc, f2, fa, ovr, tp, fp, fn, tn.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fa = fp / (fp + tn + 1e-12)
    ovr = fn / (fn + tp + 1e-12)

    return {
        "acc": float(acc),
        "f2": float(f2),
        "fa": float(fa),
        "ovr": float(ovr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def scoring_function(m: Dict[str, float]) -> float:
    """
    Hard floors prevent the two degenerate extremes from dominating:
      - F2 == 0 (never predicts 1)  → -1.0
      - FA >= 0.90 (almost always predicts 1) → -0.90
    Strong progressive penalties keep the LLM in the corridor
    OVR ∈ [0.04, 0.20] and FA ∈ [0.05, 0.25].
    """
    # Hard floor 1: rule never catches any failure
    if m["f2"] == 0.0:
        return -1.0

    # Hard floor 2: rule fires on almost everything (predict-all-1 is useless)
    if m["fa"] >= 0.90:
        return -0.90

    # Base score — paper weights
    base = (
        0.10 * m["acc"]
        + 0.40 * m["f2"]
        + 0.35 * (1.0 - m["ovr"])
        + 0.15 * (1.0 - m["fa"])
    )

    # OVR penalty — two tiers:
    #   mild above target (0.10): linear up to -0.15
    #   severe above 0.50 (oscillation zone): extra -0.30
    ovr_mild   = min(0.15, max(0.0, m["ovr"] - 0.10) * 0.375)
    ovr_severe = min(0.30, max(0.0, m["ovr"] - 0.50) * 0.75)
    ovr_penalty = ovr_mild + ovr_severe

    # FA penalty — two tiers (symmetric logic):
    #   mild above target (0.20): linear up to -0.10
    #   severe above 0.60: extra -0.20
    fa_mild   = min(0.10, max(0.0, m["fa"] - 0.20) * 0.25)
    fa_severe = min(0.20, max(0.0, m["fa"] - 0.60) * 0.50)
    fa_penalty = fa_mild + fa_severe

    score = base - ovr_penalty - fa_penalty

    # Floor just above the predict-all-1 hard floor
    return max(-0.89, score)


# ==========================================
# DATA PREPARATION
# ==========================================

def validate_required_columns(df: pd.DataFrame) -> None:
    """Asserts that all required feature columns and 'line_disconnected' are present
    in the DataFrame. Raises ``ValueError`` listing any missing columns.
    """
    required = set(FEATURES + ["line_disconnected"])
    missing = [c for c in required if c not in df.columns and c != "line_id_encoded"]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def encode_line_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Maps the ``line_disconnected`` string column to integer encodings via ``LINE_MAP``
    and stores the result in a new ``line_id_encoded`` column.

    Raises ``ValueError`` if any value in ``line_disconnected`` is absent from ``LINE_MAP``,
    listing the unrecognised line names to aid debugging.
    """
    df = df.copy()
    df["line_id_encoded"] = df["line_disconnected"].map(LINE_MAP)
    if df["line_id_encoded"].isna().any():
        unknown = sorted(df.loc[df["line_id_encoded"].isna(), "line_disconnected"].dropna().unique().tolist())
        raise ValueError(f"Found line_disconnected values not present in LINE_MAP: {unknown}")
    df["line_id_encoded"] = df["line_id_encoded"].astype(int)
    return df


def build_teacher_target(df: pd.DataFrame) -> pd.DataFrame:
    """Generates the ``target`` column that the LLM rules will learn to imitate.

    Primary path:
        Loads the pre-trained HGB model from ``CFG.MODEL_CLASSIFIER_PATH`` and runs inference
        on the full feature matrix. The resulting predictions become the distillation
        targets (label = 1 means the teacher predicts a disconnection risk).
        Sets ``target_source = 'teacher_hgb'``.

    Fallback path (if model file does not exist):
        Uses the ``failed`` column directly as binary ground truth.
        Sets ``target_source = 'failed_fallback'``.
        Raises ``ValueError`` if ``failed`` is also missing.

    Args:
        df: Full dataset with all FEATURES columns and ``load_gen_ratio`` already present.

    Returns:
        Copy of ``df`` with new columns ``target`` (int) and ``target_source`` (str).
    """
    df = df.copy()

    if CFG.MODEL_CLASSIFIER_PATH is not None and os.path.exists(CFG.MODEL_CLASSIFIER_PATH):
        model = joblib.load(CFG.MODEL_CLASSIFIER_PATH)
        X_teacher = df[FEATURES].copy()
        df["target"] = model.predict(X_teacher).astype(int)
        df["target_source"] = "teacher_hgb"
    else:
        if "failed" not in df.columns:
            raise ValueError("HGB model not found and fallback column 'failed' is missing.")
        df["target"] = df["failed"].astype(int)
        df["target_source"] = "failed_fallback"

    return df



# ==========================================
# TRAIN / TEST SPLIT STRATEGIES
# ==========================================

def sequential_guarded_split(df_line: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standard chronological split for lines with >= MIN_POS_STANDARD (15) total positives.

    Scans 11 candidate cut points from 60% to 85% of the sorted timeline and selects
    the first split that satisfies MIN_POS_TRAIN=10 in train and MIN_POS_TEST=5 in test.

    Raises ``RuntimeError`` if no valid split can be found (e.g. all failures clustered
    at one end of the timeline).
    """
    df_line = df_line.sort_index().reset_index(drop=True)

    if len(df_line) < MIN_TOTAL_SAMPLES_PER_LINE:
        raise RuntimeError(
            f"Not enough samples for this line: {len(df_line)} < {MIN_TOTAL_SAMPLES_PER_LINE}"
        )

    candidate_splits = np.linspace(0.60, 0.85, 11)

    for split in candidate_splits:
        cut = int(len(df_line) * split)
        train = df_line.iloc[:cut].copy()
        test = df_line.iloc[cut:].copy()

        n_pos_train = int(train["target"].sum())
        n_pos_test = int(test["target"].sum())

        if n_pos_train >= MIN_POS_TRAIN and n_pos_test >= MIN_POS_TEST:
            return train, test

    raise RuntimeError(
        f"Could not find a sequential split with at least "
        f"{MIN_POS_TRAIN} positives in train and {MIN_POS_TEST} positives in test."
    )


def adaptive_split(df_line: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Scans a wider range (50%–90%, 17 candidates) with a relaxed constraint: the train
    partition must contain at least (n_pos_total - 2) positives and the test at least 1.

    Last-resort fallback: if no cut satisfies the constraint, splits at the index of the
    last positive event so that it falls in the test set.

    Raises ``RuntimeError`` if the line has fewer than MIN_TOTAL_SAMPLES_SPARSE total rows
    or if no valid split can be constructed at all.
    """
    df_line = df_line.sort_index().reset_index(drop=True)
    n_pos_total = int(df_line["target"].sum())

    if len(df_line) < MIN_TOTAL_SAMPLES_SPARSE:
        raise RuntimeError(
            f"Sparse line has too few total rows: {len(df_line)} < {MIN_TOTAL_SAMPLES_SPARSE}"
        )

    candidate_splits = np.linspace(0.50, 0.90, 17)

    for split in candidate_splits:
        cut = int(len(df_line) * split)
        train = df_line.iloc[:cut].copy()
        test = df_line.iloc[cut:].copy()

        n_pos_train = int(train["target"].sum())
        n_pos_test = int(test["target"].sum())

        if n_pos_train >= max(2, n_pos_total - 2) and n_pos_test >= 1:
            print(f"  [adaptive_split] split={split:.2f} -> train_pos={n_pos_train}, test_pos={n_pos_test}")
            return train, test

    pos_indices = df_line.index[df_line["target"] == 1].tolist()
    if len(pos_indices) >= 2:
        last_pos_idx = pos_indices[-1]
        train = df_line.iloc[:last_pos_idx].copy()
        test = df_line.iloc[last_pos_idx:].copy()
        if int(train["target"].sum()) >= 1 and int(test["target"].sum()) >= 1:
            print(f"  [adaptive_split] last-resort cut at idx={last_pos_idx}")
            return train, test

    raise RuntimeError(
        f"adaptive_split failed: no valid split found for line with {n_pos_total} positives."
    )


def leave_one_out_splits(df_line: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """

    Each positive event becomes a single-row test fold; all remaining rows form the training
    set. Only folds where the training set is non-empty are returned.

    The caller (``run_line_experiment``) runs the full distillation loop per fold and
    selects the fold with the highest ``best_score`` as the representative result.

    Raises ``RuntimeError`` if no valid folds can be constructed (e.g. only one row total).
    """
    df_line = df_line.sort_index().reset_index(drop=True)
    pos_indices = df_line.index[df_line["target"] == 1].tolist()

    splits = []
    for idx in pos_indices:
        test = df_line.loc[[idx]].copy()
        train = df_line.drop(index=idx).copy()
        if len(train) > 0:
            splits.append((train, test))

    if not splits:
        raise RuntimeError("leave_one_out_splits: no valid LOO folds could be constructed.")

    return splits


# ==========================================
# WINDOW & FEATURE STATISTICS
# ==========================================

# CHANGE 3: MAX_POSITIVE_IN_WINDOW=40, NEGATIVE_RATIO=2.0
def build_balanced_window(train_df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a balanced context window from the training set for the Generator prompt.

    Sampling strategy (paper-aligned):
        1. Sample up to MAX_POSITIVE_IN_WINDOW=40 positives (with replacement if needed).
        2. Sample up to NEGATIVE_RATIO * n_positives = 2× as many negatives.
        3. Fill any remaining capacity up to WINDOW_SIZE=120 from unused training rows.
        4. Sort by original index to preserve temporal ordering for the LLM.

    If the training set has no positives, returns the first WINDOW_SIZE rows as-is
    (degenerate case — the LLM will see only negatives and should output rule→0).
    """
    positives = train_df[train_df["target"] == 1]
    negatives = train_df[train_df["target"] == 0]

    if len(positives) == 0:
        return train_df.head(min(WINDOW_SIZE, len(train_df))).copy()

    pos_n = min(len(positives), MAX_POSITIVE_IN_WINDOW)
    pos_sample = positives.sample(n=pos_n, random_state=random.randint(0, 1_000_000))

    # paper: 2:1 negatives-to-positives ratio
    neg_n = min(len(negatives), max(1, int(math.ceil(pos_n * NEGATIVE_RATIO))))
    neg_sample = negatives.sample(n=neg_n, random_state=random.randint(0, 1_000_000)) if neg_n > 0 else negatives.head(0)

    window = pd.concat([pos_sample, neg_sample], axis=0)

    remaining_n = max(0, WINDOW_SIZE - len(window))
    if remaining_n > 0:
        remaining = train_df.drop(index=window.index, errors="ignore")
        if len(remaining) > 0:
            add_n = min(len(remaining), remaining_n)
            add_sample = remaining.sample(n=add_n, random_state=random.randint(0, 1_000_000))
            window = pd.concat([window, add_sample], axis=0)

    window = window.sort_index().copy()
    return window


# CHANGE 4: feature statistics as min/mean/max (paper), anchors kept as supplementary
def summarize_window(window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Computes a compact statistical summary of the context window for the LLM.

    Per the paper: *'statistical description of the features (minimum, mean and maximum)'*.

    Returns a dict with:
        ``n_rows``        – Total rows in the window.
        ``n_positive``    – Count of positive (failure) rows.
        ``positive_rate`` – Fraction of positive rows.
        ``feature_stats`` – Per-feature dict of ``{min, mean, max}`` (NaN-safe).
    """
    summary = {
        "n_rows": int(len(window)),
        "n_positive": int(window["target"].sum()),
        "positive_rate": float(window["target"].mean()) if len(window) > 0 else float("nan"),
    }
    numeric_summary = {}
    for c in feature_cols:
        s = pd.to_numeric(window[c], errors="coerce")
        numeric_summary[c] = {
            "min":  safe_float(s.min()),
            "mean": safe_float(s.mean()),
            "max":  safe_float(s.max()),
        }
    summary["feature_stats"] = numeric_summary
    return summary


def compute_feature_anchors(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """Computes per-feature quartile statistics split by target class (failure vs normal).

    These anchors give the LLM concrete, data-driven threshold candidates that go beyond
    simple global min/mean/max. The dictionary is sorted by ``|median_diff|`` descending
    so the most discriminative features appear first in prompts.

    Per-feature output keys:
        ``pos_p25/p50/p75`` – Quartiles over failure rows (target=1).
        ``neg_p25/p50/p75`` – Quartiles over normal rows (target=0).
        ``median_diff``      – pos_median - neg_median (sign indicates direction).
        ``suggested_threshold`` – Midpoint between class medians; starting threshold
                                  for both seed rules and LLM suggestions.

    Features are skipped if either class has no non-NaN values.
    """
    anchors = {}
    pos = train_df[train_df["target"] == 1]
    neg = train_df[train_df["target"] == 0]

    for f in feature_cols:
        s_pos = pd.to_numeric(pos[f], errors="coerce").dropna()
        s_neg = pd.to_numeric(neg[f], errors="coerce").dropna()

        if len(s_pos) == 0 or len(s_neg) == 0:
            continue

        anchors[f] = {
            "pos_p25": round(float(s_pos.quantile(0.25)), 4),
            "pos_p50": round(float(s_pos.quantile(0.50)), 4),
            "pos_p75": round(float(s_pos.quantile(0.75)), 4),
            "neg_p25": round(float(s_neg.quantile(0.25)), 4),
            "neg_p50": round(float(s_neg.quantile(0.50)), 4),
            "neg_p75": round(float(s_neg.quantile(0.75)), 4),
            "median_diff": round(float(s_pos.median() - s_neg.median()), 4),
            "suggested_threshold": round(float((s_pos.median() + s_neg.median()) / 2.0), 4),
        }

    anchors = dict(
        sorted(anchors.items(), key=lambda kv: abs(kv[1]["median_diff"]), reverse=True)
    )
    return anchors


def compute_seed_rules(
    train_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_anchors: Dict[str, Any],
    top_n_features: int = 6,
) -> List[Dict[str, Any]]:
    """
    Generate and evaluate candidate seed rules from the data directly, before
    any LLM iterations. Returns a list of (rule_code, metrics, score) dicts
    sorted by score descending.

    Strategy:
      1. For each of the top-N most discriminative features, try 5 thresholds
         around the suggested_threshold (the midpoint between class medians).
         Direction is determined by median_diff: if positive (feature higher in
         failures), threshold is x[f] >= t; otherwise x[f] <= t.
      2. For the top-2 feature pairs, try AND combinations of the best
         single-feature thresholds.

    All candidates are evaluated on the test set. The best seed is used to
    initialise best_rule/best_metrics/best_score before iteration 1.
    """
    top_features = list(feature_anchors.keys())[:top_n_features]
    candidates: List[Dict[str, Any]] = []

    def _make_rule(conditions: List[Tuple[str, str, float]]) -> str:
        """conditions: list of (feature, op, threshold) — nested if/else."""
        if not conditions:
            return ""
        # Build nested if/else for AND logic (all conditions must be true → return 1)
        indent = "    "
        lines = ["def rule(x):"]
        depth = 0
        for feat, op, thresh in conditions:
            lines.append(f"{indent * (depth + 1)}if x[\"{feat}\"] {op} {thresh}:")
            depth += 1
        lines.append(f"{indent * (depth + 1)}return 1")
        # unwind else ladder
        for _ in conditions:
            depth -= 1
            lines.append(f"{indent * (depth + 1)}else:")
            lines.append(f"{indent * (depth + 2)}return 0")
            if depth == 0:
                break
        return "\n".join(lines)

    def _eval_candidate(rule_code: str) -> Optional[Dict[str, Any]]:
        try:
            valid, err = validate_rule_code(rule_code)
            if not valid:
                return None
            y_pred = evaluate_rule(rule_code, X_test)
            m = compute_metrics(y_test, y_pred)
            s = scoring_function(m)
            return {"rule_code": rule_code, "score": s, **m}
        except Exception:
            return None

    # --- Univariate candidates ---
    for feat in top_features:
        if feat not in feature_anchors:
            continue
        anchor = feature_anchors[feat]
        suggested = anchor["suggested_threshold"]
        median_diff = anchor["median_diff"]
        op = ">=" if median_diff > 0 else "<="

        # Try 5 thresholds: suggested ± 10%, ± 20%, exactly suggested
        for factor in [-0.20, -0.10, 0.0, 0.10, 0.20]:
            t = round(suggested * (1.0 + factor), 6)
            rule_code = _make_rule([(feat, op, t)])
            result = _eval_candidate(rule_code)
            if result:
                candidates.append(result)

    # --- Bivariate candidates: top-2 feature pairs ---
    for i in range(min(3, len(top_features))):
        for j in range(i + 1, min(4, len(top_features))):
            f1, f2 = top_features[i], top_features[j]
            if f1 not in feature_anchors or f2 not in feature_anchors:
                continue
            t1 = feature_anchors[f1]["suggested_threshold"]
            op1 = ">=" if feature_anchors[f1]["median_diff"] > 0 else "<="
            t2 = feature_anchors[f2]["suggested_threshold"]
            op2 = ">=" if feature_anchors[f2]["median_diff"] > 0 else "<="

            # Try (f1 AND f2) with suggested thresholds and slight relaxations
            for fac1 in [-0.10, 0.0, 0.10]:
                for fac2 in [-0.10, 0.0, 0.10]:
                    tt1 = round(t1 * (1.0 + fac1), 6)
                    tt2 = round(t2 * (1.0 + fac2), 6)
                    rule_code = _make_rule([(f1, op1, tt1), (f2, op2, tt2)])
                    result = _eval_candidate(rule_code)
                    if result:
                        candidates.append(result)

    # Sort by score, return top-10 unique by rule_code
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c["rule_code"] not in seen:
            seen.add(c["rule_code"])
            unique.append(c)
        if len(unique) >= 10:
            break

    return unique


# ==========================================
# RULE EXTRACTION & VALIDATION
# ==========================================

# Compiled regex to extract feature names referenced as x["feature"] or x['feature']
RULE_FEATURE_REGEX = re.compile(r'x\[(?:"|\')([^"\']+)(?:"|\')\]')

# Delimiters used in LLM prompts to structure the response into parseable sections
TAG_PATTERNS: Dict[str, Tuple] = {
    "justification": (r"\[Start of Justification\](.*?)\[End of Justification\]", re.DOTALL | re.IGNORECASE),
    "changes":       (r"\[Start of Changes\](.*?)\[End of Changes\]",             re.DOTALL | re.IGNORECASE),
    "rule":          (r"\[Start of Rule\](.*?)\[End of Rule\]",                   re.DOTALL | re.IGNORECASE),
}


def extract_tagged_section(text: str, section_name: str) -> str:
    """Extracts the content between a ``[Start of X]`` / ``[End of X]`` delimiter pair.

    Used to parse the three structured sections that every Generator response must contain:
    ``justification``, ``changes``, and ``rule``. Returns an empty string if the section
    is absent (e.g. the LLM forgot the tags).
    """
    pattern, flags = TAG_PATTERNS[section_name]
    m = re.search(pattern, text, flags)
    if m:
        return m.group(1).strip()
    return ""


def extract_python_rule(text: str) -> str:
    """Extracts the Python ``def rule(x):`` function from a raw LLM response string.

    Extraction strategy (applied in priority order):
        1. Content inside ``[Start of Rule] ... [End of Rule]`` tags
        2. Content inside a fenced ``python`` code block (```python ... ```).
        3. Fallback: regex search for any ``def rule(`` signature in the response,
           stopping at the first unindented non-code line.

    Returns an empty string if no ``def rule`` can be located.
    """
    if not text:
        return ""

    text = text.strip()

    # Priority 1: tagged section
    tagged = extract_tagged_section(text, "rule")
    if tagged and "def rule" in tagged:
        text = tagged

    # Priority 2: fenced code block
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        if "def rule" in block:
            return block.strip()

    # Priority 3: bare def rule(...) anywhere in the response
    m = re.search(r"(def\s+rule\s*\(\s*\w+\s*\)\s*:\s*[\s\S]*)", text)
    if m:
        candidate = m.group(1).strip()
        lines = candidate.splitlines()
        cleaned = []
        for line in lines:
            if not line.strip() and cleaned:
                cleaned.append(line)
                continue
            # Stop at the first unindented non-blank token (prose after the function)
            if cleaned and re.match(r"^[A-Za-z\[\]()_-]", line) and not line.startswith((" ", "\t")):
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    return ""


def validate_rule_code(rule_code: str) -> Tuple[bool, str]:
    """Validates a generated rule string before execution on the test set.

    Checks applied (in order):
        1. Must define ``def rule``.
        2. Must not contain forbidden tokens (imports, file I/O, eval/exec, os/sys/subprocess,
           pickle, joblib, or ``line_id_encoded`` — see list below).
        3. Must only reference features present in ``FEATURES_FOR_LLM``.
        4. Must compile without syntax errors.

    Returns:
        ``(True, "")`` if valid, or ``(False, reason_string)`` if invalid.
        The reason string is forwarded to the repair prompt and the history log.
    """
    if not rule_code or "def rule" not in rule_code:
        return False, "Missing rule function."

    forbidden = [
        "import ",
        "__import__",
        "open(",
        "exec(",
        "eval(",
        "os.",
        "sys.",
        "subprocess",
        "pickle",
        "joblib",
        "line_id_encoded",
    ]
    for token in forbidden:
        if token in rule_code:
            return False, f"Forbidden token: {token}"

    referenced_features = set(RULE_FEATURE_REGEX.findall(rule_code))
    disallowed = sorted(f for f in referenced_features if f not in FEATURES_FOR_LLM)
    if disallowed:
        return False, f"Disallowed feature(s) used in rule: {disallowed}"

    try:
        compile(rule_code, "<rule_code>", "exec")
    except Exception as e:
        return False, f"Compilation error: {e}"

    return True, ""


def evaluate_rule(rule_code: str, X: pd.DataFrame) -> np.ndarray:
    """Executes a validated rule string against a feature matrix and returns binary predictions.

    The rule code is executed in an isolated namespace (no builtins, no globals) via
    ``exec``. The resulting ``rule`` function is called row-by-row on ``X``, with each
    row passed as a pandas Series (accessible as ``x["feature_name"]``).

    Any return value other than the integer 1 is coerced to 0 (safe binary output).

    Args:
        rule_code: A string containing a valid ``def rule(x): ...`` Python function.
        X:         Feature matrix (columns = FEATURES_FOR_LLM).

    Returns:
        Integer numpy array of predictions, shape (n_samples,), values in {0, 1}.

    Raises:
        ValueError: If the executed code does not define a callable named ``rule``.
    """
    exec(rule_code, {}, local_env)

    if "rule" not in local_env:
        raise ValueError("Executed code did not define a function named 'rule'.")

    rule_fn = local_env["rule"]
    preds: List[int] = []

    for _, row in X.iterrows():
        pred = rule_fn(row)
        pred = int(pred)
        pred = 1 if pred == 1 else 0
        preds.append(pred)

    return np.asarray(preds, dtype=int)


# CHANGE 6: worst cases sorted FN first then FP, computed from current y_pred
def get_worst_cases(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_items: int = MAX_WORST_CASES_FOR_PROMPT
) -> List[Dict[str, Any]]:
    """Collects misclassified test samples for inclusion in LLM prompts.

    Per the paper: sorted FN (missed failures) first, then FP (false alarms).
    Prioritising false negatives guides both LLMs to focus on the more costly error type.

    Each returned dict contains all feature values (as safe floats), ``y_true``,
    ``y_pred``, and ``error_type`` ('FN' or 'FP').

    Args:
        X_test:    Test feature matrix.
        y_true:    Ground-truth labels.
        y_pred:    Predicted labels for the current iteration (CHANGE 6: not best_rule).
        max_items: Maximum number of worst cases to return.

    Returns:
        List of misclassified sample dicts, sorted FN → FP, length <= max_items.
    """
    rows = []
    for i in range(len(X_test)):
        if int(y_true[i]) != int(y_pred[i]):
            row = {k: safe_float(X_test.iloc[i][k]) for k in X_test.columns}
            row["y_true"] = int(y_true[i])
            row["y_pred"] = int(y_pred[i])
            row["error_type"] = "FN" if int(y_true[i]) == 1 and int(y_pred[i]) == 0 else "FP"
            rows.append(row)

    # paper: FN first, then FP
    rows = sorted(rows, key=lambda r: 0 if r["error_type"] == "FN" else 1)
    return rows[:max_items]


def compact_history_for_prompt(history: List[Dict[str, Any]], max_items: int = MAX_HISTORY_ITEMS_FOR_PROMPT) -> List[Dict[str, Any]]:
    """Returns the last ``max_items`` iteration records in a compact form for prompt injection.

    Only includes fields relevant to the LLM (scores, metric values, change summary).
    The ``changes`` field is truncated to 300 characters to prevent prompt bloat.
    Ordered chronologically (oldest first) so the LLM can observe the trend.
    """
    if not history:
        return []
    keep = history[-max_items:]
    out = []
    for h in keep:
        out.append({
            "iteration": h.get("iteration"),
            "score": h.get("score"),
            "acc": h.get("acc"),
            "f2": h.get("f2"),
            "ovr": h.get("ovr"),
            "fa": h.get("fa"),
            "valid_rule": h.get("valid_rule"),
            "changes": clip_text(h.get("changes", ""), 300),
        })
    return out


def build_feature_description_text() -> str:
    """Formats ``FEATURE_DESCRIPTIONS`` as a bullet list for injection into LLM prompts.
    Provides the LLM with domain-specific context about each feature's physical meaning."""
    lines = []
    for f in FEATURES_FOR_LLM:
        lines.append(f"- {f}: {FEATURE_DESCRIPTIONS.get(f, '')}")
    return "\n".join(lines)


# ==========================================
# LLM PROMPT BUILDERS
# ==========================================

# CHANGE 4 + CHANGE 5: generator prompt aligned with paper
def generator_prompt(
    iteration: int,
    line_name: str,
    window: pd.DataFrame,
    window_summary: Dict[str, Any],
    feature_anchors: Dict[str, Any],
    previous_rule: str,
    previous_metrics: Optional[Dict[str, Any]],
    best_rule: str,
    best_metrics: Optional[Dict[str, Any]],
    feedback_text: str,
    history_table: List[Dict[str, Any]],
    worst_cases: List[Dict[str, Any]],
    oscillation_note: str = "",
    stagnation_note: str = "",
    seed_summary: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Builds the full Generator LLM prompt for a single iteration.

    The prompt contains (in order):
        - Oscillation / stagnation alerts (if active), injected at the top.
        - Seed rule summary (iterations 1–3 only) to give the LLM a warm start.
        - Task description, scoring formula, target corridor, and hard floors.
        - MANDATORY self-diagnosis: auto-detects Case A/B/C/D from ``previous_metrics``
          and injects the required corrective action so the LLM cannot skip it.
        - Feature statistics (min/mean/max) and class-separated anchor thresholds.
        - Critic feedback from the previous iteration.
        - Previous rule + metrics, best rule + metrics found so far.
        - Last ``MAX_HISTORY_ITEMS_FOR_PROMPT`` iterations of history.
        - Balanced training window as JSON (capped at 6000 chars).
        - Worst misclassified cases from the current best rule (FN first).
        - Feature descriptions and strict rule constraints.
        - Required output format: three tagged sections (Justification, Changes, Rule).

    Args:
        iteration:        Current iteration number (1-indexed).
        line_name:        Power-line identifier for this experiment.
        window:           Balanced training window DataFrame.
        window_summary:   Output of ``summarize_window``.
        feature_anchors:  Output of ``compute_feature_anchors``.
        previous_rule:    Rule code from the previous iteration (or empty string).
        previous_metrics: Metrics dict from the previous iteration (or None).
        best_rule:        Best rule code found so far (or empty string).
        best_metrics:     Metrics dict for the best rule (or None).
        feedback_text:    Critic feedback from the previous iteration.
        history_table:    Compact history list from ``compact_history_for_prompt``.
        worst_cases:      Misclassified samples from ``get_worst_cases``.
        oscillation_note: Non-empty string injected at top if oscillation is detected.
        stagnation_note:  Non-empty string injected at top if stagnation is detected.
        seed_summary:     Top-3 seed rules shown only in iterations 1–3.

    Returns:
        The complete Generator prompt string.
    """

    training_cols = FEATURES_FOR_LLM + ["target"]
    training_json = clip_text(
        window[training_cols].to_json(orient="records"),
        6000
    )

    prev_rule_section    = clip_text(previous_rule, MAX_RULE_CHARS) if previous_rule else "None"
    prev_metrics_section = dict_to_pretty_json(previous_metrics) if previous_metrics else "None"
    best_rule_section    = clip_text(best_rule, MAX_RULE_CHARS) if best_rule else "None"
    best_metrics_section = dict_to_pretty_json(best_metrics) if best_metrics else "None"

    # Feature stats as min/mean/max (paper primary format)
    feature_stats_text = ""
    if "feature_stats" in window_summary:
        feature_stats_text = "Feature statistics over the training window (min / mean / max):\n"
        for f, stats in window_summary["feature_stats"].items():
            feature_stats_text += (
                f"  {f}: min={stats['min']:.4f}  mean={stats['mean']:.4f}  max={stats['max']:.4f}\n"
            )

    top_anchors = dict(list(feature_anchors.items())[:6])
    anchors_text = "\nSuggested thresholds (from class-separated statistics — most discriminative first):\n"
    for f, v in top_anchors.items():
        direction = "higher in failures" if v["median_diff"] > 0 else "lower in failures"
        anchors_text += (
            f"  {f}: failures_median={v['pos_p50']}  normal_median={v['neg_p50']}"
            f"  → {direction}, suggested_threshold={v['suggested_threshold']}\n"
        )

    # Build the previous metrics summary for self-diagnosis
    prev_ovr = previous_metrics.get("ovr", None) if previous_metrics else None
    prev_fa  = previous_metrics.get("fa",  None) if previous_metrics else None
    if prev_ovr is not None and prev_fa is not None:
        if prev_ovr > 0.50:
            auto_case = "A (OVR={:.2f} > 0.50 — rule misses most failures)".format(prev_ovr)
            auto_action = "Lower thresholds toward pos_p25 values of the top features."
        elif prev_fa > 0.50:
            auto_case = "B (FA={:.2f} > 0.50 — too many false alarms)".format(prev_fa)
            auto_action = "Raise thresholds toward neg_p75 values of the top features."
        elif prev_ovr > 0.10 or prev_fa > 0.20:
            auto_case = "C (OVR={:.2f}, FA={:.2f} — moderate errors)".format(prev_ovr, prev_fa)
            auto_action = "Fine-tune: adjust only the condition causing most FN or FP from worst cases."
        else:
            auto_case = "D (OVR={:.2f}, FA={:.2f} — near target)".format(prev_ovr, prev_fa)
            auto_action = "Micro-adjust only. Do not make structural changes."
        diagnosis_text = f"Detected Case {auto_case}\nRequired action: {auto_action}"
    else:
        diagnosis_text = "First iteration — use strategy D: start with suggested_threshold on top-2 features."

    alerts = ""
    if oscillation_note:
        alerts += f"\n{oscillation_note}\n"
    if stagnation_note:
        alerts += f"\n{stagnation_note}\n"

    seed_section = ""
    if seed_summary:
        seed_section = "\n## Data-driven seed rules (computed before iteration 1 — use as starting point)\n"
        seed_section += "These rules were generated by the system from the training data statistics.\n"
        seed_section += "They are your BASELINE. Your goal is to improve on the best seed score.\n"
        for s in seed_summary:
            seed_section += (
                f"  Rank {s['rank']}: score={s['score']}  f2={s['f2']}  "
                f"ovr={s['ovr']}  fa={s['fa']}\n"
                f"  {s['rule']}\n\n"
            )

    return f"""
You are the generator LLM in a two-LLM iterative framework for power-grid failure prediction.

Your task: Generate ONE improved Python if/else rule to predict binary failure risk,
imitating a teacher HGB model.
{alerts}{seed_section}
## Context
- Disconnected line: {line_name}
- Failures are rare (imbalanced dataset)
- Teacher HGB: Acc={HGB_TARGET['acc']:.4f} | F2={HGB_TARGET['f2']:.4f} | OVR={HGB_TARGET['ovr']:.4f} | FA={HGB_TARGET['fa']:.4f}

## Scoring formula
  score = 0.10*Acc + 0.40*F2 + 0.35*(1-OVR) + 0.15*(1-FA)
  Penalties:
    F2 == 0              → score = -1.00  (rule never catches failures — forbidden)
    FA >= 0.90           → score = -0.90  (rule fires almost always — also forbidden)
    OVR > 0.50 AND F2>0  → additional -0.30 progressive penalty on top of base
  Priority: F2 (0.40) > OVR (0.35) > FA (0.15) > Acc (0.10)
  TARGET CORRIDOR: OVR ∈ [0.04, 0.15] AND FA ∈ [0.05, 0.25]

## MANDATORY self-diagnosis (do this BEFORE writing the rule)
{diagnosis_text}

## {feature_stats_text}
{anchors_text}

## ⚠️ Critic feedback (from previous iteration)
{clip_text(feedback_text, 1500) if feedback_text else "None"}

## Previous rule
{prev_rule_section}

## Previous rule metrics
{prev_metrics_section}

## Best rule found so far
{best_rule_section}

## Best rule metrics so far
{best_metrics_section}

## Iteration history (last {MAX_HISTORY_ITEMS_FOR_PROMPT})
{dict_to_pretty_json(history_table) if history_table else "None"}

## Training window (JSON, balanced sample)
{training_json}

## Worst misclassified cases (FN first, then FP)
{dict_to_pretty_json(worst_cases) if worst_cases else "None"}

## Feature descriptions
{build_feature_description_text()}

## Rule constraints (MANDATORY — any violation = INVALID rule, score = null)
- Use ONLY nested if/else (NO elif)
- Use ONLY features from the Feature descriptions list
- No imports, no loops, no helper functions
- Return ONLY 0 or 1
- 2–5 conditions total; start from suggested_threshold values

## Output format (ALL THREE SECTIONS REQUIRED)

[Start of Justification]
State the Case (A/B/C/D) you detected, the action you took, and which threshold you changed.
[End of Justification]

[Start of Changes]
One sentence: what specific threshold or feature changed, and in which direction.
[End of Changes]

[Start of Rule]
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        return 0
```
[End of Rule]
""".strip()


def repair_rule_prompt(bad_output: str, error_msg: str) -> str:
    """Builds a one-shot repair prompt sent to the Generator when the first response
    fails ``validate_rule_code``.

    Includes the validation error message and the original bad output so the LLM can
    identify and fix the specific problem (e.g. missing tags, use of ``elif``, import).
    The response is expected to contain only the corrected rule inside the standard tags.
    """
    return f"""
The following output is not valid yet.

Error:
{error_msg}

You must fix it and output ONLY valid Python code wrapped in tags like this:

[Start of Rule]
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        return 0
```
[End of Rule]

Requirements:
- define exactly: def rule(x):
- return only 0 or 1
- no imports
- no helper functions
- no elif

Bad output:
{bad_output}
""".strip()


# CHANGE 5: critic produces up to 10 suggestions, detects degenerate rules, checks compliance
def critic_prompt(
    iteration: int,
    line_name: str,
    line_id_encoded: int,
    feature_anchors: Dict[str, Any],
    current_rule: str,
    current_metrics: Dict[str, Any],
    previous_rule: str,
    previous_metrics: Optional[Dict[str, Any]],
    best_rule: str,
    best_metrics: Optional[Dict[str, Any]],
    history_table: List[Dict[str, Any]],
    worst_cases: List[Dict[str, Any]],
    window_summary: Dict[str, Any],
) -> str:
    """Builds the Critic LLM prompt that evaluates the current rule and produces feedback.

    The Critic prompt includes:
        - Iteration context, HGB targets, scoring formula, and hard-floor penalties.
        - Oscillation detection from ``history_table``: if the last 4 valid
          iterations show both high-OVR (>0.50) and high-FA (>0.50), a warning overrides
          the normal multi-suggestion output with a single small-step nudge.
        - Degenerate rule detection: if F2==0, the Critic must instruct the Generator
          to drastically lower thresholds toward ``pos_p25`` values.
        - Diagnostic case instructions (A/B/C/D) with concrete feature + threshold guidance.
        - Top-5 feature anchors as threshold reference.
        - Current, previous, and best rules with their metrics.
        - Last ``MAX_HISTORY_ITEMS_FOR_PROMPT`` iterations of history.
        - Worst misclassified cases (FN first) from the current iteration's predictions.
        - Window data summary for context.
        - Output instructions: state Case, provide suggestions (or oscillation nudge),
          and a syntactic compliance checklist.

    Args:
        iteration:        Current iteration number.
        line_name:        Power-line identifier.
        line_id_encoded:  Integer ID of the line (unused in prompt text, passed for logging).
        feature_anchors:  Class-separated threshold statistics from ``compute_feature_anchors``.
        current_rule:     The rule generated in this iteration.
        current_metrics:  Metrics of the current rule on the test set.
        previous_rule:    Rule from the previous iteration.
        previous_metrics: Metrics from the previous iteration.
        best_rule:        Best rule found so far.
        best_metrics:     Metrics of the best rule.
        history_table:    Compact history list.
        worst_cases:      Misclassified samples from the current iteration's predictions.
        window_summary:   Statistical summary of the current training window.

    Returns:
        The complete Critic prompt string.
    """
    top5 = dict(list(feature_anchors.items())[:5])
    anchors_summary = ""
    for f, v in top5.items():
        direction = "↑ higher in failures" if v["median_diff"] > 0 else "↓ lower in failures"
        anchors_summary += (
            f"  {f}: failures_median={v['pos_p50']}, normal_median={v['neg_p50']}, "
            f"suggested_threshold={v['suggested_threshold']}  ({direction})\n"
        )

    cur_f2  = current_metrics.get("f2",  0.0)
    cur_ovr = current_metrics.get("ovr", 1.0)
    cur_fa  = current_metrics.get("fa",  1.0)
    cur_acc = current_metrics.get("acc", 0.0)

    # Detect oscillation from history
    recent_valid = [h for h in history_table if h.get("valid_rule") and h.get("ovr") is not None]
    ovr_vals = [h["ovr"] for h in recent_valid[-4:]]
    fa_vals  = [h["fa"]  for h in recent_valid[-4:]]
    oscillating = (
        len(ovr_vals) >= 3
        and sum(1 for v in ovr_vals if v > 0.50) >= 1
        and sum(1 for v in fa_vals  if v > 0.50) >= 1
    )
    oscillation_warning = ""
    if oscillating:
        oscillation_warning = f"""
## ⚠️ OSCILLATION ALERT
The rule has been oscillating between high-OVR ({[round(v,2) for v in ovr_vals]}) and
high-FA ({[round(v,2) for v in fa_vals]}) over recent iterations.
YOUR MOST IMPORTANT INSTRUCTION: tell the generator to make only ONE small change —
adjust a single threshold by a SMALL amount (5–15% of its current value).
Do NOT suggest adding/removing conditions, inverting conditions, or changing features.
Name the exact threshold, the exact current value, and the exact new value to try.
""".strip()

    return f"""
You are the critic LLM in a two-LLM iterative framework for power-grid failure prediction.

## Context
Iteration: {iteration} | Line: {line_name}
Teacher HGB targets: Acc={HGB_TARGET['acc']:.4f} | F2={HGB_TARGET['f2']:.4f} | OVR={HGB_TARGET['ovr']:.4f} | FA={HGB_TARGET['fa']:.4f}
Scoring formula: 0.10*Acc + 0.40*F2 + 0.35*(1-OVR) + 0.15*(1-FA)
Penalties: F2==0 → -1.0 | FA>=0.90 → -0.90 | OVR>0.50 → strong progressive penalty

## Current rule metrics
  Acc={cur_acc:.4f}  F2={cur_f2:.4f}  OVR={cur_ovr:.4f}  FA={cur_fa:.4f}
{oscillation_warning}
## Degenerate rule detection (CHECK FIRST — before anything else)
- If F2 == 0.0: the rule NEVER predicts failure → score = -1.0.
  MANDATORY: tell the generator to drastically lower thresholds toward pos_p25 values.
  Name the exact feature and exact threshold value (use pos_p25 from the anchors below).

## Diagnostic case (apply after degenerate check)
  CASE A — OVR > 0.50: misses most failures
    → Lower thresholds toward pos_p25. Name feature + value.
  CASE B — FA > 0.50: too many false alarms
    → Raise thresholds toward neg_p75. Name feature + value.
  CASE C — OVR in (0.10, 0.50) and FA in (0.10, 0.50): moderate both sides
    → Identify the ONE condition causing most FN vs FP. Adjust only that threshold.
  CASE D — OVR < 0.10 and FA < 0.20: near target
    → Micro-adjust only. No structural changes.

## Top discriminative features with suggested thresholds
{anchors_summary}

## Current rule
{clip_text(current_rule, MAX_RULE_CHARS)}

## Current rule metrics
{dict_to_pretty_json(current_metrics)}

## Previous rule
{clip_text(previous_rule, MAX_RULE_CHARS) if previous_rule else "None"}

## Previous rule metrics
{dict_to_pretty_json(previous_metrics) if previous_metrics is not None else "None"}

## Best rule so far
{clip_text(best_rule, MAX_RULE_CHARS) if best_rule else "None"}

## Best rule metrics so far
{dict_to_pretty_json(best_metrics) if best_metrics is not None else "None"}

## Iteration history (last {MAX_HISTORY_ITEMS_FOR_PROMPT})
{dict_to_pretty_json(history_table)}

## Worst misclassified cases (FN first, then FP)
{dict_to_pretty_json(worst_cases)}

## Data summary
{dict_to_pretty_json(window_summary)}

## Instructions for your response
1. State which CASE applies (degenerate / A / B / C / D).
2. If oscillating (see alert above): give ONLY ONE suggestion — a small single-threshold nudge.
   Otherwise: provide up to 10 concrete threshold-guided suggestions.
   Every suggestion must state: feature name, current threshold value, proposed new value.
3. If score unchanged for 3+ consecutive iterations: demand a structurally different rule
   (change which features are used, not just threshold values).
4. NEVER suggest raising thresholds when CASE A or degenerate applies.
5. NEVER suggest lowering thresholds when CASE B applies.

## Syntactic compliance checklist
Verify the current rule:
- [ ] No elif (only nested if/else)
- [ ] No imports, no loops, no helper functions
- [ ] Returns only 0 or 1
- [ ] Uses only allowed features (not line_id_encoded)

State: PASS or FAIL (list which checks failed).
""".strip()



# ==========================================
# LLM API CLIENT
# ==========================================

def call_llm(prompt: str, temperature: float) -> str:
    """Sends a prompt to an OpenAI-compatible chat completions API and returns the response.

    Configuration is loaded from ``LLM_CONFIG_FILE_PATH`` on every call (supports hot-reload).
    Uses Bearer token authentication and a standard ``/chat/completions`` payload.

    Retry behaviour:
        On any HTTP error or exception, retries up to ``API_RETRIES`` times with
        ``SLEEP_BETWEEN_RETRIES`` seconds between attempts. Raises ``RuntimeError``
        with full context after all retries are exhausted.

    Args:
        prompt:      Full prompt string (combined system + user content).
        temperature: LLM sampling temperature in [0, 1].

    Returns:
        The model's text response as a plain string.

    Raises:
        RuntimeError: If HTTP status >= 400, response structure is invalid,
                      or all retry attempts fail.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }

    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": config.get("max_tokens", 4000),
    }

    last_error = None

    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.post(
                config["api_url"],
                headers=headers,
                json=payload,
                verify=config.get("verify_ssl", True),
                timeout=config.get("timeout", API_TIMEOUT),
            )
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")

            data = r.json()

            if "choices" not in data or not data["choices"]:
                raise RuntimeError(f"Invalid API response structure: {str(data)[:1000]}")

            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                content = str(content)
            return content

        except Exception as e:
            last_error = e
            if attempt < API_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                raise RuntimeError(f"LLM call failed after {API_RETRIES} attempts: {e}") from e

    raise RuntimeError(f"Unexpected LLM failure: {last_error}")



# ==========================================
# EXPERIMENT ORCHESTRATION
# ==========================================

def run_line_experiment(
    df_line: pd.DataFrame,
    line_name: str,
    line_id_encoded: int,
    temperature: float,
    OUTPUT_DIR_LLM: str,
    split_mode: str = "standard",
) -> Dict[str, Any]:
    """Orchestrates the full distillation experiment for a single power line.

    Selects the appropriate split strategy and delegates to ``_run_experiment_on_split``.
    For LOO mode, runs one full distillation loop per fold and returns the result of the
    best-scoring fold (by ``best_score``).

    Args:
        df_line:         Dataset slice for this line (already filtered by ``line_disconnected``).
        line_name:       Line identifier string (e.g. ``'34_35_110'``).
        line_id_encoded: Integer encoding from ``LINE_MAP``.
        temperature:     LLM sampling temperature for this experiment.
        OUTPUT_DIR_LLM:      Base output directory for this temperature sweep.
        split_mode:      One of ``'standard'``, ``'adaptive'``, or ``'loo'``.
                         Determined by ``n_pos_total`` in ``main``.

    Returns:
        Result dictionary from ``_run_experiment_on_split`` (or the best LOO fold),
        enriched with ``split_mode`` and, for LOO, ``n_loo_folds``.
    """

    line_dir = os.path.join(OUTPUT_DIR_LLM, f"line_{line_name}")
    ensure_dir(line_dir)

    if split_mode == "loo":
        loo_folds = leave_one_out_splits(df_line)
        print(f"  [LOO] {line_name}: {len(loo_folds)} folds")
        fold_results = []
        for fold_i, (train_df_fold, test_df_fold) in enumerate(loo_folds):
            fold_result = _run_experiment_on_split(
                df_line=df_line,
                train_df=train_df_fold,
                test_df=test_df_fold,
                line_name=line_name,
                line_id_encoded=line_id_encoded,
                temperature=temperature,
                line_dir=os.path.join(line_dir, f"fold_{fold_i}"),
                split_mode="loo",
                fold_label=f"fold{fold_i}",
            )
            fold_results.append(fold_result)

        best_fold = max(fold_results, key=lambda r: r["best_score"] or float("-inf"))
        best_fold["line_name"] = line_name
        best_fold["split_mode"] = "loo"
        best_fold["n_loo_folds"] = len(loo_folds)
        return best_fold

    elif split_mode == "adaptive":
        train_df, test_df = adaptive_split(df_line)
    else:
        train_df, test_df = sequential_guarded_split(df_line)

    return _run_experiment_on_split(
        df_line=df_line,
        train_df=train_df,
        test_df=test_df,
        line_name=line_name,
        line_id_encoded=line_id_encoded,
        temperature=temperature,
        line_dir=line_dir,
        split_mode=split_mode,
        fold_label="",
    )


def _run_experiment_on_split(
    df_line: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    line_name: str,
    line_id_encoded: int,
    temperature: float,
    line_dir: str,
    split_mode: str,
    fold_label: str,
) -> Dict[str, Any]:
    """Core Generator–Critic distillation loop for a single train/test split.

    Execution flow:
        0. SEED PHASE: ``compute_seed_rules`` generates univariate + bivariate threshold
           rules from feature anchors, evaluates them on the test set, and bootstraps
           ``best_rule / best_score / best_metrics`` so iteration 1 starts warm.

        Per iteration (1 … ITERATIONS):
            1. Build a balanced context window via ``build_balanced_window``.
            2. Compute oscillation / stagnation flags from ``_recent_ovr``, ``_recent_fa``,
               and ``_stagnant_iters``
            3. Compute worst-cases context from the current best_rule predictions (CHANGE 6).
            4. Build and call the Generator prompt → extract rule code.
            5. Validate the rule; if invalid, attempt one repair call.
               If still invalid, log and continue to next iteration.
            6. Execute the rule on the test set via ``evaluate_rule``.
            7. Compute metrics via ``compute_metrics`` and score via ``scoring_function``.
            8. Compute worst-cases from this iteration's predictions for the Critic.
            9. Build and call the Critic prompt → extract feedback string.
           10. Log the iteration to ``history``.
           11. Update ``best_*`` if this iteration's score exceeds the current best.
           12. Update oscillation/stagnation trackers.

        Post-loop:
            - Saves ``iteration_history.csv``, ``seed_candidates.csv``,
              ``best_rule.py``, ``best_justification.txt``, ``best_feedback.txt``,
              ``best_changes.txt`` to ``line_dir``.

    Error counters tracked:
        ``invalid_rule_count`` – Rules that failed validation even after repair.
        ``api_error_count``     – LLM API call failures (HTTP errors, timeouts).
        ``eval_error_count``    – Exceptions during rule execution or metric computation.

    Args:
        df_line:         Full line dataset (used for support statistics in the result).
        train_df:        Training partition for window sampling and anchor computation.
        test_df:         Held-out test partition for rule evaluation.
        line_name:       Line identifier string.
        line_id_encoded: Integer encoding.
        temperature:     LLM sampling temperature.
        line_dir:        Directory where all outputs for this split are saved.
        split_mode:      Split strategy label (saved in result for diagnostics).
        fold_label:      LOO fold label (empty string for non-LOO splits).

    Returns:
        A result dictionary containing best_score, best_metrics (acc/f2/fa/ovr/tp/fp/fn/tn),
        support counts (n_total/n_train/n_test/n_pos_*), error counts, seed stats,
        and file paths (history_path, line_dir).
    """

    ensure_dir(line_dir)

    X_test = test_df[FEATURES_FOR_LLM].copy()
    y_test = test_df["target"].astype(int).values

    feature_anchors = compute_feature_anchors(train_df, FEATURES_FOR_LLM)

    # ------------------------------------------------------------------ #
    # SEED PHASE: compute best data-driven rules before any LLM call.    #
    # This gives the LLM a strong starting point and avoids cold-start.  #
    # ------------------------------------------------------------------ #
    seed_candidates = compute_seed_rules(
        train_df=train_df,
        X_test=X_test,
        y_test=y_test,
        feature_anchors=feature_anchors,
        top_n_features=6,
    )

    best_rule = ""
    best_justification = ""
    best_feedback = ""
    best_changes = ""
    best_metrics: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_iteration = None

    # Bootstrap best_* from the top seed (if any valid seed exists)
    if seed_candidates:
        top_seed = seed_candidates[0]
        if top_seed["score"] > best_score:
            best_rule      = top_seed["rule_code"]
            best_score     = top_seed["score"]
            best_metrics   = {k: top_seed[k] for k in ("acc","f2","fa","ovr","tp","fp","fn","tn")}
            best_iteration = 0   # iteration 0 = seed phase
            best_justification = "Seed rule generated by data-driven univariate/bivariate search."
        print(
            f"  [SEED] best seed score={top_seed['score']:.4f} "
            f"f2={top_seed['f2']:.4f} ovr={top_seed['ovr']:.4f} fa={top_seed['fa']:.4f}"
        )

    # Compact seed summary for use in prompts (top-3 seeds)
    seed_summary_for_prompt = []
    for i, s in enumerate(seed_candidates[:3]):
        seed_summary_for_prompt.append({
            "rank": i + 1,
            "score": round(s["score"], 4),
            "f2": round(s["f2"], 4),
            "ovr": round(s["ovr"], 4),
            "fa":  round(s["fa"], 4),
            "rule": s["rule_code"],
        })

    previous_rule = ""
    previous_metrics: Optional[Dict[str, Any]] = None
    previous_feedback = ""

    history: List[Dict[str, Any]] = []

    invalid_rule_count = 0
    api_error_count = 0
    eval_error_count = 0

    # Oscillation tracking
    _recent_ovr: List[float] = []   # last 4 valid OVR values
    _recent_fa:  List[float] = []   # last 4 valid FA values
    _stagnant_iters = 0             # consecutive iters without score improvement
    _last_best_score = float("-inf")

    for iteration in range(1, ITERATIONS + 1):
        window = build_balanced_window(train_df)
        window_summary = summarize_window(window, FEATURES_FOR_LLM)
        history_table = compact_history_for_prompt(history)

        # Detect oscillation: alternating high-OVR / high-FA over last 4 iters
        _is_oscillating = False
        _oscillation_note = ""
        if len(_recent_ovr) >= 3:
            _hi_ovr = sum(1 for v in _recent_ovr[-4:] if v > 0.50)
            _hi_fa  = sum(1 for v in _recent_fa[-4:]  if v > 0.50)
            if _hi_ovr >= 1 and _hi_fa >= 1:
                _is_oscillating = True
                _oscillation_note = (
                    f"⚠️ OSCILLATION DETECTED over last {len(_recent_ovr[-4:])} iterations: "
                    f"the rule is swinging between high-OVR (misses failures) and high-FA "
                    f"(too many alarms). recent_OVR={[round(v,2) for v in _recent_ovr[-4:]]} "
                    f"recent_FA={[round(v,2) for v in _recent_fa[-4:]]}. "
                    f"You MUST make a SMALL, INCREMENTAL threshold adjustment — "
                    f"change at most ONE threshold by at most 10% of its current value. "
                    f"Do NOT flip the sign of any condition. "
                    f"Do NOT add or remove conditions."
                )

        _stagnation_note = ""
        if _stagnant_iters >= 5:
            _stagnation_note = (
                f"⚠️ STAGNATION: score has not improved for {_stagnant_iters} consecutive "
                f"iterations. Try changing the feature set (use a different top feature) "
                f"or adding a second condition using the next most discriminative feature."
            )

        # CHANGE 6: worst cases from current best_rule predictions on test set
        # (if no best_rule yet, use zeros — all positives become FN, shown first)
        if best_rule:
            try:
                best_y_pred_for_wc = evaluate_rule(best_rule, X_test)
            except Exception:
                best_y_pred_for_wc = np.zeros_like(y_test)
        else:
            best_y_pred_for_wc = np.zeros_like(y_test)

        worst_cases_context = get_worst_cases(
            X_test, y_test,
            best_y_pred_for_wc,
            max_items=min(10, MAX_WORST_CASES_FOR_PROMPT)
        )

        gen_prompt = generator_prompt(
            iteration=iteration,
            line_name=line_name,
            window=window,
            window_summary=window_summary,
            feature_anchors=feature_anchors,
            previous_rule=previous_rule,
            previous_metrics=previous_metrics,
            best_rule=best_rule,
            best_metrics=best_metrics,
            feedback_text=previous_feedback,
            history_table=history_table,
            worst_cases=worst_cases_context,
            oscillation_note=_oscillation_note,
            stagnation_note=_stagnation_note,
            seed_summary=seed_summary_for_prompt if iteration <= 3 else [],
        )

        raw_generator_response = ""
        justification = ""
        changes = ""
        rule_code = ""
        critic_feedback = ""

        try:
            raw_generator_response = call_llm(gen_prompt, temperature)
            justification = clip_text(extract_tagged_section(raw_generator_response, "justification"), MAX_JUSTIFICATION_CHARS)
            changes = clip_text(extract_tagged_section(raw_generator_response, "changes"), MAX_CHANGES_CHARS)
            rule_code = clip_text(extract_python_rule(raw_generator_response), MAX_RULE_CHARS)

            valid_rule, valid_rule_error = validate_rule_code(rule_code)

            if not valid_rule:
                repair_prompt = repair_rule_prompt(raw_generator_response, valid_rule_error)
                repaired_response = call_llm(repair_prompt, temperature)
                repaired_rule = extract_python_rule(repaired_response)
                valid_rule, valid_rule_error = validate_rule_code(repaired_rule)

                if valid_rule:
                    rule_code = repaired_rule
                else:
                    invalid_rule_count += 1
                    history.append({
                        "iteration": iteration,
                        "line_name": line_name,
                        "line_id_encoded": line_id_encoded,
                        "valid_rule": False,
                        "rule_error": valid_rule_error,
                        "score": None,
                        "acc": None,
                        "f2": None,
                        "ovr": None,
                        "fa": None,
                        "tp": None,
                        "fp": None,
                        "fn": None,
                        "tn": None,
                        "changes": changes,
                        "feedback": "",
                        "rule_code": rule_code,
                        "justification": justification,
                    })
                    previous_feedback = f"Previous candidate was invalid. Fix this problem first: {valid_rule_error}"
                    previous_rule = rule_code or previous_rule
                    previous_metrics = None
                    continue

            y_pred = evaluate_rule(rule_code, X_test)
            metrics = compute_metrics(y_test, y_pred)
            score = scoring_function(metrics)

            # CHANGE 6: worst cases for critic computed from THIS iteration's y_pred
            current_worst_cases = get_worst_cases(X_test, y_test, y_pred)

            critic_input = critic_prompt(
                iteration=iteration,
                line_name=line_name,
                line_id_encoded=line_id_encoded,
                feature_anchors=feature_anchors,
                current_rule=rule_code,
                current_metrics=metrics,
                previous_rule=previous_rule,
                previous_metrics=previous_metrics,
                best_rule=best_rule,
                best_metrics=best_metrics,
                history_table=history_table,
                worst_cases=current_worst_cases,
                window_summary=window_summary,
            )

            critic_feedback = clip_text(call_llm(critic_input, temperature), MAX_FEEDBACK_CHARS)

            row = {
                "iteration": iteration,
                "line_name": line_name,
                "line_id_encoded": line_id_encoded,
                "valid_rule": True,
                "rule_error": "",
                "score": score,
                **metrics,
                "changes": changes,
                "feedback": critic_feedback,
                "rule_code": rule_code,
                "justification": justification,
            }
            history.append(row)

            if score > best_score:
                best_score = score
                best_rule = rule_code
                best_justification = justification
                best_feedback = critic_feedback
                best_changes = changes
                best_metrics = metrics
                best_iteration = iteration

            # Update oscillation / stagnation trackers
            _recent_ovr.append(metrics["ovr"])
            _recent_fa.append(metrics["fa"])
            if len(_recent_ovr) > 6:
                _recent_ovr.pop(0)
                _recent_fa.pop(0)
            if score > _last_best_score:
                _stagnant_iters = 0
                _last_best_score = score
            else:
                _stagnant_iters += 1

            previous_rule = rule_code
            previous_metrics = metrics
            previous_feedback = critic_feedback

            if iteration % 20 == 0 or iteration == 1:
                print(
                    f"[{line_name}] iter={iteration} "
                    f"score={score:.4f} acc={metrics['acc']:.4f} f2={metrics['f2']:.4f} "
                    f"ovr={metrics['ovr']:.4f} fa={metrics['fa']:.4f}"
                )

        except Exception as e:
            api_error_count += 1 if "LLM call failed" in str(e) or "HTTP" in str(e) else 0
            eval_error_count += 1 if "rule" in str(e).lower() or "compile" in str(e).lower() else 0

            history.append({
                "iteration": iteration,
                "line_name": line_name,
                "line_id_encoded": line_id_encoded,
                "valid_rule": False,
                "rule_error": str(e),
                "score": None,
                "acc": None,
                "f2": None,
                "ovr": None,
                "fa": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None,
                "changes": changes,
                "feedback": critic_feedback,
                "rule_code": rule_code,
                "justification": justification,
            })
            previous_feedback = (
                "The previous iteration failed. "
                f"Repair the issue and keep the rule compact. Error: {str(e)[:1000]}"
            )

    history_df = pd.DataFrame(history)
    history_path = os.path.join(line_dir, "iteration_history.csv")
    history_df.to_csv(history_path, index=False)

    # Save seed candidates for diagnostics
    if seed_candidates:
        seed_rows = []
        for i, s in enumerate(seed_candidates):
            seed_rows.append({
                "rank": i + 1,
                "score": s["score"],
                "f2": s["f2"],
                "ovr": s["ovr"],
                "fa": s["fa"],
                "acc": s["acc"],
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "tn": s["tn"],
                "rule_code": s["rule_code"],
            })
        pd.DataFrame(seed_rows).to_csv(
            os.path.join(line_dir, "seed_candidates.csv"), index=False
        )

    with open(os.path.join(line_dir, "best_rule.py"), "w", encoding="utf-8") as f:
        f.write(best_rule if best_rule else "# No valid rule was found.\n")

    with open(os.path.join(line_dir, "best_justification.txt"), "w", encoding="utf-8") as f:
        f.write(best_justification or "No valid justification available.\n")

    with open(os.path.join(line_dir, "best_feedback.txt"), "w", encoding="utf-8") as f:
        f.write(best_feedback or "No valid critic feedback available.\n")

    with open(os.path.join(line_dir, "best_changes.txt"), "w", encoding="utf-8") as f:
        f.write(best_changes or "No valid changes log available.\n")

    support_info = {
        "n_total": int(len(df_line)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_pos_total": int(df_line["target"].sum()),
        "n_pos_train": int(train_df["target"].sum()),
        "n_pos_test": int(test_df["target"].sum()),
        "positive_rate_total": float(df_line["target"].mean()),
        "positive_rate_train": float(train_df["target"].mean()),
        "positive_rate_test": float(test_df["target"].mean()),
        "split_mode": split_mode,
        "fold_label": fold_label,
    }

    if best_metrics is None:
        best_metrics = {
            "acc": None, "f2": None, "fa": None, "ovr": None,
            "tp": None, "fp": None, "fn": None, "tn": None
        }

    result = {
        "line_name": line_name,
        "line_id_encoded": line_id_encoded,
        "best_iteration": best_iteration,
        "best_score": None if best_score == float("-inf") else float(best_score),
        **best_metrics,
        **support_info,
        "seed_best_score": round(seed_candidates[0]["score"], 4) if seed_candidates else None,
        "seed_best_f2":    round(seed_candidates[0]["f2"],    4) if seed_candidates else None,
        "seed_best_ovr":   round(seed_candidates[0]["ovr"],   4) if seed_candidates else None,
        "seed_best_fa":    round(seed_candidates[0]["fa"],    4) if seed_candidates else None,
        "invalid_rule_count": int(invalid_rule_count),
        "api_error_count": int(api_error_count),
        "eval_error_count": int(eval_error_count),
        "history_path": history_path,
        "line_dir": line_dir,
        "had_valid_rule": bool(best_rule),
    }

    return result



# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main() -> None:
    """Top-level pipeline entry point.

    Execution flow:
        1. Load the dataset from ``CFG.CSV_OUTPUT_PATH``.
        2. Compute ``load_gen_ratio``
        3. Validate that all required columns are present.
        4. Encode line IDs via ``encode_line_ids``.
        5. Generate distillation targets via ``build_teacher_target`` (uses HGB model if available).
        6. For each temperature in ``TEMPERATURES``:
            a. Create temperature-specific output sub-directory.
            b. For each line in ``LINE_MAP``:
                - Skip if no rows or fewer than 2 positives exist.
                - Select split mode (standard / adaptive / loo) based on ``n_pos_total``.
                - Run ``run_line_experiment``; catch and log any exceptions.
            c. Save ``summary_per_line.csv`` and ``skipped_lines.csv`` for this temperature.
        7. Save ``summary_all_temperatures.csv`` across all temperatures and lines.

    Output files:
        ``<OUTPUT_DIR_LLM>/summary_all_temperatures.csv`` – Global sweep summary.
        ``<OUTPUT_DIR_LLM>/temp_<T>/summary_per_line.csv`` – Per-temperature results.
        ``<OUTPUT_DIR_LLM>/temp_<T>/skipped_lines.csv``    – Lines skipped with reason.
        ``<OUTPUT_DIR_LLM>/temp_<T>/line_<n>/``            – Per-line outputs (see _run_experiment_on_split).
    """


    df["load_gen_ratio"] = df.apply(
        lambda row: (row["sum_load_p"] / row["sum_gen_p"]) if row.get("sum_gen_p", 0) != 0 else 0.0, axis=1
    )
    validate_required_columns(df)

    df = encode_line_ids(df)
    df = build_teacher_target(df)

    global_summary = []

    for temperature in TEMPERATURES:

        print(f"\n==============================")
        print(f"Running experiments with temperature {temperature}")
        print(f"==============================")

        temp_OUTPUT_DIR_LLM = os.path.join(OUTPUT_DIR_LLM, f"temp_{temperature}")
        ensure_dir(temp_OUTPUT_DIR_LLM)

        summary_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []

        for line_name, line_id_encoded in LINE_MAP.items():

            df_line = df[df["line_disconnected"] == line_name].copy()

            if len(df_line) == 0:
                skipped_rows.append({
                    "temperature": temperature,
                    "line_name": line_name,
                    "line_id_encoded": line_id_encoded,
                    "reason": "No rows in dataset for this line.",
                })
                print(f"[SKIP] {line_name}: no rows found.")
                continue

            n_pos_total = int(df_line["target"].sum())

            if n_pos_total < 2:
                skipped_rows.append({
                    "temperature": temperature,
                    "line_name": line_name,
                    "line_id_encoded": line_id_encoded,
                    "reason": f"Only {n_pos_total} positive(s) — cannot build any train/test split.",
                })
                print(f"[SKIP] {line_name}: only {n_pos_total} positive(s), truly unusable.")
                continue

            if n_pos_total < MIN_POS_LOO_THRESHOLD:
                split_mode = "loo"
            elif n_pos_total < MIN_POS_STANDARD:
                split_mode = "adaptive"
            else:
                split_mode = "standard"

            print(f"[{line_name}] n_pos={n_pos_total}, split_mode={split_mode}")

            try:

                result = run_line_experiment(
                    df_line=df_line,
                    line_name=line_name,
                    line_id_encoded=line_id_encoded,
                    temperature=temperature,
                    OUTPUT_DIR_LLM=temp_OUTPUT_DIR_LLM,
                    split_mode=split_mode,
                )

                result["temperature"] = temperature

                summary_rows.append(result)
                global_summary.append(result)

            except Exception as e:

                skipped_rows.append({
                    "temperature": temperature,
                    "line_name": line_name,
                    "line_id_encoded": line_id_encoded,
                    "reason": str(e),
                })

                print(f"[ERROR] {line_name}: {e}")
                traceback.print_exc()

        summary_df = pd.DataFrame(summary_rows)
        skipped_df = pd.DataFrame(skipped_rows)

        summary_df.to_csv(
            os.path.join(temp_OUTPUT_DIR_LLM, "summary_per_line.csv"),
            index=False
        )

        skipped_df.to_csv(
            os.path.join(temp_OUTPUT_DIR_LLM, "skipped_lines.csv"),
            index=False
        )

    global_df = pd.DataFrame(global_summary)

    global_df.to_csv(
        os.path.join(OUTPUT_DIR_LLM, "summary_all_temperatures.csv"),
        index=False
    )

    print("\nFinished.")
    print(f"Global summary saved to: {os.path.join(OUTPUT_DIR_LLM, 'summary_all_temperatures.csv')}")


if __name__ == "__main__":
    main()