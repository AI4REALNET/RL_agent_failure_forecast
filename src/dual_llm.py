ort os
import re
import json
import math
import time
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Paths and I/O
# ---------------------------------------------------------------------------

DATA_PATH: str = "uncertainty_disconnection_analysis.csv"
HGB_MODEL_PATH: str = "classifier_full_uncertainty.pkl"
LLM_CONFIG_FILE: str = "llm_config_inesctec.json"
OUTPUT_DIR: str = "llm_rule_results"


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

#: Full feature set used by the HGB classifier (includes the line identifier).
FEATURES: List[str] = [
    "line_id_encoded",
    "sum_load_p",
    "sum_load_q",
    "sum_gen_p",
    "var_line_rho",
    "avg_line_rho",
    "max_line_rho",
    "nb_rho_ge_0.95",
    "aleatoric_load_p_mean",
    "aleatoric_load_q_mean",
    "aleatoric_gen_p_mean",
    "load_gen_ratio",
    "epistemic_before",
    "epistemic_after",
    "fcast_sum_load_p",
    "fcast_sum_load_q",
    "fcast_sum_gen_p",
    "fcast_var_line_rho",
    "fcast_avg_line_rho",
    "fcast_max_line_rho",
    "fcast_nb_rho_ge_0.95",
]

#: Subset exposed to the LLM (line identifier excluded to avoid data leakage).
FEATURES_FOR_LLM: List[str] = [f for f in FEATURES if f != "line_id_encoded"]

#: Uncertainty features — used to guide the LLM toward interpretable rules that
#: integrate the epistemic and aleatoric uncertainty signals.
UNCERTAINTY_FEATURES: List[str] = [
    "epistemic_before",
    "epistemic_after",
    "aleatoric_load_p_mean",
    "aleatoric_load_q_mean",
    "aleatoric_gen_p_mean",
]

#: Grid-state features (complement of uncertainty features within FEATURES_FOR_LLM).
GRID_STATE_FEATURES: List[str] = [
    f for f in FEATURES_FOR_LLM if f not in UNCERTAINTY_FEATURES
]

#: Human-readable descriptions for prompt construction.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "sum_load_p":            "Total active load (MW).",
    "sum_load_q":            "Total reactive load (MVAR).",
    "sum_gen_p":             "Total active generation (MW).",
    "var_line_rho":          "Variance of line loading (rho) across all lines.",
    "avg_line_rho":          "Average line loading (rho).",
    "max_line_rho":          "Maximum line loading (rho).",
    "nb_rho_ge_0.95":        "Number of lines with rho >= 0.95.",
    "aleatoric_load_p_mean": "Mean aleatoric uncertainty for active load.",
    "aleatoric_load_q_mean": "Mean aleatoric uncertainty for reactive load.",
    "aleatoric_gen_p_mean":  "Mean aleatoric uncertainty for active generation.",
    "load_gen_ratio":        "Ratio of total load to total generation.",
    "epistemic_before":      "Epistemic uncertainty of the RL agent at time t.",
    "epistemic_after":       "Epistemic uncertainty of the RL agent at t+12 (before disconnection).",
    "fcast_sum_load_p":      "Forecasted total active load at t+12 (MW).",
    "fcast_sum_load_q":      "Forecasted total reactive load at t+12 (MVAR).",
    "fcast_sum_gen_p":       "Forecasted total active generation at t+12 (MW).",
    "fcast_var_line_rho":    "Forecasted variance of line loading (rho) at t+12.",
    "fcast_avg_line_rho":    "Forecasted average line loading (rho) at t+12.",
    "fcast_max_line_rho":    "Forecasted maximum line loading (rho) at t+12.",
    "fcast_nb_rho_ge_0.95":  "Forecasted number of lines with rho >= 0.95 at t+12.",
}

#: Mapping from line name to integer identifier used by the HGB classifier.
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


# ---------------------------------------------------------------------------
# Experiment hyper-parameters
# ---------------------------------------------------------------------------

#: LLM temperatures to evaluate. Higher values produce more diverse rules.
TEMPERATURES: List[float] = [0.3, 0.5, 0.7, 0.8]

#: Number of Generator-Critic iterations per line per temperature.
ITERATIONS: int = 500

#: Maximum size of the balanced training window passed to the LLM.
WINDOW_SIZE: int = 120

#: Maximum number of positive (failure) samples in the training window.
MAX_POSITIVE_IN_WINDOW: int = 40

#: Target ratio of negative to positive samples in the training window.
NEGATIVE_RATIO: float = 2.0

# Minimum sample counts for split strategy selection.
MIN_POS_TRAIN: int = 10
MIN_POS_TEST: int = 5
MIN_TOTAL_SAMPLES_PER_LINE: int = 20
MIN_POS_STANDARD: int = MIN_POS_TRAIN + MIN_POS_TEST
MIN_POS_LOO_THRESHOLD: int = 4
MIN_TOTAL_SAMPLES_SPARSE: int = 6

# Prompt truncation limits (characters).
MAX_HISTORY_ITEMS_FOR_PROMPT: int = 6
MAX_WORST_CASES_FOR_PROMPT: int = 10
MAX_RULE_CHARS: int = 12_000
MAX_JUSTIFICATION_CHARS: int = 3_000
MAX_CHANGES_CHARS: int = 2_000
MAX_FEEDBACK_CHARS: int = 4_000

# LLM API settings.
API_TIMEOUT: int = 180
API_RETRIES: int = 3
SLEEP_BETWEEN_RETRIES: int = 3

RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#: Reference metrics of the HGB teacher model (used in prompt construction).
HGB_TARGET: Dict[str, float] = {
    "acc": 0.9563,
    "f2":  0.7719,
    "ovr": 0.0567,
    "fa":  0.1083,
}


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def scoring_function(m: Dict[str, float]) -> float:
    """Compute the composite rule quality score.

    The score is designed for a safety-critical, class-imbalanced setting where
    missing a failure (false negative) is far more costly than a false alarm.

    Base score
    ----------
        Score = 0.5 * F2 + 0.3 * (1 - OVR) + 0.2 * (1 - FA)

    Progressive penalties
    ---------------------
    - OVR > 0.10: mild penalty up to 0.15; severe penalty above 0.50.
    - F2  < 0.30: penalty up to 0.15.
    - FA  > 0.12: mild penalty up to 0.15; severe penalty above 0.35.

    Hard floors
    -----------
    - F2 == 0.0  -> -1.00 (rule never detects any failure).
    - FA >= 0.90 -> -0.90 (rule fires on almost all observations).

    Parameters
    ----------
    m:
        Dictionary with keys ``f2``, ``ovr``, and ``fa``.

    Returns
    -------
    float
        Score in the range [-1.0, ~0.86].
    """
    if m["f2"] == 0.0:
        return -1.0
    if m["fa"] >= 0.90:
        return -0.90

    base = 0.50 * m["f2"] + 0.30 * (1.0 - m["ovr"]) + 0.20 * (1.0 - m["fa"])

    if m["ovr"] > 0.10:
        base -= min(0.15, (m["ovr"] - 0.10) * 0.40)
        base -= min(0.25, max(0.0, m["ovr"] - 0.50) * 0.60)
    if m["f2"] < 0.30:
        base -= min(0.15, (0.30 - m["f2"]) * 0.50)
    if m["fa"] > 0.12:
        base -= min(0.15, (m["fa"] - 0.12) * 0.40)
        base -= min(0.20, max(0.0, m["fa"] - 0.35) * 0.55)

    return max(-0.89, base)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if they do not exist."""
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any) -> float:
    """Convert *x* to float, returning NaN on failure."""
    try:
        return float("nan") if x is None else float(x)
    except Exception:
        return float("nan")


def clip_text(text: Optional[str], max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters."""
    return "" if text is None else str(text)[:max_chars]


def dict_to_pretty_json(d: Any) -> str:
    """Serialize *d* to a human-readable JSON string."""
    return json.dumps(d, ensure_ascii=False, indent=2, default=str)


def load_llm_config() -> Dict[str, Any]:
    """Load the LLM API configuration from disk."""
    with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics for a binary prediction.

    Returns a dictionary with keys: ``acc``, ``f2``, ``fa``, ``ovr``,
    ``tp``, ``fp``, ``fn``, ``tn``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = accuracy_score(y_true, y_pred)
    f2  = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fa  = fp / (fp + tn + 1e-12)
    ovr = fn / (fn + tp + 1e-12)
    return {
        "acc": float(acc), "f2": float(f2),
        "fa":  float(fa),  "ovr": float(ovr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any required column is missing from *df*."""
    required = set(FEATURES + ["line_disconnected"])
    missing  = [c for c in required if c not in df.columns and c != "line_id_encoded"]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def encode_line_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``line_disconnected`` column to integer identifiers."""
    df = df.copy()
    df["line_id_encoded"] = df["line_disconnected"].map(LINE_MAP)
    if df["line_id_encoded"].isna().any():
        unknown = sorted(
            df.loc[df["line_id_encoded"].isna(), "line_disconnected"]
            .dropna().unique().tolist()
        )
        raise ValueError(f"Unknown lines encountered: {unknown}")
    df["line_id_encoded"] = df["line_id_encoded"].astype(int)
    return df


def build_teacher_target(df: pd.DataFrame) -> pd.DataFrame:
    """Generate binary failure labels using the HGB teacher model.

    If the HGB model file is not found, falls back to the ``failed`` column.
    """
    df = df.copy()
    if HGB_MODEL_PATH and os.path.exists(HGB_MODEL_PATH):
        model = joblib.load(HGB_MODEL_PATH)
        df["target"] = model.predict(df[FEATURES].copy()).astype(int)
        df["target_source"] = "teacher_hgb"
    else:
        if "failed" not in df.columns:
            raise ValueError("HGB model not found and 'failed' column is missing.")
        df["target"] = df["failed"].astype(int)
        df["target_source"] = "failed_fallback"
    return df


# ---------------------------------------------------------------------------
# Train / test split strategies
# ---------------------------------------------------------------------------

def sequential_guarded_split(
    df_line: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential split that guarantees minimum positives in both partitions.

    Iterates over candidate split ratios (60-85%) and returns the first split
    where both the training and test sets contain at least ``MIN_POS_TRAIN``
    and ``MIN_POS_TEST`` positive samples, respectively.
    """
    df_line = df_line.sort_index().reset_index(drop=True)
    if len(df_line) < MIN_TOTAL_SAMPLES_PER_LINE:
        raise RuntimeError(f"Not enough samples for this line: {len(df_line)}")
    for split in np.linspace(0.60, 0.85, 11):
        cut = int(len(df_line) * split)
        train, test = df_line.iloc[:cut].copy(), df_line.iloc[cut:].copy()
        if (int(train["target"].sum()) >= MIN_POS_TRAIN
                and int(test["target"].sum()) >= MIN_POS_TEST):
            return train, test
    raise RuntimeError("No valid sequential split found for this line.")


def adaptive_split(
    df_line: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adaptive split for lines with few positive samples.

    Tries multiple split ratios and falls back to placing the last positive
    sample in the test set if no balanced split is found.
    """
    df_line = df_line.sort_index().reset_index(drop=True)
    n_pos   = int(df_line["target"].sum())
    if len(df_line) < MIN_TOTAL_SAMPLES_SPARSE:
        raise RuntimeError(f"Too few rows for adaptive split: {len(df_line)}")
    for split in np.linspace(0.50, 0.90, 17):
        cut = int(len(df_line) * split)
        train, test = df_line.iloc[:cut].copy(), df_line.iloc[cut:].copy()
        if (int(train["target"].sum()) >= max(2, n_pos - 2)
                and int(test["target"].sum()) >= 1):
            return train, test
    pos_indices = df_line.index[df_line["target"] == 1].tolist()
    if len(pos_indices) >= 2:
        idx   = pos_indices[-1]
        train = df_line.iloc[:idx].copy()
        test  = df_line.iloc[idx:].copy()
        if int(train["target"].sum()) >= 1 and int(test["target"].sum()) >= 1:
            return train, test
    raise RuntimeError("adaptive_split failed for this line.")


def leave_one_out_splits(
    df_line: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate leave-one-out folds over positive samples.

    Each fold uses one positive sample as the test set and the rest as training.
    Used for lines with very few positive samples (< ``MIN_POS_LOO_THRESHOLD``).
    """
    df_line     = df_line.sort_index().reset_index(drop=True)
    pos_indices = df_line.index[df_line["target"] == 1].tolist()
    splits = [
        (df_line.drop(index=idx).copy(), df_line.loc[[idx]].copy())
        for idx in pos_indices
        if len(df_line.drop(index=idx)) > 0
    ]
    if not splits:
        raise RuntimeError("No valid LOO folds for this line.")
    return splits


# ---------------------------------------------------------------------------
# Training window and feature statistics
# ---------------------------------------------------------------------------

def build_balanced_window(train_df: pd.DataFrame) -> pd.DataFrame:
    """Sample a balanced subset of the training data for prompt construction.

    The window maintains a ``NEGATIVE_RATIO``-to-1 ratio of negative to positive
    samples and is capped at ``WINDOW_SIZE`` rows to fit within the LLM context.
    """
    positives = train_df[train_df["target"] == 1]
    negatives = train_df[train_df["target"] == 0]
    if len(positives) == 0:
        return train_df.head(min(WINDOW_SIZE, len(train_df))).copy()

    pos_n      = min(len(positives), MAX_POSITIVE_IN_WINDOW)
    pos_sample = positives.sample(n=pos_n, random_state=random.randint(0, 1_000_000))
    neg_n      = min(len(negatives), max(1, int(math.ceil(pos_n * NEGATIVE_RATIO))))
    neg_sample = (
        negatives.sample(n=neg_n, random_state=random.randint(0, 1_000_000))
        if neg_n > 0 else negatives.head(0)
    )
    window = pd.concat([pos_sample, neg_sample], axis=0)

    remaining_n = max(0, WINDOW_SIZE - len(window))
    if remaining_n > 0:
        remaining = train_df.drop(index=window.index, errors="ignore")
        if len(remaining) > 0:
            add_n  = min(len(remaining), remaining_n)
            window = pd.concat(
                [window, remaining.sample(n=add_n, random_state=random.randint(0, 1_000_000))],
                axis=0,
            )
    return window.sort_index().copy()


def summarize_window(
    window: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Compute summary statistics (min, mean, max) for each feature in the window."""
    summary: Dict[str, Any] = {
        "n_rows":        int(len(window)),
        "n_positive":    int(window["target"].sum()),
        "positive_rate": float(window["target"].mean()) if len(window) > 0 else float("nan"),
        "feature_stats": {
            c: {
                "min":  safe_float(pd.to_numeric(window[c], errors="coerce").min()),
                "mean": safe_float(pd.to_numeric(window[c], errors="coerce").mean()),
                "max":  safe_float(pd.to_numeric(window[c], errors="coerce").max()),
            }
            for c in feature_cols
        },
    }
    return summary


def compute_feature_anchors(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Compute class-separated quantile anchors for each feature.

    Anchors are sorted in descending order of the absolute difference between
    class medians, so the most discriminative features appear first in the prompt.

    Returns
    -------
    Dict[str, Any]
        Mapping from feature name to a dictionary with keys:
        ``pos_p25``, ``pos_p50``, ``pos_p75``, ``neg_p25``, ``neg_p50``,
        ``neg_p75``, ``median_diff``, ``suggested_threshold``.
    """
    pos = train_df[train_df["target"] == 1]
    neg = train_df[train_df["target"] == 0]
    anchors: Dict[str, Any] = {}
    for f in feature_cols:
        s_pos = pd.to_numeric(pos[f], errors="coerce").dropna()
        s_neg = pd.to_numeric(neg[f], errors="coerce").dropna()
        if len(s_pos) == 0 or len(s_neg) == 0:
            continue
        anchors[f] = {
            "pos_p25":             round(float(s_pos.quantile(0.25)), 4),
            "pos_p50":             round(float(s_pos.quantile(0.50)), 4),
            "pos_p75":             round(float(s_pos.quantile(0.75)), 4),
            "neg_p25":             round(float(s_neg.quantile(0.25)), 4),
            "neg_p50":             round(float(s_neg.quantile(0.50)), 4),
            "neg_p75":             round(float(s_neg.quantile(0.75)), 4),
            "median_diff":         round(float(s_pos.median() - s_neg.median()), 4),
            "suggested_threshold": round(float((s_pos.median() + s_neg.median()) / 2.0), 4),
        }
    return dict(sorted(anchors.items(), key=lambda kv: abs(kv[1]["median_diff"]), reverse=True))


# ---------------------------------------------------------------------------
# Heuristic seed rules
# ---------------------------------------------------------------------------

def compute_seed_rules(
    train_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_anchors: Dict[str, Any],
    top_n_features: int = 6,
) -> List[Dict[str, Any]]:
    """Generate data-driven seed rules to initialise the LLM search.

    Seed rules are bivariate AND combinations of grid-state features paired with
    uncertainty features. Univariate rules are excluded because a single condition
    cannot simultaneously achieve low OVR and low FA in this imbalanced setting.

    Returns
    -------
    List[Dict[str, Any]]
        Up to 10 unique rules sorted by score (descending), each containing
        ``rule_code``, ``score``, and all keys from ``compute_metrics``.
    """
    top_features      = list(feature_anchors.keys())[:top_n_features]
    top_grid_features = [f for f in top_features if f not in UNCERTAINTY_FEATURES]
    top_unc_features  = [f for f in top_features if f in UNCERTAINTY_FEATURES]

    # Ensure uncertainty features are always represented.
    for uf in UNCERTAINTY_FEATURES:
        if uf in feature_anchors and uf not in top_unc_features:
            top_unc_features.append(uf)

    candidates: List[Dict[str, Any]] = []

    def _make_and_rule(conditions: List[Tuple[str, str, float]]) -> str:
        """Build a nested if-else rule from a list of (feature, op, threshold) tuples."""
        if not conditions:
            return ""
        indent = "    "
        lines  = ["def rule(x):"]
        depth  = 0
        for feat, op, thresh in conditions:
            lines.append(f"{indent * (depth + 1)}if x[\"{feat}\"] {op} {thresh}:")
            depth += 1
        lines.append(f"{indent * (depth + 1)}return 1")
        for _ in conditions:
            depth -= 1
            lines.append(f"{indent * (depth + 1)}else:")
            lines.append(f"{indent * (depth + 2)}return 0")
            if depth == 0:
                break
        return "\n".join(lines)

    def _eval(rule_code: str) -> Optional[Dict[str, Any]]:
        try:
            valid, _ = validate_rule_code(rule_code)
            if not valid:
                return None
            y_pred = evaluate_rule(rule_code, X_test)
            m = compute_metrics(y_test, y_pred)
            return {"rule_code": rule_code, "score": scoring_function(m), **m}
        except Exception:
            return None

    # Primary seeds: grid-state feature AND uncertainty feature.
    for gf in top_grid_features[:4]:
        for uf in top_unc_features[:3]:
            if gf not in feature_anchors or uf not in feature_anchors:
                continue
            tg  = feature_anchors[gf]["pos_p25"]
            opg = ">=" if feature_anchors[gf]["median_diff"] > 0 else "<="
            tu  = feature_anchors[uf]["suggested_threshold"]
            opu = ">=" if feature_anchors[uf]["median_diff"] > 0 else "<="
            for fac_g in [-0.20, -0.10, 0.0, 0.10]:
                for fac_u in [-0.20, 0.0, 0.20]:
                    ttg = round(tg * (1.0 + fac_g), 6) if tg != 0 else tg
                    ttu = round(tu * (1.0 + fac_u), 6) if tu != 0 else tu
                    r   = _eval(_make_and_rule([(gf, opg, ttg), (uf, opu, ttu)]))
                    if r:
                        candidates.append(r)

    # Fallback seeds: grid-state feature AND grid-state feature.
    for i in range(min(3, len(top_grid_features))):
        for j in range(i + 1, min(4, len(top_grid_features))):
            f1, f2 = top_grid_features[i], top_grid_features[j]
            if f1 not in feature_anchors or f2 not in feature_anchors:
                continue
            t1  = feature_anchors[f1]["pos_p25"]
            op1 = ">=" if feature_anchors[f1]["median_diff"] > 0 else "<="
            t2  = feature_anchors[f2]["suggested_threshold"]
            op2 = ">=" if feature_anchors[f2]["median_diff"] > 0 else "<="
            for fac1 in [-0.10, 0.0, 0.10]:
                for fac2 in [-0.10, 0.0, 0.10]:
                    tt1 = round(t1 * (1.0 + fac1), 6) if t1 != 0 else t1
                    tt2 = round(t2 * (1.0 + fac2), 6) if t2 != 0 else t2
                    r   = _eval(_make_and_rule([(f1, op1, tt1), (f2, op2, tt2)]))
                    if r:
                        candidates.append(r)

    # Deduplicate and return the top 10 by score.
    seen:   set               = set()
    unique: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c["rule_code"] not in seen:
            seen.add(c["rule_code"])
            unique.append(c)
        if len(unique) >= 10:
            break
    return unique


# ---------------------------------------------------------------------------
# Rule extraction and validation
# ---------------------------------------------------------------------------

#: Regex patterns for extracting tagged sections from LLM responses.
TAG_PATTERNS: Dict[str, Tuple[str, int]] = {
    "justification": (r"\[Start of Justification\](.*?)\[End of Justification\]", re.DOTALL | re.IGNORECASE),
    "changes":       (r"\[Start of Changes\](.*?)\[End of Changes\]",             re.DOTALL | re.IGNORECASE),
    "rule":          (r"\[Start of Rule\](.*?)\[End of Rule\]",                   re.DOTALL | re.IGNORECASE),
}


def extract_tagged_section(text: str, section_name: str) -> str:
    """Extract the content of a named tagged section from an LLM response."""
    pattern, flags = TAG_PATTERNS[section_name]
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def extract_python_rule(text: str) -> str:
    """Extract the Python ``def rule(x)`` function from an LLM response.

    Checks (in order): tagged section, fenced code block, raw function definition.
    """
    if not text:
        return ""
    text   = text.strip()
    tagged = extract_tagged_section(text, "rule")
    if tagged and "def rule" in tagged:
        text = tagged

    for block in re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        if "def rule" in block:
            return block.strip()

    m = re.search(r"(def\s+rule\s*\(\s*\w+\s*\)\s*:\s*[\s\S]*)", text)
    if m:
        candidate = m.group(1).strip()
        cleaned   = []
        for line in candidate.splitlines():
            if not line.strip() and cleaned:
                cleaned.append(line)
                continue
            if cleaned and re.match(r"^[A-Za-z\[\]()_-]", line) and not line.startswith((" ", "\t")):
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()
    return ""


_RULE_FEATURE_REGEX = re.compile(r'x\[(?:"|\')([^"\']+)(?:"|\')\]')


def validate_rule_code(rule_code: str) -> Tuple[bool, str]:
    """Validate that a candidate rule is syntactically and structurally compliant.

    Checks performed
    ----------------
    - Contains a ``def rule`` function definition.
    - Does not reference any forbidden token (imports, system calls, etc.).
    - Only references features from ``FEATURES_FOR_LLM``.
    - Compiles without syntax errors.

    Returns
    -------
    Tuple[bool, str]
        ``(True, "")`` if valid; ``(False, error_message)`` otherwise.
    """
    if not rule_code or "def rule" not in rule_code:
        return False, "Missing rule function definition."
    forbidden = [
        "import ", "__import__", "open(", "exec(", "eval(",
        "os.", "sys.", "subprocess", "pickle", "joblib", "line_id_encoded",
    ]
    for token in forbidden:
        if token in rule_code:
            return False, f"Forbidden token found: '{token}'."
    referenced = set(_RULE_FEATURE_REGEX.findall(rule_code))
    disallowed  = sorted(f for f in referenced if f not in FEATURES_FOR_LLM)
    if disallowed:
        return False, f"Disallowed features referenced: {disallowed}."
    try:
        compile(rule_code, "<rule_code>", "exec")
    except Exception as e:
        return False, f"Compilation error: {e}."
    return True, ""


def evaluate_rule(rule_code: str, X: pd.DataFrame) -> np.ndarray:
    """Execute a rule function on a feature DataFrame and return binary predictions."""
    local_env: Dict[str, Any] = {}
    exec(rule_code, {}, local_env)  # noqa: S102
    if "rule" not in local_env:
        raise ValueError("Rule code did not define the function 'rule'.")
    rule_fn = local_env["rule"]
    return np.asarray(
        [1 if int(rule_fn(row)) == 1 else 0 for _, row in X.iterrows()],
        dtype=int,
    )


def get_worst_cases(
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_items: int = MAX_WORST_CASES_FOR_PROMPT,
) -> List[Dict[str, Any]]:
    """Return the most informative misclassified samples for prompt construction.

    False negatives (missed failures) are listed first as they are the most
    safety-critical errors in this application.
    """
    rows = []
    for i in range(len(X_test)):
        if int(y_true[i]) != int(y_pred[i]):
            row = {k: safe_float(X_test.iloc[i][k]) for k in X_test.columns}
            row["y_true"]     = int(y_true[i])
            row["y_pred"]     = int(y_pred[i])
            row["error_type"] = "FN" if (int(y_true[i]) == 1 and int(y_pred[i]) == 0) else "FP"
            rows.append(row)
    rows.sort(key=lambda r: 0 if r["error_type"] == "FN" else 1)
    return rows[:max_items]


def compact_history_for_prompt(
    history: List[Dict[str, Any]],
    max_items: int = MAX_HISTORY_ITEMS_FOR_PROMPT,
) -> List[Dict[str, Any]]:
    """Return the most recent *max_items* history entries in a compact format."""
    return [
        {
            "iteration":  h.get("iteration"),
            "score":      h.get("score"),
            "f2":         h.get("f2"),
            "ovr":        h.get("ovr"),
            "fa":         h.get("fa"),
            "valid_rule": h.get("valid_rule"),
            "changes":    clip_text(h.get("changes", ""), 300),
        }
        for h in history[-max_items:]
    ]


def build_feature_description_text() -> str:
    """Build the feature description block used in generator prompts.

    Uncertainty features are highlighted to make them salient to the LLM.
    """
    lines = ["Grid-state features:"]
    for f in GRID_STATE_FEATURES:
        lines.append(f"  - {f}: {FEATURE_DESCRIPTIONS.get(f, '')}")
    lines.append("")
    lines.append("Uncertainty features (use at least one in your rule):")
    for f in UNCERTAINTY_FEATURES:
        lines.append(f"  - {f}: {FEATURE_DESCRIPTIONS.get(f, '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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
    """Build the Generator LLM prompt for a given iteration.

    The prompt includes the scoring formula, mandatory rule structure, class-separated
    feature anchors, a self-diagnosis section mapping previous metrics to a concrete
    corrective action, and the balanced training window with worst misclassified cases.
    """
    training_json = clip_text(
        window[FEATURES_FOR_LLM + ["target"]].to_json(orient="records"), 6000
    )

    # Separate anchor blocks for grid-state and uncertainty features.
    unc_anchors  = {f: v for f, v in feature_anchors.items() if f in UNCERTAINTY_FEATURES}
    grid_anchors = {
        f: v for f, v in dict(list(feature_anchors.items())[:6]).items()
        if f not in UNCERTAINTY_FEATURES
    }

    anchors_text = "\nGrid-state threshold anchors (sorted by discriminative power):\n"
    anchors_text += "  Use pos_p25 for Case A/CRITICAL, not suggested_threshold.\n"
    for f, v in grid_anchors.items():
        direction = "HIGHER in failures" if v["median_diff"] > 0 else "LOWER in failures"
        anchors_text += (
            f"  {f}:\n"
            f"    failures  p25={v['pos_p25']}  p50={v['pos_p50']}  p75={v['pos_p75']}\n"
            f"    normal    p25={v['neg_p25']}  p50={v['neg_p50']}  p75={v['neg_p75']}\n"
            f"    -> {direction} | suggested={v['suggested_threshold']} | pos_p25={v['pos_p25']}\n"
        )
    anchors_text += "\nUncertainty feature anchors (use at least one):\n"
    if unc_anchors:
        for f, v in unc_anchors.items():
            direction = "HIGHER in failures" if v["median_diff"] > 0 else "LOWER in failures"
            anchors_text += (
                f"  {f}:\n"
                f"    failures  p25={v['pos_p25']}  p50={v['pos_p50']}  p75={v['pos_p75']}\n"
                f"    normal    p25={v['neg_p25']}  p50={v['neg_p50']}  p75={v['neg_p75']}\n"
                f"    -> {direction} | suggested={v['suggested_threshold']}\n"
            )
    else:
        anchors_text += "  (No uncertainty anchors available for this line.)\n"

    # Retrieve top feature metadata for concrete threshold suggestions.
    top_feat   = list(feature_anchors.keys())[0] if feature_anchors else None
    top_anchor = feature_anchors.get(top_feat, {}) if top_feat else {}
    top_p25    = top_anchor.get("pos_p25", "N/A")
    top_sug    = top_anchor.get("suggested_threshold", "N/A")
    top_dir    = ">=" if top_anchor.get("median_diff", 1) > 0 else "<="

    # Self-diagnosis: map previous metrics to a labelled case and action.
    prev_ovr = previous_metrics.get("ovr") if previous_metrics else None
    prev_fa  = previous_metrics.get("fa")  if previous_metrics else None
    prev_f2  = previous_metrics.get("f2")  if previous_metrics else None

    if prev_f2 is not None:
        if prev_f2 == 0.0 or (prev_ovr is not None and prev_ovr > 0.80):
            case   = f"CRITICAL — F2={prev_f2:.3f}, OVR={prev_ovr:.3f}"
            action = (
                f"Drastically lower the main threshold to pos_p25.\n"
                f"  CONCRETE: x[\"{top_feat}\"] {top_dir} {top_p25}"
            )
        elif prev_ovr is not None and prev_ovr > 0.40:
            case   = f"A — OVR={prev_ovr:.3f}: too many missed failures"
            action = (
                f"Lower the main threshold toward pos_p25.\n"
                f"  CONCRETE: x[\"{top_feat}\"] {top_dir} {top_p25} "
                f"(suggested={top_sug}, pos_p25={top_p25})."
            )
        elif prev_fa is not None and prev_fa > 0.15:
            top_unc = next(iter(unc_anchors), None)
            unc_val = unc_anchors[top_unc]["suggested_threshold"] if top_unc else "N/A"
            unc_dir = ">=" if (top_unc and unc_anchors[top_unc]["median_diff"] > 0) else "<="
            case    = f"B — FA={prev_fa:.3f}: too many false alarms"
            action  = (
                f"Add an uncertainty AND condition.\n"
                f"  CONCRETE: add x[\"{top_unc}\"] {unc_dir} {unc_val}.\n"
                f"  Do NOT raise the grid-state threshold (would increase OVR)."
            )
        elif (prev_ovr is not None and prev_ovr > 0.10) or (prev_fa is not None and prev_fa > 0.20):
            if prev_fa is not None and prev_fa > 0.20:
                case   = f"C — FA={prev_fa:.3f}: FA still above 0.20"
                action = "Add a third AND condition using a forecast feature."
            else:
                case   = f"C — OVR={prev_ovr:.3f}, FA={prev_fa:.3f}: moderate errors"
                action = "Adjust the threshold causing the most FN or FP cases."
        else:
            case   = f"D — OVR={prev_ovr:.3f}, FA={prev_fa:.3f}: near target"
            action = "Micro-adjust a single threshold by at most 5%. No structural changes."
        diagnosis_text = f"Case: {case}\nAction required: {action}"
    else:
        # First iteration: mandate OR structure with 2 paths.
        top_unc     = next(iter(unc_anchors), None)
        unc_val     = unc_anchors[top_unc]["suggested_threshold"] if top_unc else "N/A"
        unc_dir     = ">=" if (top_unc and unc_anchors[top_unc]["median_diff"] > 0) else "<="
        fcast_feats = [f for f in FEATURES_FOR_LLM if f.startswith("fcast_") and f in feature_anchors]
        fcast_feat  = fcast_feats[0] if fcast_feats else top_feat
        fcast_val   = feature_anchors[fcast_feat]["pos_p25"] if fcast_feat in feature_anchors else "N/A"
        fcast_dir   = ">=" if (fcast_feat in feature_anchors and feature_anchors[fcast_feat]["median_diff"] > 0) else "<="
        diagnosis_text = (
            f"First iteration — build a rule with 2 distinct failure paths:\n"
            f"  PATH 1 (current state + uncertainty):\n"
            f"    x[\"{top_feat}\"] {top_dir} {top_p25} AND x[\"{top_unc}\"] {unc_dir} {unc_val}\n"
            f"  PATH 2 (forecasted state):\n"
            f"    x[\"{fcast_feat}\"] {fcast_dir} {fcast_val}\n"
            f"  PATH 1 catches current OOD failures; PATH 2 catches predicted future failures."
        )

    alerts = ""
    if oscillation_note:
        alerts += f"\n[WARNING] {oscillation_note}\n"
    if stagnation_note:
        alerts += f"\n[WARNING] {stagnation_note}\n"

    seed_section = ""
    if seed_summary:
        seed_section = "\n## Data-driven seed rules (baseline — improve on these)\n"
        for s in seed_summary:
            seed_section += (
                f"  Rank {s['rank']}: score={s['score']}  f2={s['f2']}  "
                f"ovr={s['ovr']}  fa={s['fa']}\n  {s['rule']}\n\n"
            )

    feature_stats_text = ""
    if "feature_stats" in window_summary:
        feature_stats_text = "Feature statistics (min / mean / max):\n"
        for f, stats in window_summary["feature_stats"].items():
            feature_stats_text += (
                f"  {f}: min={stats['min']:.4f}  mean={stats['mean']:.4f}  max={stats['max']:.4f}\n"
            )

    return f"""
You are the Generator LLM in an iterative two-LLM framework for power-grid failure prediction.

Objective: generate ONE Python if/else rule that maximises the F2-score.
This is a safety-critical, class-imbalanced problem — missing a failure is worse than a false alarm.
{alerts}{seed_section}
## Problem context
Line: {line_name}
Teacher HGB: Acc={HGB_TARGET['acc']:.4f} | F2={HGB_TARGET['f2']:.4f} | OVR={HGB_TARGET['ovr']:.4f} | FA={HGB_TARGET['fa']:.4f}
Targets: F2 >= 0.70 | OVR <= 0.10 | FA <= 0.20

## Scoring formula
  Score = 0.50 * F2 + 0.30 * (1 - OVR) + 0.20 * (1 - FA)
  Progressive penalties: OVR > 0.10, FA > 0.12, F2 < 0.30
  Hard floors: F2 == 0 -> -1.00 | FA >= 0.90 -> -0.90

## Mandatory rule structure
Rules must have exactly 2 failure paths (nested if-else only, no elif):
  PATH 1: current grid-state condition AND uncertainty condition
  PATH 2: forecasted grid-state condition

Example:
```python
def rule(x):
    if x["max_line_rho"] >= THRESHOLD_1:
        if x["epistemic_before"] >= UNC_THRESHOLD:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= THRESHOLD_2:
            return 1
        else:
            return 0
```

## Self-diagnosis
{diagnosis_text}

## {feature_stats_text}
{anchors_text}

## Critic feedback
{clip_text(feedback_text, 1500) if feedback_text else "None"}

## Previous rule
{clip_text(previous_rule, MAX_RULE_CHARS) if previous_rule else "None"}

## Previous metrics
{dict_to_pretty_json(previous_metrics) if previous_metrics else "None"}

## Best rule so far
{clip_text(best_rule, MAX_RULE_CHARS) if best_rule else "None"}

## Best metrics so far
{dict_to_pretty_json(best_metrics) if best_metrics else "None"}

## Iteration history (last {MAX_HISTORY_ITEMS_FOR_PROMPT})
{dict_to_pretty_json(history_table) if history_table else "None"}

## Training window
{training_json}

## Worst misclassified cases (FN = missed failures listed first)
{dict_to_pretty_json(worst_cases) if worst_cases else "None"}

## Feature descriptions
{build_feature_description_text()}

## Constraints (violation = INVALID)
- Only nested if-else (no elif, no loops)
- Only features from FEATURES_FOR_LLM (not line_id_encoded)
- No imports or external calls
- Return only 0 or 1
- Exactly 2 failure paths, each with 1-2 AND conditions
- At least one path must use an uncertainty feature

## Required output format
[Start of Justification]
State the case (CRITICAL/A/B/C/D) and the action taken.
[End of Justification]

[Start of Changes]
One sentence: what changed vs the previous rule.
[End of Changes]

[Start of Rule]
```python
def rule(x):
    ...
```
[End of Rule]
""".strip()


def repair_rule_prompt(bad_output: str, error_msg: str) -> str:
    """Build a prompt instructing the Repair LLM to fix an invalid rule."""
    return f"""
The rule below is invalid. Correct it.

Error: {error_msg}

Output ONLY the corrected rule in this format:

[Start of Rule]
```python
def rule(x):
    if x["max_line_rho"] >= 0.95:
        return 1
    else:
        return 0
```
[End of Rule]

Constraints: def rule(x), return only 0 or 1, no imports, no elif, no loops.

Invalid output to fix:
{bad_output}
""".strip()


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
    """Build the Critic LLM prompt for a given iteration.

    The Critic reviews the current rule's performance metrics and provides
    targeted feedback covering oscillation detection, uncertainty feature usage,
    and a labelled diagnostic case with concrete threshold suggestions.
    """
    cur_f2  = current_metrics.get("f2",  0.0)
    cur_ovr = current_metrics.get("ovr", 1.0)
    cur_fa  = current_metrics.get("fa",  1.0)

    # Top-5 anchor summary.
    anchors_summary = ""
    for f, v in dict(list(feature_anchors.items())[:5]).items():
        direction = "higher in failures" if v["median_diff"] > 0 else "lower in failures"
        anchors_summary += (
            f"  {f}: failures p25={v['pos_p25']} p50={v['pos_p50']} | "
            f"normal p50={v['neg_p50']} | suggested={v['suggested_threshold']} ({direction})\n"
        )

    # Oscillation detection over the last 4 valid iterations.
    recent_valid = [h for h in history_table if h.get("valid_rule") and h.get("ovr") is not None]
    ovr_vals = [h["ovr"] for h in recent_valid[-4:]]
    fa_vals  = [h["fa"]  for h in recent_valid[-4:]]
    oscillating = (
        len(ovr_vals) >= 3
        and sum(1 for v in ovr_vals if v > 0.50) >= 1
        and sum(1 for v in fa_vals  if v > 0.50) >= 1
    )
    osc_text = ""
    if oscillating:
        osc_text = (
            f"[OSCILLATION DETECTED]\n"
            f"OVR history: {[round(v,2) for v in ovr_vals]} | "
            f"FA history: {[round(v,2) for v in fa_vals]}\n"
            f"Provide ONE micro-adjustment only (5-15% on a single threshold).\n"
        )

    # Uncertainty feature check.
    unc_anchors = {f: v for f, v in feature_anchors.items() if f in UNCERTAINTY_FEATURES}
    top_unc     = next(iter(unc_anchors), None)
    top_unc_val = unc_anchors[top_unc]["suggested_threshold"] if top_unc else "N/A"
    top_unc_dir = ">=" if (top_unc and unc_anchors[top_unc]["median_diff"] > 0) else "<="
    uses_unc    = any(f in current_rule for f in UNCERTAINTY_FEATURES)

    unc_check = ""
    if not uses_unc:
        unc_check = (
            f"[NO UNCERTAINTY FEATURE]\n"
            f"Mandatory: instruct the generator to add "
            f"x[\"{top_unc}\"] {top_unc_dir} {top_unc_val}.\n"
        )
    elif cur_fa > 0.20:
        unc_check = (
            f"[FA TOO HIGH: {cur_fa:.3f}]\n"
            f"Options: tighten uncertainty threshold toward "
            f"p75={unc_anchors.get(top_unc,{}).get('pos_p75','N/A')}, "
            f"or add a third condition using a forecast feature.\n"
        )

    return f"""
You are the Critic LLM in an iterative two-LLM framework for power-grid failure prediction.

## Scoring formula
  Score = 0.50*F2 + 0.30*(1-OVR) + 0.20*(1-FA)
  Penalties: FA > 0.15, OVR > 0.10 | Hard floors: F2==0 -> -1.0 | FA>=0.90 -> -0.90
  Target: OVR <= 0.10 AND FA <= 0.15 AND F2 >= 0.70

## Current metrics
  F2={cur_f2:.4f}  OVR={cur_ovr:.4f}  FA={cur_fa:.4f}
  Teacher: F2={HGB_TARGET['f2']:.4f} | OVR={HGB_TARGET['ovr']:.4f} | FA={HGB_TARGET['fa']:.4f}

{osc_text}
{unc_check}

## Priority checks

1. DEGENERATE: F2 == 0 or TP == 0 -> lower main threshold to pos_p25 immediately.
2. Uncertainty check: does the rule use at least one uncertainty feature? If NO, add one.
3. Diagnostic case:
   A — OVR > 0.40: lower threshold toward pos_p25.
   B — FA > 0.15: add/tighten uncertainty condition; do NOT raise the grid-state threshold.
   C — OVR in (0.10, 0.40) or FA in (0.10, 0.40): adjust the threshold causing most errors.
   D — OVR < 0.10 and FA < 0.20: micro-adjust only.

## Feature anchors
{anchors_summary}

## Uncertainty anchors
{dict_to_pretty_json(unc_anchors) if unc_anchors else "None"}

## Current rule
{clip_text(current_rule, MAX_RULE_CHARS)}

## Current metrics (detail)
{dict_to_pretty_json(current_metrics)}

## Previous rule
{clip_text(previous_rule, MAX_RULE_CHARS) if previous_rule else "None"}

## Best rule so far
{clip_text(best_rule, MAX_RULE_CHARS) if best_rule else "None"}

## Best metrics so far
{dict_to_pretty_json(best_metrics) if best_metrics is not None else "None"}

## Iteration history
{dict_to_pretty_json(history_table)}

## Worst misclassified cases (FN first)
{dict_to_pretty_json(worst_cases)}

## Data summary
{dict_to_pretty_json(window_summary)}

## Response format
1. State the case (DEGENERATE / A / B / C / D).
2. State whether the rule uses an uncertainty feature (YES / NO).
3. If oscillating: ONE micro-adjustment only.
   Otherwise: up to 8 suggestions, each with feature + current threshold + new threshold.
4. Never raise thresholds for DEGENERATE or Case A.
5. Never lower thresholds for Case B.
6. Syntactic check: no elif | no imports | returns 0 or 1 | allowed features only -> PASS/FAIL
""".strip()


# ---------------------------------------------------------------------------
# LLM API
# ---------------------------------------------------------------------------

def call_llm(prompt: str, temperature: float) -> str:
    """Send a prompt to the LLM API and return the text response.

    Retries up to ``API_RETRIES`` times on failure, sleeping between attempts.

    Parameters
    ----------
    prompt:
        The full prompt string to send.
    temperature:
        Sampling temperature for the LLM (higher = more diverse output).

    Returns
    -------
    str
        The LLM's text response.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    config  = load_llm_config()
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    payload = {
        "model":       config["model"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  config.get("max_tokens", 4000),
    }
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.post(
                config["api_url"], headers=headers, json=payload,
                verify=config.get("verify_ssl", True),
                timeout=config.get("timeout", API_TIMEOUT),
            )
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            content = r.json()["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else str(content)
        except Exception as e:
            if attempt < API_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                raise RuntimeError(f"LLM call failed after {API_RETRIES} attempts: {e}") from e
    raise RuntimeError("Unexpected exit from retry loop.")


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

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
    """Run the Generator-Critic loop for one line, one temperature, and one data split.

    Executes the full seed-and-refine pipeline, saving the best rule and iteration
    history to *line_dir* at the end of every iteration.

    Parameters
    ----------
    df_line:
        Full dataset for this transmission line.
    train_df, test_df:
        Pre-computed train and test partitions.
    line_name:
        Human-readable line identifier (e.g. ``"41_48_131"``).
    line_id_encoded:
        Integer encoding used by the HGB model.
    temperature:
        LLM sampling temperature.
    line_dir:
        Directory where results for this line will be saved.
    split_mode:
        One of ``"standard"``, ``"adaptive"``, or ``"loo"``.
    fold_label:
        Fold identifier string (empty for non-LOO splits).

    Returns
    -------
    Dict[str, Any]
        Summary of the best rule found, including all evaluation metrics.
    """
    ensure_dir(line_dir)
    X_test = test_df[FEATURES_FOR_LLM].copy()
    y_test = test_df["target"].astype(int).values

    feature_anchors = compute_feature_anchors(train_df, FEATURES_FOR_LLM)

    # --- Seed phase ---
    seed_candidates = compute_seed_rules(
        train_df, X_test, y_test, feature_anchors, top_n_features=6,
    )
    best_rule          = ""
    best_justification = best_feedback = best_changes = ""
    best_metrics: Optional[Dict[str, Any]] = None
    best_score         = float("-inf")
    best_iteration:    Optional[int] = None

    if seed_candidates:
        top_seed = seed_candidates[0]
        if top_seed["score"] > best_score:
            best_rule         = top_seed["rule_code"]
            best_score        = top_seed["score"]
            best_metrics      = {k: top_seed[k] for k in ("acc", "f2", "fa", "ovr", "tp", "fp", "fn", "tn")}
            best_iteration    = 0
            best_justification = "Seed rule (data-driven)."
        print(
            f"  [SEED] score={top_seed['score']:.4f}  f2={top_seed['f2']:.4f}  "
            f"ovr={top_seed['ovr']:.4f}  fa={top_seed['fa']:.4f}"
        )
        with open(os.path.join(line_dir, "best_rule.py"), "w") as fh:
            fh.write(best_rule or "# No valid rule.\n")
        with open(os.path.join(line_dir, "best_justification.txt"), "w") as fh:
            fh.write(best_justification)

    seed_summary_for_prompt = [
        {
            "rank":  i + 1,
            "score": round(s["score"], 4), "f2":  round(s["f2"],  4),
            "ovr":   round(s["ovr"],   4), "fa":  round(s["fa"],  4),
            "rule":  s["rule_code"],
        }
        for i, s in enumerate(seed_candidates[:3])
    ]

    # --- Iterative refinement ---
    previous_rule:    str                   = ""
    previous_metrics: Optional[Dict]        = None
    previous_feedback: str                  = ""
    history:          List[Dict[str, Any]]  = []
    invalid_rule_count = api_error_count    = 0

    _recent_ovr:    List[float] = []
    _recent_fa:     List[float] = []
    _stagnant_iters             = 0
    _last_best_score            = float("-inf")

    for iteration in range(1, ITERATIONS + 1):
        window         = build_balanced_window(train_df)
        window_summary = summarize_window(window, FEATURES_FOR_LLM)
        history_table  = compact_history_for_prompt(history)

        # Oscillation detection.
        _oscillation_note = ""
        if len(_recent_ovr) >= 3:
            if (sum(1 for v in _recent_ovr[-4:] if v > 0.50) >= 1
                    and sum(1 for v in _recent_fa[-4:] if v > 0.50) >= 1):
                _oscillation_note = (
                    f"OSCILLATION: OVR={[round(v, 2) for v in _recent_ovr[-4:]]} "
                    f"FA={[round(v, 2) for v in _recent_fa[-4:]]}. "
                    f"Make ONE small change (5-10%) only."
                )

        _stagnation_note = ""
        if _stagnant_iters >= 8:
            _stagnation_note = (
                f"STAGNATION: no improvement for {_stagnant_iters} iterations. "
                f"Try a structurally different rule."
            )

        try:
            best_y_pred_wc = evaluate_rule(best_rule, X_test) if best_rule else np.zeros_like(y_test)
        except Exception:
            best_y_pred_wc = np.zeros_like(y_test)
        worst_cases_context = get_worst_cases(X_test, y_test, best_y_pred_wc)

        gen_prompt_text = generator_prompt(
            iteration=iteration, line_name=line_name, window=window,
            window_summary=window_summary, feature_anchors=feature_anchors,
            previous_rule=previous_rule, previous_metrics=previous_metrics,
            best_rule=best_rule, best_metrics=best_metrics,
            feedback_text=previous_feedback, history_table=history_table,
            worst_cases=worst_cases_context,
            oscillation_note=_oscillation_note, stagnation_note=_stagnation_note,
            seed_summary=seed_summary_for_prompt if iteration <= 3 else [],
        )

        rule_code = justification = changes = critic_feedback = ""
        try:
            raw_gen       = call_llm(gen_prompt_text, temperature)
            justification = clip_text(extract_tagged_section(raw_gen, "justification"), MAX_JUSTIFICATION_CHARS)
            changes       = clip_text(extract_tagged_section(raw_gen, "changes"),       MAX_CHANGES_CHARS)
            rule_code     = clip_text(extract_python_rule(raw_gen),                     MAX_RULE_CHARS)

            # Validate; attempt repair if invalid.
            valid, err = validate_rule_code(rule_code)
            if not valid:
                repaired      = call_llm(repair_rule_prompt(raw_gen, err), temperature)
                repaired_rule = extract_python_rule(repaired)
                valid, err    = validate_rule_code(repaired_rule)
                if valid:
                    rule_code = repaired_rule
                else:
                    invalid_rule_count += 1
                    history.append({
                        "iteration": iteration, "line_name": line_name,
                        "valid_rule": False, "rule_error": err,
                        "score": None, "f2": None, "ovr": None, "fa": None,
                        "acc": None, "tp": None, "fp": None, "fn": None, "tn": None,
                        "changes": changes, "feedback": "", "rule_code": rule_code,
                        "justification": justification,
                    })
                    previous_feedback = f"Rule invalid: {err}"
                    previous_rule     = rule_code or previous_rule
                    previous_metrics  = None
                    continue

            # Evaluate the validated rule.
            y_pred  = evaluate_rule(rule_code, X_test)
            metrics = compute_metrics(y_test, y_pred)
            score   = scoring_function(metrics)

            # Critic feedback.
            critic_input = critic_prompt(
                iteration=iteration, line_name=line_name, line_id_encoded=line_id_encoded,
                feature_anchors=feature_anchors, current_rule=rule_code,
                current_metrics=metrics, previous_rule=previous_rule,
                previous_metrics=previous_metrics, best_rule=best_rule,
                best_metrics=best_metrics, history_table=history_table,
                worst_cases=get_worst_cases(X_test, y_test, y_pred),
                window_summary=window_summary,
            )
            critic_feedback = clip_text(call_llm(critic_input, temperature), MAX_FEEDBACK_CHARS)

            history.append({
                "iteration": iteration, "line_name": line_name,
                "valid_rule": True, "rule_error": "",
                "score": score, **metrics,
                "changes": changes, "feedback": critic_feedback,
                "rule_code": rule_code, "justification": justification,
            })

            # Update best rule if score improved.
            if score > best_score:
                best_score         = score
                best_rule          = rule_code
                best_justification = justification
                best_feedback      = critic_feedback
                best_changes       = changes
                best_metrics       = metrics
                best_iteration     = iteration
                for fname, content in [
                    ("best_rule.py",           best_rule),
                    ("best_justification.txt", best_justification),
                    ("best_feedback.txt",      best_feedback),
                    ("best_changes.txt",       best_changes),
                ]:
                    with open(os.path.join(line_dir, fname), "w") as fh:
                        fh.write(content)

            # Update oscillation and stagnation trackers.
            _recent_ovr.append(metrics["ovr"])
            _recent_fa.append(metrics["fa"])
            if len(_recent_ovr) > 6:
                _recent_ovr.pop(0)
                _recent_fa.pop(0)
            if score > _last_best_score:
                _stagnant_iters  = 0
                _last_best_score = score
            else:
                _stagnant_iters += 1

            previous_rule     = rule_code
            previous_metrics  = metrics
            previous_feedback = critic_feedback

            if iteration % 10 == 0 or iteration == 1:
                print(
                    f"  [iter={iteration:3d}] score={score:.4f}  "
                    f"f2={metrics['f2']:.4f}  ovr={metrics['ovr']:.4f}  fa={metrics['fa']:.4f}"
                )

        except Exception as e:
            if "LLM call failed" in str(e) or "HTTP" in str(e):
                api_error_count += 1
            history.append({
                "iteration": iteration, "line_name": line_name,
                "valid_rule": False, "rule_error": str(e),
                "score": None, "f2": None, "ovr": None, "fa": None,
                "acc": None, "tp": None, "fp": None, "fn": None, "tn": None,
                "changes": changes, "feedback": critic_feedback,
                "rule_code": rule_code, "justification": justification,
            })
            previous_feedback = f"Previous iteration failed: {str(e)[:500]}"

        pd.DataFrame(history).to_csv(os.path.join(line_dir, "iteration_history.csv"), index=False)

    # --- Save final outputs ---
    if seed_candidates:
        pd.DataFrame([
            {
                "rank": i + 1, "score": s["score"], "f2":  s["f2"],
                "ovr":  s["ovr"],  "fa":  s["fa"],  "acc": s["acc"],
                "rule_code": s["rule_code"],
            }
            for i, s in enumerate(seed_candidates)
        ]).to_csv(os.path.join(line_dir, "seed_candidates.csv"), index=False)

    with open(os.path.join(line_dir, "best_rule.py"), "w") as fh:
        fh.write(best_rule or "# No valid rule was found.\n")
    with open(os.path.join(line_dir, "best_justification.txt"), "w") as fh:
        fh.write(best_justification or "No valid justification.\n")

    if best_metrics is None:
        best_metrics = {"acc": None, "f2": None, "fa": None, "ovr": None,
                        "tp": None, "fp": None, "fn": None, "tn": None}

    return {
        "line_name":           line_name,
        "line_id_encoded":     line_id_encoded,
        "best_iteration":      best_iteration,
        "best_score":          None if best_score == float("-inf") else float(best_score),
        **best_metrics,
        "n_total":             int(len(df_line)),
        "n_train":             int(len(train_df)),
        "n_test":              int(len(test_df)),
        "n_pos_total":         int(df_line["target"].sum()),
        "n_pos_train":         int(train_df["target"].sum()),
        "n_pos_test":          int(test_df["target"].sum()),
        "split_mode":          split_mode,
        "fold_label":          fold_label,
        "seed_best_score":     round(seed_candidates[0]["score"], 4) if seed_candidates else None,
        "seed_best_f2":        round(seed_candidates[0]["f2"],    4) if seed_candidates else None,
        "invalid_rule_count":  int(invalid_rule_count),
        "api_error_count":     int(api_error_count),
        "line_dir":            line_dir,
        "had_valid_rule":      bool(best_rule),
    }


def run_line_experiment(
    df_line: pd.DataFrame,
    line_name: str,
    line_id_encoded: int,
    temperature: float,
    output_dir: str,
    split_mode: str = "standard",
) -> Dict[str, Any]:
    """Run the full experiment for one transmission line and one temperature.

    Selects the appropriate split strategy based on *split_mode*. For LOO mode,
    runs all folds and returns the result of the best-scoring fold.
    """
    line_dir = os.path.join(output_dir, f"line_{line_name}")
    ensure_dir(line_dir)

    if split_mode == "loo":
        folds   = leave_one_out_splits(df_line)
        results = []
        for fold_i, (train_fold, test_fold) in enumerate(folds):
            results.append(_run_experiment_on_split(
                df_line=df_line, train_df=train_fold, test_df=test_fold,
                line_name=line_name, line_id_encoded=line_id_encoded,
                temperature=temperature,
                line_dir=os.path.join(line_dir, f"fold_{fold_i}"),
                split_mode="loo", fold_label=f"fold{fold_i}",
            ))
        best = max(results, key=lambda r: r["best_score"] or float("-inf"))
        best.update({"line_name": line_name, "split_mode": "loo", "n_loo_folds": len(folds)})
        return best

    train_df, test_df = (
        adaptive_split(df_line) if split_mode == "adaptive"
        else sequential_guarded_split(df_line)
    )
    return _run_experiment_on_split(
        df_line=df_line, train_df=train_df, test_df=test_df,
        line_name=line_name, line_id_encoded=line_id_encoded,
        temperature=temperature, line_dir=line_dir,
        split_mode=split_mode, fold_label="",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load data, build teacher labels, and run experiments for all lines and temperatures."""
    df = pd.read_csv(DATA_PATH)
    df["load_gen_ratio"] = df.apply(
        lambda r: (r["sum_load_p"] / r["sum_gen_p"]) if r.get("sum_gen_p", 0) != 0 else 0.0,
        axis=1,
    )
    validate_required_columns(df)
    df = encode_line_ids(df)
    df = build_teacher_target(df)

    global_summary: List[Dict[str, Any]] = []

    for temperature in TEMPERATURES:
        print(f"\n{'='*60}")
        print(f"  Temperature: {temperature}  |  Iterations: {ITERATIONS}")
        print(f"{'='*60}")

        temp_dir     = os.path.join(OUTPUT_DIR, f"temp_{temperature}")
        ensure_dir(temp_dir)
        summary_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []

        for line_name, line_id_encoded in LINE_MAP.items():
            # Resume support: skip lines that have already been processed.
            history_csv = os.path.join(temp_dir, f"line_{line_name}", "iteration_history.csv")
            if os.path.exists(history_csv):
                print(f"  [SKIP] {line_name}: already completed.")
                continue

            df_line     = df[df["line_disconnected"] == line_name].copy()
            n_pos_total = int(df_line["target"].sum())

            if len(df_line) == 0:
                skipped_rows.append({"temperature": temperature, "line_name": line_name, "reason": "No rows."})
                continue
            if n_pos_total < 2:
                skipped_rows.append({
                    "temperature": temperature,
                    "line_name":   line_name,
                    "reason":      f"Only {n_pos_total} positive sample(s).",
                })
                continue

            split_mode = (
                "loo"      if n_pos_total < MIN_POS_LOO_THRESHOLD else
                "adaptive" if n_pos_total < MIN_POS_STANDARD      else
                "standard"
            )
            print(f"\n  [{line_name}] n_pos={n_pos_total}  split={split_mode}  tau={temperature}")

            try:
                result = run_line_experiment(
                    df_line=df_line, line_name=line_name, line_id_encoded=line_id_encoded,
                    temperature=temperature, output_dir=temp_dir, split_mode=split_mode,
                )
                result["temperature"] = temperature
                summary_rows.append(result)
                global_summary.append(result)
            except Exception as e:
                skipped_rows.append({"temperature": temperature, "line_name": line_name, "reason": str(e)})
                print(f"  [ERROR] {line_name}: {e}")
                traceback.print_exc()

        pd.DataFrame(summary_rows).to_csv(os.path.join(temp_dir, "summary_per_line.csv"), index=False)
        pd.DataFrame(skipped_rows).to_csv(os.path.join(temp_dir, "skipped_lines.csv"),   index=False)

        if summary_rows:
            print(f"\n  Summary (tau={temperature}):")
            for row in summary_rows:
                print(
                    f"    {row.get('line_name', '?'):20s}  "
                    f"score={row.get('best_score') or 'N/A':>7}  "
                    f"f2={row.get('f2') or 'N/A':>6}  "
                    f"ovr={row.get('ovr') or 'N/A':>6}  "
                    f"fa={row.get('fa') or 'N/A':>6}"
                )

    pd.DataFrame(global_summary).to_csv(
        os.path.join(OUTPUT_DIR, "summary_all_temperatures.csv"), index=False,
    )
    print(f"\nDone. Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()