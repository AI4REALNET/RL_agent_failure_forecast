import os
from typing import Any, Dict, List, Optional, Tuple
from config import CFG, OUTPUT_DIR_LLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPISODE_SEED: int = 50
RESULTS_DIR = os.path.join(OUTPUT_DIR_LLM, "temp_0.8")

# ---------------------------------------------------------------------------
# Path setup —
# ---------------------------------------------------------------------------

import grid2op
from grid2op.Agent import CurriculumAgent

try:
    from lightsim2grid import LightSimBackend
    _BACKEND = LightSimBackend()
except ImportError:
    _BACKEND = None

from rule_predictor import RulePredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_load_models() -> Tuple[Any, Any, Any]:
    mp = ma = me = None
    try:
        import joblib
        if os.path.exists(CFG.MODEL_MEAN_PATH):
            mp = joblib.load(CFG.MODEL_MEAN_PATH)
        if os.path.exists(CFG.MODEL_ALEATORIC_PATH):
            ma = joblib.load(CFG.MODEL_ALEATORIC_PATH)
    except Exception:
        pass
    try:
        import torch
        from enn_models import ENN
        if os.path.exists(CFG.MODEL_ENN_PATH):
            me = ENN()
            me.load_state_dict(torch.load(CFG.MODEL_ENN_PATH, map_location="cpu"))
            me.eval()
    except Exception:
        pass
    return mp, ma, me


def _line_name_to_idx(env: Any, line_name: str) -> Optional[int]:
    clean = line_name.replace("line_", "")
    for i, name in enumerate(env.name_line):
        if str(name) == clean or str(name).replace(" ", "_") == clean:
            return i
    return None


def _simulate_contingency(obs: Any, line_idx: int, env: Any) -> bool:
    """Returns True if disconnecting the line causes a game over."""
    try:
        act = env.action_space({"set_line_status": [(line_idx, -1)]})
        _, _, done, info = obs.simulate(act)
        return done or info.get("is_illegal", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    predictor: RulePredictor,
    lines_to_monitor: List[str],
    line_indices: Dict[str, int],
    use_attack: bool,
) -> None:
    observations_array: List[Any] = []
    predictor.observations_array  = observations_array

    env.set_id(EPISODE_SEED)
    obs   = env.reset()
    agent = DoNothingAgent(env.action_space)
    done  = False
    step  = 0

    label = "WITH ATTACK" if use_attack else "NO ATTACK"
    print(f"\n{'═'*65}")
    print(f"  Scenario: {label}  |  seed={EPISODE_SEED}")
    print(f"{'═'*65}\n")

    # Track: for each rule alert, did the actual failure happen?
    # List of (step, line_name, actual_fail)
    alerts: List[Tuple[int, str, bool]] = []

    while not done:
        step += 1
        observations_array.append(obs)

        for line_name in lines_to_monitor:
            result    = predictor.predict(obs, line_name)
            predicted = result["prediction"]

            if predicted == 1:
                # Check what actually happens if the line is disconnected now.
                idx         = line_indices[line_name]
                actual_fail = _simulate_contingency(obs, idx, env)
                alerts.append((step, line_name, actual_fail))

                print(f"  Step {step:4d} | ️  ALERT [{line_name}]")
                print(f"  {result['sentence']}\n")

        # Attack: disconnect a monitored line every 30 steps.
        if use_attack and step % 30 == 0:
            attack_line = lines_to_monitor[step // 30 % len(lines_to_monitor)]
            idx         = line_indices.get(attack_line)
            action      = (env.action_space({"set_line_status": [(idx, -1)]})
                           if idx is not None else agent.act(obs, None, None))
            print(f"  Step {step:4d} |   ATTACK: line [{attack_line}] disconnected\n")
        else:
            action = agent.act(obs, None, None)

        obs, _, done, _ = env.step(action)

    # ── End-of-episode summary ───────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"  Episode finished after {step} steps.\n")

    if not alerts:
        print("  No alerts were triggered during this episode.\n")
        return

    print("  Alert summary:")
    print(f"  {'Step':>5}  {'Line':<20}  {'Predicted':<12}  {'Actual outcome'}")
    print(f"  {'─'*5}  {'─'*20}  {'─'*12}  {'─'*20}")
    correct = 0
    for s, ln, actual in alerts:
        actual_str = "FAILURE ✓" if actual else "OK (false alarm)"
        print(f"  {s:>5}  {ln:<20}  {'FAILURE':<12} {actual_str}")
        if actual:
            correct += 1

    total = len(alerts)
    print(f"\n  Correct alerts: {correct}/{total}  "
          f"({'%.0f' % (correct/total*100)}% of alerts corresponded to actual failures)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("═" * 65)
    print("  Grid2Op Rule Predictor — Episode Test")
    print("═" * 65)

    kwargs = {"backend": _BACKEND} if _BACKEND is not None else {}
    env    = grid2op.make(CFG.ENV_NAME, **kwargs)

    mp, ma, me = _try_load_models()

    predictor = RulePredictor(
        rules_dir=OUTPUT_DIR_LLM,
        model_predict=mp,
        model_aleatoric=ma,
        model_enn=me,
        observations_array=[],
    )

    if not predictor.available_lines:
        print(f"[ERROR] No rules found in '{OUTPUT_DIR_LLM}'.")
        return

    all_lines    = CFG.LINES_TO_TEST or predictor.available_lines
    line_indices = {ln: _line_name_to_idx(env, ln)
                    for ln in all_lines
                    if _line_name_to_idx(env, ln) is not None}
    lines        = list(line_indices.keys())

    print(f"\n[INFO] Environment : {CFG.ENV_NAME}")
    print(f"[INFO] Lines monitored: {lines}")

    print("\n--- Rule descriptions ---")
    for ln in lines:
        print(f"  [{ln}]\n  {predictor.sentence_only(ln)}\n")

    run_episode(env, predictor, lines, line_indices, use_attack=False)
    run_episode(env, predictor, lines, line_indices, use_attack=True)


if __name__ == "__main__":
    main()
