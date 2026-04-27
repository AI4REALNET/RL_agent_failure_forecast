def rule(x):
    if x["sum_load_q"] >= 100.868:
        if x["epistemic_before"] <= 0.97308:
            return 1
        else:
            return 0
    else:
        return 0