def rule(x):
    if x["sum_gen_p"] >= 488.74896:
        if x["epistemic_after"] <= 0.819:
            return 1
        else:
            return 0
    else:
        return 0