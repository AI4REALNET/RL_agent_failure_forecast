def rule(x):
    if x["sum_gen_p"] >= 495.0:
        if x["epistemic_after"] <= 0.819:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_gen_p"] >= 610.0114:
            return 1
        else:
            return 0