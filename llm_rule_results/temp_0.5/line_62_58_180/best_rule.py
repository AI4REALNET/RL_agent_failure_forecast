def rule(x):
    if x["max_line_rho"] >= 0.75:
        if x["epistemic_before"] >= 0.787:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 590.9352:
            if x["sum_load_q"] >= 134.42:
                return 1
            else:
                return 0
        else:
            return 0