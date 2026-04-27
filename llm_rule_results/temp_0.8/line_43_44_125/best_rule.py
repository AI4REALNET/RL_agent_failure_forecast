def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["epistemic_after"] <= 0.8124:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 593.7134:
            if x["fcast_max_line_rho"] >= 0.68:
                return 1
            else:
                return 0
        else:
            return 0