def rule(x):
    if x["max_line_rho"] >= 0.75:
        if x["epistemic_before"] >= 0.78:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= 0.75:
            return 1
        else:
            return 0