def rule(x):
    if x["sum_load_p"] <= 595.94:
        if x["epistemic_before"] >= 0.7906:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= 0.75:
            return 1
        else:
            return 0