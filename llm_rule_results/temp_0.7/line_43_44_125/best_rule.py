def rule(x):
    if x["sum_load_p"] <= 648.95:
        if x["epistemic_after"] <= 0.8400:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= 0.72:
            if x["epistemic_after"] <= 0.8400:
                return 1
            else:
                return 0
        else:
            return 0