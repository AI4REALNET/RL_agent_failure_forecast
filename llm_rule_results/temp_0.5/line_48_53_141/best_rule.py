def rule(x):
    if x["max_line_rho"] >= 0.75:
        if x["epistemic_before"] >= 0.7884:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 590.9994:
            if x["fcast_max_line_rho"] >= 0.65:
                return 1
            else:
                return 0
        else:
            return 0