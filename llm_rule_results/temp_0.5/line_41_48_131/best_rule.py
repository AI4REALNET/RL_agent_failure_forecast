def rule(x):
    if x["max_line_rho"] >= 0.70:
        if x["epistemic_before"] >= 0.7906:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= 0.75:
            if x["aleatoric_gen_p_mean"] <= 0.3379:
                return 1
            else:
                return 0
        else:
            return 0