def rule(x):
    if x["max_line_rho"] >= 0.60:
        if x["aleatoric_gen_p_mean"] <= 0.674:
            if x["fcast_max_line_rho"] >= 0.70:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 644.5042:
            if x["epistemic_before"] >= 0.7871:
                return 1
            else:
                return 0
        else:
            return 0