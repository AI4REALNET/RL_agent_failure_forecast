def rule(x):
    if x["max_line_rho"] >= 0.85:
        if x["epistemic_before"] >= 0.83:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 155.5429:
            if x["aleatoric_gen_p_mean"] <= 0.1500:
                return 1
            else:
                return 0
        else:
            return 0