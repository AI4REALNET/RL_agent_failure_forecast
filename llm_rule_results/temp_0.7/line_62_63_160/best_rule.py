def rule(x):
    if x["max_line_rho"] >= 0.75:
        if x["aleatoric_gen_p_mean"] <= 0.3335:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 134.3694:
            if x["epistemic_before"] <= 0.8068:
                return 1
            else:
                return 0
        else:
            return 0