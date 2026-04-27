def rule(x):
    if x["max_line_rho"] >= 0.85:
        if x["aleatoric_gen_p_mean"] <= 0.2842:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 639.8191:
            if x["fcast_sum_load_q"] >= 141.37:
                return 1
            else:
                return 0
        else:
            return 0