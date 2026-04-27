def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["aleatoric_gen_p_mean"] <= 0.2450:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 141.37:
            if x["fcast_max_line_rho"] >= 0.90:
                return 1
            else:
                return 0
        else:
            return 0