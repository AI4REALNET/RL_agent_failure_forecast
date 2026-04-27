def rule(x):
    if x["max_line_rho"] >= 0.638:
        if x["aleatoric_gen_p_mean"] <= 0.2842:
            if x["sum_load_q"] >= 144.15:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 599.9462:
            if x["fcast_max_line_rho"] >= 0.95:
                return 1
            else:
                return 0
        else:
            return 0