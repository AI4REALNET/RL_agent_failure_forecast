def rule(x):
    if x["max_line_rho"] >= 0.62:
        if x["aleatoric_gen_p_mean"] <= 0.1850:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 600.7665:
            if x["fcast_sum_load_q"] >= 155.5429:
                return 1
            else:
                return 0
        else:
            return 0