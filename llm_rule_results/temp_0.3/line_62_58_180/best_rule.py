def rule(x):
    if x["sum_load_q"] >= 134.42:
        if x["aleatoric_gen_p_mean"] <= 0.3327:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 134.5603:
            if x["aleatoric_gen_p_mean"] <= 0.3327:
                return 1
            else:
                return 0
        else:
            return 0