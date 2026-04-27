def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["aleatoric_gen_p_mean"] <= 0.2768:
            if x["sum_load_q"] >= 158.39:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 155.5429:
            if x["epistemic_after"] >= 0.8331:
                return 1
            else:
                return 0
        else:
            return 0