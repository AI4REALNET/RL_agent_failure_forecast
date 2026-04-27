def rule(x):
    if x["max_line_rho"] >= 0.80:
        if x["aleatoric_gen_p_mean"] <= 0.8639:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 590.9994:
            if x["fcast_max_line_rho"] >= 0.70:
                return 1
            else:
                return 0
        else:
            return 0