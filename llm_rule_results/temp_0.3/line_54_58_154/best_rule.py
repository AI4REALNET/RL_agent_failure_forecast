def rule(x):
    if x["max_line_rho"] >= 0.85:
        if x["epistemic_before"] >= 0.8134:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_q"] >= 150.4021:
            if x["aleatoric_gen_p_mean"] <= 0.2781:
                return 1
            else:
                return 0
        else:
            return 0