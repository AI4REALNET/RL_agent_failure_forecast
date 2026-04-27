def rule(x):
    if x["max_line_rho"] >= 0.85:
        if x["epistemic_before"] >= 0.8141:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 639.207:
            if x["aleatoric_gen_p_mean"] <= 0.1262:
                return 1
            else:
                return 0
        else:
            return 0