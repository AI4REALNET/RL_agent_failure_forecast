def rule(x):
    if x["max_line_rho"] >= 0.70:
        if x["epistemic_after"] <= 0.8124:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_gen_p"] <= 644.0342:
            if x["epistemic_after"] <= 0.8124:
                return 1
            else:
                return 0
        else:
            return 0