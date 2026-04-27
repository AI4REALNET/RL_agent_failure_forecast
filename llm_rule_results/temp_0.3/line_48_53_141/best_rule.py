def rule(x):
    if x["max_line_rho"] >= 0.70:
        if x["epistemic_before"] >= 0.7884:
            return 1
        else:
            return 0
    else:
        if x["fcast_max_line_rho"] >= 0.75:
            if x["aleatoric_gen_p_mean"] <= 0.4296:
                return 1
            else:
                return 0
        else:
            return 0