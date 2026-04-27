def rule(x):
    if x["max_line_rho"] >= 0.65:
        if x["fcast_nb_rho_ge_0.95"] >= 1.0 or x["epistemic_after"] <= 0.8300:
            return 1
        else:
            return 0
    else:
        if x["fcast_sum_load_p"] <= 593.7134:
            if x["epistemic_after"] <= 0.8004:
                return 1
            else:
                return 0
        else:
            return 0