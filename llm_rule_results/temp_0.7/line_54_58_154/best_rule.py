def rule(x):
    if x["fcast_sum_load_p"] <= 659.43075:
        if x["fcast_sum_gen_p"] <= 650.8097:
            return 1
        else:
            return 0
    else:
        return 0