import pandas as pd

def append_to_df(model, name, print_performance=False):

    if name not in ["udpp", "udpp_5"]:
        model.report["model"] = name
        model.report["comp_time"] = model.time
        model.report["protections"] = 0
        model.report["positive"] = 0
        model.report["positive_min"] = 0
        model.report["negative"] = 0
        model.report["negative_min"] = 0

    else:
        model.report["model"] = name
        model.report["comp_time"] = [model.computationalTime] + \
                                            [airline.udppComputationalTime for airline in model.airlines]

        protections = [airline.protections for airline in model.airlines]
        model.report["protections"] = [sum(protections)] + protections

        positive = [airline.positiveImpact for airline in model.airlines]
        positiveMins = [airline.positiveImpactMinutes for airline in model.airlines]

        model.report["positive"] = [sum(positive)] + positive
        model.report["positive_min"] = [sum(positiveMins)] + positiveMins

        negative = [airline.negativeImpact for airline in model.airlines]
        negativeMins = [airline.negativeImpactMinutes for airline in model.airlines]

        model.report["negative"] = [sum(negative)] + negative
        model.report["negative_min"] = [sum(negativeMins)] + negativeMins

    if print_performance:
        model.print_performance()


def append_results(df, models_list, i, n_flights, c_reduction, airport, start_time,
                   print_df=False):
    df_run = models_list[0].report
    for model in models_list[1:]:
        df_run = pd.concat([df_run, model.report], ignore_index=True)
    df_run["run"] = i
    df_run["n_flights"] = n_flights
    df_run["c_reduction"] = c_reduction
    df_run["airport"] = airport
    df_run["time"] = start_time
    if print_df:
        print(df_run, "\n")

    return pd.concat([df, df_run], ignore_index=True)
