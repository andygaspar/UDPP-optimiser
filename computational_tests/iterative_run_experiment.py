import time

import pandas as pd

import numpy as np
import random

from GlobalOptimum.globalOptimum import GlobalOptimum
from Iterative.iterative_UDPP import IterativeUDPP
from Iterative.iterative_global import IterativeGlobal
from NNBound.nnBound import NNBoundModel
import computational_tests.make_sol_df as sol
from ScheduleMaker.real_schedule import RealisticSchedule, Regulation
from UDPP.udppModel import UDPPmodel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def run_test(df, n_runs, sched_maker):
    for i in range(n_runs):
        regulation: Regulation
        regulation = sched_maker.get_regulation()
        print(i, regulation.airport)
        print(regulation.nFlights, regulation.cReduction, regulation.startTime)
        slot_list, fl_list = sched_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                              n_flights=regulation.nFlights,
                                                              capacity_reduction=regulation.cReduction,
                                                              compute=True, regulation_time=regulation.startTime)

        global_model = GlobalOptimum(slot_list, fl_list)
        global_model.run()
        sol.append_to_df(global_model, "mincost")
        # global_model.print_performance()

        max_model = NNBoundModel(slot_list, fl_list)
        max_model.run(verbose=False, time_limit=120, rescaling=False)
        sol.append_to_df(max_model, "nnbound")
        # max_model.print_performance()

        iterative_model = IterativeGlobal(slot_list, fl_list)
        iterative_model.run()
        # iterative_model.print_performance()
        sol.append_to_df(iterative_model, "iter_global")

        udpp_model = UDPPmodel(slot_list, fl_list, hfes=0)
        udpp_model.run(optimised=True)
        # udpp_model.print_performance()
        sol.append_to_df(udpp_model, "udpp")

        iterative_udpp = IterativeUDPP(slot_list, fl_list)
        iterative_udpp.run()
        # iterative_udpp.print_performance()
        sol.append_to_df(iterative_udpp, "iter_udpp")

        model_list = [global_model, max_model, iterative_model, udpp_model, iterative_udpp]
        df = sol.append_results(df, model_list, i, regulation.nFlights, regulation.cReduction,
                                regulation.airport, regulation.startTime, print_df=True)

    return df


# in case remember both

# scheduleType = schedule_types(show=True)


df_test = pd.DataFrame(
    columns=["airline", "num flights", "initial costs", "final costs", "reduction %", "run", "n_flights", "c_reduction",
             "model", "init_mis_con", "final_mis_con", "init_curfew", "final_curfew"])

schedule_maker = RealisticSchedule()
seed = 0
np.random.seed(seed)
random.seed(seed)

df_test = run_test(n_runs=100, df=df_test, sched_maker=schedule_maker)
df_test.to_csv("Results/iterative_test_results.csv", index_label=False, index=False)
