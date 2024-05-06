import time

import pandas as pd

import numpy as np
import random

from GlobalOptimum.globalOptimum import GlobalOptimum
from NNBound.nnBound import NNBoundModel
import udpp_tests.make_sol_df as sol
from ScheduleMaker.real_schedule import RealisticSchedule, Regulation
from UDPP.udppModel import UDPPmodel

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

schedule_maker = RealisticSchedule()
seed = 0
np.random.seed(seed)
random.seed(seed)

HFES = 0

def run_test(df, n_runs, hfes):

    for i in range(n_runs):
        regulation: Regulation
        regulation = schedule_maker.get_regulation()
        print(i, regulation.airport)
        print(regulation.nFlights, regulation.cReduction, regulation.startTime)
        slot_list, fl_list = schedule_maker.make_sl_fl_from_data(airport=regulation.airport,
                                                                 n_flights=regulation.nFlights,
                                                                 capacity_reduction=regulation.cReduction,
                                                                 compute=True, regulation_time=regulation.startTime)

        global_model = GlobalOptimum(slot_list, fl_list)
        global_model.run()
        sol.append_to_df(global_model, "mincost")
        # global_model.print_performance()

        # print("n", regulation.nFlights, regulation.cReduction, i)
        max_model = NNBoundModel(slot_list, fl_list)
        max_model.run(verbose=False, time_limit=120, rescaling=False)
        sol.append_to_df(max_model, "nnbound")
        # max_model.print_performance()

        # print("u", regulation.nFlights, regulation.cReduction, i)
        udpp_model = UDPPmodel(slot_list, fl_list, hfes=0)
        udpp_model.run(optimised=True)
        # udpp_model.print_performance()
        sol.append_to_df(udpp_model, "udpp")

        udpp_model_5 = UDPPmodel(slot_list, fl_list, hfes=5)
        udpp_model_5.run(optimised=True)
        # udpp_model_5.print_performance()
        sol.append_to_df(udpp_model_5, "udpp_5")

        model_list = [global_model, max_model, udpp_model, udpp_model_5]
        df = sol.append_results(df, model_list, i, regulation.nFlights, regulation.cReduction,
                                regulation.airport, regulation.startTime, print_df= False)

    return df


# in case remember both

# scheduleType = schedule_types(show=True)


df_test = pd.DataFrame(
    columns=["airline", "num flights", "initial costs", "final costs", "reduction %", "run", "n_flights", "c_reduction",
             "model", "init_mis_con", "final_mis_con", "init_curfew", "final_curfew"])

df_test = run_test(n_runs=1000, df=df_test, hfes=0)
df_test.to_csv("udpp_tests/cap_n_fl_test_1000_hfes_curfew.csv", index_label=False, index=False)