import time
from typing import List

import pandas as pd

import numpy as np
import random

from GlobalOptimum import globalOptimum
from GlobalOptimum.globalOptimum import GlobalOptimum
from ModelStructure.Flight.flight import Flight
from ModelStructure.Slot.slot import Slot
from ModelStructure.modelStructure import ModelStructure
from ScheduleMaker.real_schedule import RealisticSchedule, Regulation
from UDPP.udppModel import UDPPmodel


class IterativeUDPP(ModelStructure):

    def __init__(self, slot_list: List[Slot], flight_list: List[Flight]):
        super().__init__(slot_list, flight_list)
        self.iterations = 0

    @staticmethod
    def get_excluded_flights(exc_flights: List, model: globalOptimum):
        initial_exc_flights = len(exc_flights)
        for airline in model.airlines:
            if airline.initialCosts < airline.finalCosts:
                exc_flights += airline.flights
        neg_impact = True if len(exc_flights) > initial_exc_flights else False
        return exc_flights, neg_impact

    def run(self):
        t = time.time()
        global_model = GlobalOptimum(self.slots, self.flights)
        global_model.run()

        udpp_model = UDPPmodel(self.slots, self.flights, hfes=0)
        for i in range(self.numAirlines):
            if global_model.airlines[i].initialCosts < global_model.airlines[i].finalCosts:
                udpp_model.airlines[i].participating_in_UDPP = False
        udpp_model.run(optimised=True)

        self.time = time.time() - t
        self.solution = udpp_model.solution
        self.report = udpp_model.report
        self.airlines = udpp_model.airlines
        self.flights = udpp_model.flights


