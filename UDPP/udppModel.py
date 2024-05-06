import copy
import string
from typing import Union, Callable, List

import pandas as pd

from GlobalFuns.globalFuns import HiddenPrints
from ModelStructure.Airline.airline import Airline
from ModelStructure.modelStructure import ModelStructure
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt

from UDPP.udppMerge import udpp_merge
from ModelStructure.Solution import solution
from UDPP.UDPPflight.udppFlight import UDPPflight
from ModelStructure.Slot.slot import Slot
from ModelStructure.Flight import flight as fl
import time

import ModelStructure.modelStructure as ms
from UDPP.Local import local

class UDPPmodel(ModelStructure):

    def __init__(self, slot_list: List[Slot], flights: List[UDPPflight], hfes=0):

        udpp_flights = [UDPPflight(flight) for flight in flights if flight is not None]
        self.hfes = hfes
        self.computationalTime = None
        super().__init__(slot_list, udpp_flights, air_ctor=Airline)

    def run(self, optimised=True, xp=False):
        airline: Airline
        start = time.time()
        for airline in self.airlines:
            airline.initialCosts = self.compute_costs(airline.flights, "initial")
            if airline.numFlights > 1:
                if optimised:
                    # with HiddenPrints():
                    local_time = time.time()

                    UDPPlocalOpt(airline, self.slots)
                    airline.udppComputationalTime = time.time() - local_time
                else:
                    local.udpp_local(airline, self.slots)
            else:
                airline.flights[0].newSlot = airline.flights[0].slot

            for flight in airline.flights:
                flight.localTime = flight.newSlot.time


        udpp_merge(self.flights, self.slots, self.hfes)

        self.computationalTime = time.time() - start
        # print(time.time() - start)
        self.update_missed_connecting()
        self.update_hitting_curfew()
        solution.make_solution(self)
        for airline in self.airlines:
            # airline.finalCosts = self.compute_costs(airline.flights, "final")
            # if airline.initialCosts < airline.finalCosts:
            for flight in airline.flights:
                if flight.localTime < flight.newSlot.time:
                    airline.negativeImpact += 1
                    airline.negativeImpactMinutes += flight.newSlot.time - flight.localTime

                if flight.localTime > flight.newSlot.time:
                    airline.positiveImpact += 1
                    airline.positiveImpactMinutes += flight.localTime - flight.newSlot.time

    def compute_optimal_prioritisation(self):
        airline: Airline
        for airline in self.airlines:
            if airline.numFlights > 1:
                with HiddenPrints():
                    UDPPlocalOpt(airline, self.slots)
            else:
                airline.flights[0].udppPriority = "N"
                airline.flights[0].udppPriorityNumber = 0



