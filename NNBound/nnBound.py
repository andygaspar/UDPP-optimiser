from typing import Callable, List, Union

from ModelStructure import modelStructure as mS
from gurobipy import Model, GRB, quicksum, Env

from ModelStructure.Airline import airline as air
from ModelStructure.Flight.flight import Flight
from ModelStructure.Solution import solution
from ModelStructure.Slot.slot import Slot

import numpy as np
import pandas as pd

import time


def stop(model, where):

    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        run_time = model.cbGet(GRB.Callback.RUNTIME)

        if run_time > model._time_limit and abs(objbst - objbnd) < 0.005 * abs(objbst):
            print("stop at", run_time)
            model.terminate()


class NNBoundModel(mS.ModelStructure):

    def __init__(self, slot_list: List[Slot], flight_list: List[Flight]):

        super().__init__(slot_list, flight_list)

        self.m = Model()
        # self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
        self.m.modelSense = GRB.MINIMIZE
        max_cost = max([flight.maxCost for flight in flight_list])
        self.rescaling = 1000 / max_cost if max_cost > 0 else 1
        self.x = None
        self.time = None

    def set_variables(self):
        flight: Flight
        airline: air.Airline
        self.x = self.m.addVars([(i, j) for i in range(self.numFlights) for j in range(self.numFlights)],
                                vtype=GRB.BINARY)

    def set_constraints(self):
        flight: Flight
        airline: air.Airline
        for flight in self.flights:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for slot in flight.compatibleSlots) == 1
            )

        for slot in self.slots:
            self.m.addConstr(
                quicksum(self.x[flight.index, slot.index] for flight in self.flights) <= 1
            )

        for airline in self.airlines:
            self.m.addConstr(
                quicksum(flight.cost_fun(flight.slot) * self.rescaling for flight in airline.flights) >= \
                quicksum(self.x[flight.index, slot.index] * flight.cost_fun(slot) * self.rescaling
                       for flight in airline.flights for slot in self.slots)
            )

    def set_objective(self):
        flight: Flight
        self.m.setObjective(
            quicksum(self.x[flight.index, slot.index] * flight.cost_fun(slot)*self.rescaling
                   for flight in self.flights for slot in self.slots)
        )

    def run(self, timing=False, verbose=False, time_limit=60, rescaling=True):

        if not rescaling:
            self.rescaling = 1

        self.m._time_limit = time_limit
        if not verbose:
            self.m.setParam('OutputFlag', 0)

        start = time.time()
        self.set_variables()
        self.set_constraints()
        if timing:
            print("Variables and constraints setting time ", time.time() - start)

        self.set_objective()

        self.m.optimize(stop)
        # self.m.printStats()
        self.time = time.time() - start
        if timing:
            print("Simplex time ", self.time)

        self.assign_flights(self.x)
        self.update_missed_connecting()
        self.update_hitting_curfew()
        solution.make_solution(self)


    def assign_flights(self, sol):
        for flight in self.flights:
            for slot in self.slots:
                if sol[flight.index, slot.index].x > 0.5:
                    flight.newSlot = slot

    def get_sol_array(self):
        solution = np.zeros((len(self.flights), len(self.slots)))
        for flight in self.flights:
            for slot in self.slots:
                if self.x[flight.index, slot.index].x > 0.5:
                    solution[flight.index, slot.index] = 1
        return solution

