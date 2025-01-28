from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
import repast4py.random
from repast4py.space import DiscretePoint as dpt
import yaml
import numba
from numba import int32, int64
from numba.experimental import jitclass
import math

class ExternalInput(core.Agent):

    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=ExternalInput.TYPE, rank=rank)
        possible_types = [params["external_input"]["diet"],params["external_input"]["antibiotics"],params["external_input"]["stress"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.input_name = input_name
        self.pt = pt

    def save(self) -> Tuple:
        return (self.uid, self.input_name, self.pt.coordinates)

    #if the external input is "diet" or "stress" then the microbiota bacteria decrease in good bacteria classes and increase in pathogenic ones.
    #otherwise it only decreases the good bacteria classes.
    #random percentage to change the params of the microbiota
    def step(self):
        if model.barrier_impermeability >= model.barrier_permeability_threshold_stop:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_remove = int((model.microbiota_good_bacteria_class * np.random.uniform(0, good_bacteria_factor)) / 100)
                model.microbiota_good_bacteria_class -= to_remove
                to_add = int((params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)
                model.microbiota_pathogenic_bacteria_class += to_add

            if self.input_name == params["external_input"]["diet"]:
                adjust_bacteria(3, 3)
            elif self.input_name == params["external_input"]["antibiotics"]:
                adjust_bacteria(5, 2)
            else:
                adjust_bacteria(3, 3)
