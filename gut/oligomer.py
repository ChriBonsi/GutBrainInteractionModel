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

class Oligomer(core.Agent):

    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name, pt: dpt):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = oligomer_name
        self.pt = pt
        self.toRemove = False

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates)

    def step(self):
        if self.pt is None:
            return
        else:
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt)
