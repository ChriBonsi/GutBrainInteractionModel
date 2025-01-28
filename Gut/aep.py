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

class AEP(core.Agent):

    TYPE = 0

    def __init__(self, local_id: int, rank: int, pt: dpt):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.state = params["aep_state"]["active"]
        self.pt = pt

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates)
    
    def is_hyperactive(self):
        if self.state == params["aep_state"]["active"]:
            return False
        else:
            return True
        
    def step(self):
        if self.pt is None:
            return
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        protein = self.percepts(nghs_coords)
        if protein is not None:
            if(self.is_hyperactive() == True):
                self.cleave(protein)
        else: 
            random_index = np.random.randint(0, len(nghs_coords))
            model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]))

    def percepts(self, nghs_coords):
        for ngh_coords in nghs_coords:
            nghs_array = model.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Protein:
                    return ngh  
        return None 
    
    def cleave(self, protein):
        protein.change_state()