from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt

# from repast4py.parameters import params
# from gut_brain_system import model


class Oligomer(core.Agent):
    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name, pt: dpt, context):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = oligomer_name
        self.pt = pt
        self.toRemove = False
        self.toMove = False
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.name, self.pt.coordinates, self.toRemove, self.context

    # Oligomer step function
    def step(self, model):
        if self.pt is None:
            return
        else:
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt, self.context)
            if len(nghs_coords) <= 6 and self.context == 'gut':
                if model.barrier_impermeability < model.params["barrier_impermeability"]:
                    percentage_threshold = int((model.barrier_impermeability * model.params["barrier_impermeability"]) / 100)
                    choice = np.random.randint(0, 100)
                    if choice > percentage_threshold:
                        self.toMove = True