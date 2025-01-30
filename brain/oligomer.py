from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


class Oligomer(core.Agent):
    TYPE = 3

    def __init__(self, local_id: int, rank: int, name, pt):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = name
        self.pt = pt
        self.toRemove = False

    def save(self) -> Tuple:
        return self.uid, self.name, self.pt.coordinates, self.toRemove

    def step(self):
        if self.pt is None:
            return
        else:
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt)
