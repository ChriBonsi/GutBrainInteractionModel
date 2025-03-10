from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


class Protein(core.Agent):
    TYPE = 1

    def __init__(self, local_id: int, rank: int, protein_name, pt: dpt, context):
        super().__init__(id=local_id, type=Protein.TYPE, rank=rank)
        self.name = protein_name
        self.pt = pt
        self.toCleave = False
        self.toRemove = False
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.name, self.pt.coordinates, self.toCleave, self.toRemove, self.context

    # Protein step function
    def step(self, model):
        if self.pt is None:
            return
        else:
            nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            model.move(self, chosen_dpt, self.context)

    # changes the state of the protein agent
    def change_state(self):
        if not self.toCleave:
            self.toCleave = True
