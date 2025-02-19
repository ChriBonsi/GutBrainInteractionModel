from typing import Tuple
import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt

# from repast4py.parameters import params
# from gut_brain_system import model

from agents.gut_agents.normal_protein import Protein

class AEP(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, pt: dpt, context, model):
        super().__init__(id=local_id, type=AEP.TYPE, rank=rank)
        self.state = model.params["aep_state"]["active"]
        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates, self.context

    # returns True if the agent is hyperactive, False otherwise
    def is_hyperactive(self, model):
        if self.state == model.params["aep_state"]["active"]:
            return False
        else:
            return True

    # AEP step function   
    def step(self, model):
        if self.pt is None:
            return
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        protein = self.percepts(nghs_coords, model)
        if protein is not None:
            if self.is_hyperactive(model):
                self.cleave(protein)
        else:
            random_index = np.random.randint(0, len(nghs_coords))
            model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)

    # returns the protein agent in the neighborhood of the agent
    def percepts(self, nghs_coords, model):
        for ngh_coords in nghs_coords:
            nghs_array = model.gut_grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Protein:
                    return ngh
        return None

        # cleaves the protein agent

    def cleave(self, protein):
        protein.change_state()