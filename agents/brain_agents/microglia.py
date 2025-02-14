from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt

# from repast4py.parameters import params
# from gut_brain_system import model
from agents.gut_brain_agents.oligomer import Oligomer


class Microglia(core.Agent):
    TYPE = 6

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context):
        super().__init__(id=local_id, type=Microglia.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates, self.context

    # Microglia step function
    def step(self, model):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        ngh = self.check_oligomer_nghs(nghs_coords, model)
        if ngh is not None:
            if self.state == model.params["microglia_state"]["resting"]:
                self.state = model.params["microglia_state"]["active"]
            else:
                ngh.toRemove = True

    # returns the oligomer agent in the neighborhood of the agent     
    def check_oligomer_nghs(self, nghs_coords, model):
        for ngh_coord in nghs_coords:
            ngh_array = model.brain_grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == Oligomer:
                    return ngh
        return None