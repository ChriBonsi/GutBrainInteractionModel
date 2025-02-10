from typing import Tuple

from repast4py import core
from repast4py.parameters import params
from repast4py.space import DiscretePoint as dpt

from brain.oligomer import Oligomer


class Microglia(core.Agent):
    TYPE = 0

    # params["agent_state"]["resting"] state of an agent that is resting
    def __init__(self, local_id: int, rank: int, initial_state, pt):
        super().__init__(id=local_id, type=Microglia.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt

    def setAgentData(self, newState):
        """
        Function created as a stopgag due to the inconsistency of the 'state' parameter between agent classes.
        This function updates the parameter of the class related to the value 'agent_data[1]'.
        The modified parameter for 'Microglia' class is 'state'.
        """
        self.state = newState

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates

    def step(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        ngh = self.check_oligomer_nghs(nghs_coords)
        if ngh is not None:
            if self.state == params["microglia_state"]["resting"]:
                self.state = params["microglia_state"]["active"]
            else:
                ngh.toRemove = True

    def check_oligomer_nghs(self, nghs_coords):
        for ngh_coord in nghs_coords:
            ngh_array = model.grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == Oligomer:
                    return ngh
        return None
