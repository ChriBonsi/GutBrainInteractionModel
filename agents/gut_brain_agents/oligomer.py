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

        self.state = self.return_state(self) #stopgag parameter

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

    def return_state(self):
        """This method was added as a stopgap fix for a structural issue."""
        return self.name
    
    def setAgentData(self, newName):
        """
        Function created as a stopgag due to the inconsistency of the 'state' parameter between agent classes.
        This function updates the parameter of the class related to the value 'agent_data[1]'.
        The modified parameter for 'Oligomer' class is 'name'.
        """
        self.name = newName