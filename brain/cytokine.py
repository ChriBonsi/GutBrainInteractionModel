from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.parameters import params
from repast4py.space import DiscretePoint as dpt

from brain.microglia import Microglia


class Cytokine(core.Agent):
    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt):
        super().__init__(id=local_id, type=Cytokine.TYPE, rank=rank)
        self.pt = pt
        possible_types = [params["cyto_state"]["pro_inflammatory"], params["cyto_state"]["non_inflammatory"]]
        random_index = np.random.randint(0, len(possible_types))
        self.state = possible_types[random_index]
        if self.state == params["cyto_state"]["pro_inflammatory"]:
            model.pro_cytokine += 1
        else:
            model.anti_cytokine += 1

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates

    def step(self):
        if self.pt is None:
            return
        microglie_nghs, nghs_coords = self.get_microglie_nghs()
        if len(microglie_nghs) == 0:
            random_index = np.random.randint(0, len(nghs_coords))
            model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]))
        else:
            ngh_microglia = microglie_nghs[0]
            if self.state == params["cyto_state"]["pro_inflammatory"] and ngh_microglia.state == \
                    params["microglia_state"]["resting"]:
                ngh_microglia.state = params["microglia_state"]["active"]
            elif self.state == params["cyto_state"]["non_inflammatory"] and ngh_microglia.state == \
                    params["microglia_state"]["active"]:
                ngh_microglia.state = params["microglia_state"]["resting"]

    def get_microglie_nghs(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        microglie = []
        for ngh_coords in nghs_coords:
            nghs_array = model.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Microglia:
                    microglie.append(ngh)
        return microglie, nghs_coords
