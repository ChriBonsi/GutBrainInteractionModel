from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


class CleavedProtein(core.Agent):
    TYPE = 2

    def __init__(self, local_id: int, rank: int, name, pt: dpt):
        super().__init__(id=local_id, type=CleavedProtein.TYPE, rank=rank)
        self.name = name
        self.pt = pt
        self.alreadyAggregate = False
        self.toRemove = False
        self.toAggregate = False

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates, self.toAggregate, self.alreadyAggregate, self.toRemove)

    def step(self):
        if self.alreadyAggregate == True or self.toAggregate == True or self.pt is None:
            pass
        else:
            cleaved_nghs_number, _, nghs_coords = self.check_and_get_nghs()
            if cleaved_nghs_number == 0:
                random_index = np.random.randint(0, len(nghs_coords))
                model.move(self, nghs_coords[random_index])
            elif cleaved_nghs_number >= 4:
                self.change_state()
            else:
                self.change_group_aggregate_status()

    def change_group_aggregate_status(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        for ngh_coords in nghs_coords:
            nghs_array = model.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if ngh is not None:
                    ngh.alreadyAggregate = False

    def is_valid(self):
        cont = 0
        _, nghs_cleaved, _ = self.check_and_get_nghs()
        for agent in nghs_cleaved:
            if (agent.alreadyAggregate == True):
                cont += 1
        if cont >= 4:
            return True
        else:
            return False

    def change_state(self):
        if self.toAggregate == False:
            self.toAggregate = True

    def check_and_get_nghs(self):
        nghs_coords = model.ngh_finder.find(self.pt.x, self.pt.y)
        cont = 0
        cleavedProteins = []
        for ngh_coords in nghs_coords:
            ngh_array = model.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in ngh_array:
                if (type(ngh) == CleavedProtein and self.name == ngh.name):
                    cleavedProteins.append(ngh)
                    if ngh.toAggregate == False and ngh.alreadyAggregate == False:
                        ngh.alreadyAggregate = True
                        cont += 1
        return cont, cleavedProteins, nghs_coords
