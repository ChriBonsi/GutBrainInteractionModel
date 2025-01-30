from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.parameters import params


class Neuron(core.Agent):
    TYPE = 1

    def __init__(self, local_id: int, rank: int, initial_state, pt):
        super().__init__(id=local_id, type=Neuron.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.toRemove = False

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates, self.toRemove

    def step(self):
        difference_pro_anti_cytokine = model.pro_cytokine - model.anti_cytokine
        if difference_pro_anti_cytokine > 0:
            level_of_inflammation = (difference_pro_anti_cytokine * 100) / (model.pro_cytokine + model.anti_cytokine)
            if np.random.randint(0, 100) < level_of_inflammation:
                self.change_state()
        else:
            pass

    def change_state(self):
        if self.state == params["neuron_state"]["healthy"]:
            self.state = params["neuron_state"]["damaged"]
        elif self.state == params["neuron_state"]["damaged"]:
            self.state = params["neuron_state"]["dead"]
            self.toRemove = True
            model.dead_neuron += 1
