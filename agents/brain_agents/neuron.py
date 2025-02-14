from typing import Tuple
import numpy as np
from repast4py.space import DiscretePoint as dpt
from repast4py import core

# from repast4py.parameters import params
# from gut_brain_system import model


class Neuron(core.Agent):
    TYPE = 7

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context: str):
        super().__init__(id=local_id, type=Neuron.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.toRemove = False
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.state, self.pt.coordinates, self.toRemove, self.context

    # Neuron step function
    def step(self, model):
        difference_pro_anti_cytokine = model.pro_cytokine - model.anti_cytokine
        if difference_pro_anti_cytokine > 0:
            level_of_inflammation = (difference_pro_anti_cytokine * 100) / (model.pro_cytokine + model.anti_cytokine)
            if np.random.randint(0, 100) < level_of_inflammation:
                self.change_state(model)
        else:
            pass

    # changes the state of the neuron agent
    def change_state(self, model):
        if self.state == model.params["neuron_state"]["healthy"]:
            self.state = model.params["neuron_state"]["damaged"]
        elif self.state == model.params["neuron_state"]["damaged"]:
            self.state = model.params["neuron_state"]["dead"]
            self.toRemove = True
            model.dead_neuron += 1