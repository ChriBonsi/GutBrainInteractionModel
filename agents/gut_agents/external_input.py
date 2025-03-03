from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


class ExternalInput(core.Agent):
    TYPE = 4

    def __init__(self, local_id: int, rank: int, pt: dpt, context, model):
        super().__init__(id=local_id, type=ExternalInput.TYPE, rank=rank)
        possible_types = [model.params["external_input"]["diet"], model.params["external_input"]["antibiotics"],
                          model.params["external_input"]["stress"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.input_name = input_name
        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.input_name, self.pt.coordinates, self.context

    # External input step function
    def step(self, model):
        if model.barrier_impermeability >= model.barrier_permeability_threshold_stop:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_remove = int(
                    (model.microbiota_good_bacteria_class * np.random.uniform(0, good_bacteria_factor)) / 100)
                model.microbiota_good_bacteria_class -= to_remove
                to_add = int((model.params["microbiota_pathogenic_bacteria_class"] * np.random.uniform(0,
                                                                                                       pathogenic_bacteria_factor)) / 100)
                model.microbiota_pathogenic_bacteria_class += to_add

            if self.input_name == model.params["external_input"]["diet"]:
                adjust_bacteria(3, 3)
            elif self.input_name == model.params["external_input"]["antibiotics"]:
                adjust_bacteria(5, 2)
            else:
                adjust_bacteria(3, 3)
