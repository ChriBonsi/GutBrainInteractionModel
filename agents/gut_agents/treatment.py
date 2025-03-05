from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


class Treatment(core.Agent):
    TYPE = 5

    def __init__(self, local_id: int, rank: int, pt: dpt, context, model):
        super().__init__(id=local_id, type=Treatment.TYPE, rank=rank)
        possible_types = list(model.params["treatment_input"].keys())
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]

        self.pt = pt
        self.input_name = input_name
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.input_name, self.pt.coordinates, self.context

    # Treatment step function
    def step(self, model):
        if model.barrier_impermeability < model.barrier_permeability_threshold_start:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_add = int(
                    (model.params["microbiota_good_bacteria_class"] * np.random.uniform(0, good_bacteria_factor)) / 100)
                model.microbiota_good_bacteria_class += to_add

                to_remove = int((model.microbiota_pathogenic_bacteria_class * np.random.uniform(0,
                                                                                                pathogenic_bacteria_factor)) / 100)
                model.microbiota_pathogenic_bacteria_class -= to_remove

            good_bact = model.params["treatment_input"][self.input_name][0]
            pathogen_bact = model.params["treatment_input"][self.input_name][1]
            adjust_bacteria(good_bact, pathogen_bact)
