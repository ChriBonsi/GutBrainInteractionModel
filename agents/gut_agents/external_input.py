from typing import Tuple

import numpy as np
from repast4py import core
from repast4py.space import DiscretePoint as dpt


def adjust_bacteria(model, good_bacteria_factor, pathogenic_bacteria_factor):
    good_bacteria_change = int((model.good_bact_count * np.random.uniform(0, good_bacteria_factor)) / 100)
    pathogenic_bacteria_change = int(
        (model.pathogenic_bact_count * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)

    model.good_bact_count += good_bacteria_change
    model.pathogenic_bact_count += pathogenic_bacteria_change


class ExternalInput(core.Agent):
    TYPE = 4
    _counter = 0  # Class variable to track selection order

    def __init__(self, local_id: int, rank: int, pt: dpt, context, model):
        super().__init__(id=local_id, type=ExternalInput.TYPE, rank=rank)
        # cartella healthy: 5 healthy_diet, 1 unhealthy_diet
        # cartella unhealthy: 5 unhealthy_diet, 1 healthy_diet
        # cartella baseline: 3 e 3
        possible_types = ["healthy_diet", "healthy_diet", "healthy_diet", "unhealthy_diet", "unhealthy_diet",
                          "unhealthy_diet", "antibiotics", "antibiotics", "stress", "stress"]

        # Select in order instead of randomly
        input_index = ExternalInput._counter % len(possible_types)
        self.input_name = possible_types[input_index]

        # Increment counter for next instance
        ExternalInput._counter += 1

        print(self.input_name)  # To verify selection order

        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return self.uid, self.input_name, self.pt.coordinates, self.context

    # External input step function
    def step(self, model):
        if model.barrier_impermeability >= model.barrier_permeability_threshold_stop:
            if self.input_name in ["unhealthy_diet", "healthy_diet"]:
                bacteria = self.calculate_effectiveness(model)
                adjust_bacteria(model, bacteria[0], bacteria[1])
            else:
                good_bact, pathogen_bact = model.params["external_input"][self.input_name]
                adjust_bacteria(model, good_bact, pathogen_bact)

    def calculate_effectiveness(self, model):
        diet = model.params["external_input"][self.input_name]
        eff = model.params["effectiveness"]

        good_bact, pathogen_bact = 0, 0
        for meal in diet:
            good_bact += diet[meal][0] * eff[meal]
            pathogen_bact += diet[meal][1] * eff[meal]

        return [good_bact / 5, pathogen_bact / 5]
