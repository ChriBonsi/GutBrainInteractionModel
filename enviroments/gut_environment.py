from dataclasses import dataclass
from typing import Tuple

import numba
import numpy as np
from numba import int32, int64
from numba.experimental import jitclass
from repast4py import parameters
from repast4py.space import DiscretePoint as dpt

from agents.gut_agents.aep import AEP
from agents.gut_agents.external_input import ExternalInput
from agents.gut_agents.normal_protein import Protein
from agents.gut_agents.treatment import Treatment
from agents.gut_brain_agents.cleaved_protein import CleavedProtein
from agents.gut_brain_agents.oligomer import Oligomer


@dataclass
class Log:
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein: int = 0
    tau_protein: int = 0
    alpha_cleaved: int = 0
    tau_cleaved: int = 0
    alpha_oligomer: int = 0
    tau_oligomer: int = 0
    barrier_impermeability: int = 0
    microbiota_good_bacteria_class: int = 0
    microbiota_pathogenic_bacteria_class: int = 0


@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


spec = [('mo', int32[:]), ('no', int32[:]), ('xmin', int32), ('ymin', int32), ('ymax', int32), ('xmax', int32)]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


# removed to permit the use of a unique common cache (defined in 'gut_brain_system.py')
# agent_cache = {}


# def restore_agent(agent_data: Tuple):
def restore_agent(agent_data: Tuple, agent_cache: dict, usingContext: bool = False):
    # uid: 0 id, 1 type, 2 rank
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    agent_state = agent_data[1]
    createNewAgent = True

    if uid in agent_cache:
        agent = agent_cache[uid]
        createNewAgent = False
    else:
        if uid[1] == AEP.TYPE:
            if createNewAgent:
                agent = AEP(uid[0], uid[2], pt)
            if usingContext:
                agent.context = agent_data[3]
        elif uid[1] == Protein.TYPE:
            # protein_name = agent_data[1]
            # agent = Protein(uid[0], uid[2], protein_name, pt)
            if createNewAgent:
                agent = Protein(uid[0], uid[2], agent_state, pt)
            agent.toCleave = agent_data[3]
            agent.toRemove = agent_data[4]
            if usingContext:
                agent.context = agent_data[5]
        elif uid[1] == CleavedProtein.TYPE:
            # cleaved_protein_name = agent_data[1]
            # agent = CleavedProtein(uid[0], uid[2], cleaved_protein_name, pt)
            if createNewAgent:
                agent = CleavedProtein(uid[0], uid[2], agent_state, pt)
            agent.toAggregate = agent_data[3]
            agent.alreadyAggregate = agent_data[4]
            agent.toRemove = agent_data[5]
            if usingContext:
                agent.context = agent_data[6]
        elif uid[1] == Oligomer.TYPE:
            # oligomer_name = agent_data[1]
            # agent = Oligomer(uid[0], uid[2], oligomer_name, pt)
            if createNewAgent:
                agent = Oligomer(uid[0], uid[2], agent_state, pt)
            if usingContext:
                agent.context = agent_data[4]
        elif uid[1] == ExternalInput.TYPE:
            if createNewAgent:
                agent = ExternalInput(uid[0], uid[2], pt)
            if usingContext:
                agent.context = agent_data[3]
        elif uid[1] == Treatment.TYPE:
            if createNewAgent:
                agent = Treatment(uid[0], uid[2], pt)
            if usingContext:
                agent.context = agent_data[3]  # agent_cache[uid] = agent

    if createNewAgent:
        agent_cache[uid] = agent
    agent.setAgentData(agent_state)
    agent.pt = pt

    return agent


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)  # run(params)
