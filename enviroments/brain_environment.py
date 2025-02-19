from dataclasses import dataclass
from typing import Tuple

import numba
import numpy as np
from numba import int32, int64
from numba.experimental import jitclass
from repast4py import parameters
from repast4py.space import DiscretePoint as dpt

from agents.brain_agents.cytokine import Cytokine
from agents.brain_agents.microglia import Microglia
from agents.brain_agents.neuron import Neuron
from agents.gut_brain_agents.cleaved_protein import CleavedProtein
from agents.gut_brain_agents.oligomer import Oligomer


@dataclass
class Log:
    number_of_resting_microglia: int = 0
    number_of_active_microglia: int = 0
    number_of_healthy_neuron: int = 0
    number_of_damaged_neuron: int = 0
    number_of_dead_neuron: int = 0
    number_of_cleaved_alpha_syn: int = 0
    number_alpha_syn_oligomer: int = 0
    number_of_cleaved_tau: int = 0
    number_tau_oligomer: int = 0
    number_of_cytokine_pro_inflammatory: int = 0
    number_of_cytokine_non_inflammatory: int = 0


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
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    # prepare configuration depending on the agent type
    toRemoveID = None  # not used in all cases
    aggregatingAgent = False
    agent_state = agent_data[1]
    generateNewAgent = not (uid in agent_cache)

    match uid[1]:
        case Microglia.TYPE:  # type 0
            # toRemoveID not assigned (it's a meant feature probably)
            if generateNewAgent:
                if usingContext:
                    context = agent_data[3]
                    agent = Microglia(uid[0], uid[2], agent_state, pt, context)
                else:
                    agent = Microglia(uid[0], uid[2], agent_state, pt)
        case Neuron.TYPE:  # type 1
            toRemoveID = 3
            if generateNewAgent:
                if usingContext:
                    context = agent_data[4]
                    agent = Neuron(uid[0], uid[2], agent_state, pt, context)
                else:
                    agent = Neuron(uid[0], uid[2], agent_state, pt)
        case CleavedProtein.TYPE:  # type 2
            toRemoveID = 5
            aggregatingAgent = True
            if generateNewAgent:
                if usingContext:
                    context = agent_data[6]
                    agent = CleavedProtein(uid[0], uid[2], agent_state, pt, context)
                else:
                    agent = CleavedProtein(uid[0], uid[2], agent_state, pt)
        case Oligomer.TYPE:  # type 3
            toRemoveID = 3
            aggregatingAgent = True
            if generateNewAgent:
                if usingContext:
                    context = agent_data[4]
                    agent = Oligomer(uid[0], uid[2], agent_state, pt, context)
                else:
                    agent = Oligomer(uid[0], uid[2], agent_state, pt)
        case Cytokine.TYPE:  # type 4
            # toRemoveID not assigned (it's a meant feature probably)
            aggregatingAgent = True
            if generateNewAgent:
                if usingContext:
                    context = agent_data[3]
                    agent = Cytokine(uid[0], uid[2], agent_state, pt, context)
                else:
                    agent = Cytokine(uid[0], uid[2], agent_state, pt)

    # configure/update agent attributes
    if generateNewAgent:
        agent_cache[uid] = agent  # update the agent_cache if new agent has been created
    else:
        agent = agent_cache[uid]  # use already existing agent otherwise (in the worst scenario 'agent' is defined here)
    if usingContext:
        agent.context = context
    if toRemoveID != None:
        agent.toRemove = agent_data[toRemoveID]
    if aggregatingAgent:
        agent.toAggregate = agent_data[3]
        agent.alreadyAggregate = agent_data[4]
    agent.setAgentData(agent_state)
    agent.pt = pt

    return agent


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)

    # run(params)
