from dataclasses import dataclass
from typing import Tuple

import numba
from numba import int32, int64
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


# brain agents
def create_new_cytokine(local_id: int, rank: int, pt, model) -> Cytokine:
    return Cytokine(local_id, rank, pt, "brain", model)


def create_new_microglia(local_id: int, rank: int, initial_state, pt) -> Microglia:
    return Microglia(local_id, rank, initial_state, pt, "brain")


def create_new_neuron(local_id: int, rank: int, initial_state, pt) -> Neuron:
    return Neuron(local_id, rank, initial_state, pt, "brain")


# gut-brain agents
def create_new_cleavedProtein(local_id: int, rank: int, cleaved_protein_name, pt) -> CleavedProtein:
    return CleavedProtein(local_id, rank, cleaved_protein_name, pt, "brain")


def create_new_Oligomer(local_id: int, rank: int, oligomer_name, pt) -> Oligomer:
    return Oligomer(local_id, rank, oligomer_name, pt, "brain")


# def restore_agent(agent_data: Tuple):
def restore_agent(agent_data: Tuple, agent_cache: dict, usingContext: bool = False):
    if agent_data == None:
        return  # stopgap resolution for None agents

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
