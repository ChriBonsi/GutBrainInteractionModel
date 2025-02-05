from dataclasses import dataclass
from typing import Dict, Tuple

import numba
import numpy as np
import repast4py
import repast4py.random
from mpi4py import MPI
from numba import int32, int64
from numba.experimental import jitclass
from repast4py import context as ctx
from repast4py import space, schedule, logging, parameters
from repast4py.space import DiscretePoint as dpt

from brain.cleaved_protein import CleavedProtein
from brain.cytokine import Cytokine
from brain.microglia import Microglia
from brain.neuron import Neuron
from brain.oligomer import Oligomer


# TODO understand why duplicate agents are created


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

        # Remove the original (x, y) coordinate
        # mask = (xs != x) | (ys != y)
        # xs = xs[mask]
        # ys = ys[mask]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


agent_cache = {}


def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)
    if uid[1] == Microglia.TYPE:
        agent_state = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Microglia(uid[0], uid[2], agent_state, pt)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.pt = pt
    elif uid[1] == Neuron.TYPE:
        agent_state = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Neuron(uid[0], uid[2], agent_state, pt)
            agent_cache[uid] = agent
        agent.state = agent_state
        agent.toRemove = agent_data[3]
        agent.pt = pt
    elif uid[1] == CleavedProtein.TYPE:
        agent_name = agent_data[1]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = CleavedProtein(uid[0], uid[2], agent_name, pt)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.toAggregate = agent_data[3]
        agent.alreadyAggregate = agent_data[4]
        agent.toRemove = agent_data[5]
        agent.pt = pt
    elif uid[1] == Oligomer.TYPE:
        agent_name = agent_data[1]
        toRemove = agent_data[3]
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Oligomer(uid[0], uid[2], agent_name, pt)
            agent_cache[uid] = agent
        agent.name = agent_name
        agent.pt = pt
        agent.toRemove = toRemove
    elif uid[1] == Cytokine.TYPE:
        if uid in agent_cache:
            agent = agent_cache[uid]
        else:
            agent = Cytokine(uid[0], uid[2], pt)
            agent_cache[uid] = agent
        agent.pt = pt
        agent.state = agent_data[1]
    return agent


class Model:
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        self.box = space.BoundingBox(0, params['world.width'] - 1, 0, params['world.height'] - 1, 0, 0)
        self.grid = space.SharedGrid(name='grid', bounds=self.box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)
        self.ngh_finder = GridNghFinder(0, 0, self.box.xextent, self.box.yextent)

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.check_nervous_system_death, priority_type=1)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng

        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['brain_log_file'], buffer_size=1)

        self.world_size = self.comm.Get_size()
        self.added_agents_id = 0

        agent_types = [('neuron_healthy.count', Neuron, 'healthy'), ('neuron_damaged.count', Neuron, 'damaged'),
                       ('neuron_dead.count', Neuron, 'dead'), ('resting_microglia.count', Microglia, 'resting'),
                       ('active_microglia.count', Microglia, 'active'),
                       ('alpha_syn_cleaved.count', CleavedProtein, 'alpha_syn'),
                       ('tau_cleaved.count', CleavedProtein, 'tau'),
                       ('alpha_syn_oligomer.count', Oligomer, 'alpha_syn'), ('tau_oligomer.count', Oligomer, 'tau'),
                       ('cytokine.count', Cytokine, None)]

        for agent_type in agent_types:
            total_count = params[agent_type[0]]
            pp_count = self.calculate_partitioned_count(total_count)
            self.create_agents(agent_type[1], pp_count, agent_type[2], params)

        self.pro_cytokine = 0
        self.anti_cytokine = 0
        self.dead_neuron = self.calculate_partitioned_count(params['neuron_dead.count'])

    def calculate_partitioned_count(self, total_count):
        pp_count = total_count // self.world_size
        if self.rank < total_count % self.world_size:
            pp_count += 1
        return pp_count

    def create_agents(self, agent_class, pp_count, state_key, params):
        for j in range(pp_count):
            pt = self.grid.get_random_local_pt(self.rng)
            if agent_class == Neuron or agent_class == Microglia:
                agent = agent_class(self.added_agents_id + j, self.rank,
                                    params[f"{agent_class.__name__.lower()}_state"][state_key], pt)
            elif agent_class == CleavedProtein or agent_class == Oligomer:
                agent = agent_class(self.added_agents_id + j, self.rank, params["protein_name"][state_key], pt)
            else:  # for cytokine
                agent = agent_class(self.added_agents_id + j, self.rank, pt)
            self.context.add(agent)
            self.move(agent, pt)
        self.added_agents_id += pp_count

    def check_nervous_system_death(self):
        non_dead_neurons_count = 0
        for agent in self.context.agents():
            if isinstance(agent, Neuron) and (
                    agent.state == params["neuron_state"]["healthy"] or agent.state == params["neuron_state"][
                "damaged"]):
                non_dead_neurons_count += 1

        if non_dead_neurons_count == 0:
            print("There are no more alive neurons, ending the simulation.")
            self.comm.Abort()

    def step(self):
        self.context.synchronize(restore_agent)
        self.log_counts()

        def gather_agents_to_remove():
            return [agent for agent in self.context.agents() if
                    isinstance(agent, (Oligomer, CleavedProtein, Neuron)) and agent.toRemove]

        # Remove agents marked for removal
        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.context.synchronize(restore_agent)

        # Let each agent perform its step
        for agent in self.context.agents():
            agent.step()

        # Collect data and perform operations based on agent states
        oligomer_to_remove = []
        active_microglia = 0
        damaged_neuron = 0
        all_true_cleaved_aggregates = []

        for agent in self.context.agents():
            if isinstance(agent, Oligomer) and agent.toRemove:
                oligomer_to_remove.append(agent)
            elif isinstance(agent, Microglia) and agent.state == params["microglia_state"]["active"]:
                active_microglia += 1
            elif isinstance(agent, Neuron) and agent.state == params["neuron_state"]["damaged"]:
                damaged_neuron += 1
            elif isinstance(agent, CleavedProtein) and agent.toAggregate:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True

        for _ in range(active_microglia):
            self.add_cytokine()
        for _ in range(damaged_neuron):
            self.add_cleaved_protein()
        for oligomer in oligomer_to_remove:
            if self.context.agent(oligomer.uid) is not None:
                self.remove_agent(oligomer)
                removed_ids.add(oligomer.uid)

        self.context.synchronize(restore_agent)

        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid():
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.context.agent(x.uid) is not None:
                                self.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                self.add_oligomer_protein(agent.name)
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.context.synchronize(restore_agent)

        # Remove agents marked for removal after all processing
        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid)

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_cleaved_protein(self):
        self.added_agents_id += 1
        possible_types = [params["protein_name"]["alpha_syn"], params["protein_name"]["tau"]]
        random_index = np.random.randint(0, len(possible_types))
        cleaved_protein_name = possible_types[random_index]
        pt = self.grid.get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt)
        self.context.add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt)

    def add_oligomer_protein(self, oligomer_name):
        self.added_agents_id += 1
        pt = self.grid.get_random_local_pt(self.rng)
        oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt)
        self.context.add(oligomer_protein)
        self.move(oligomer_protein, oligomer_protein.pt)

    def add_cytokine(self):
        self.added_agents_id += 1
        pt = self.grid.get_random_local_pt(self.rng)
        cytokine = Cytokine(self.added_agents_id, self.rank, pt)
        self.context.add(cytokine)
        self.move(cytokine, cytokine.pt)

    def move(self, agent, pt):
        self.grid.move(agent, pt)
        agent.pt = self.grid.get_location(agent)

    def log_counts(self):
        tick = self.runner.schedule.tick

        counts = {"microglia_resting": 0, "microglia_active": 0, "neuron_healthy": 0, "neuron_damaged": 0,
                  "alpha_cleaved": 0, "tau_cleaved": 0, "alpha_oligomer": 0, "tau_oligomer": 0}

        for agent in self.context.agents():
            if isinstance(agent, Oligomer):
                if agent.name == params["protein_name"]["alpha_syn"]:
                    counts["alpha_oligomer"] += 1
                else:
                    counts["tau_oligomer"] += 1
            elif isinstance(agent, CleavedProtein):
                if agent.name == params["protein_name"]["alpha_syn"]:
                    counts["alpha_cleaved"] += 1
                else:
                    counts["tau_cleaved"] += 1
            elif isinstance(agent, Neuron):
                if agent.state == params["neuron_state"]["healthy"]:
                    counts["neuron_healthy"] += 1
                elif agent.state == params["neuron_state"]["damaged"]:
                    counts["neuron_damaged"] += 1
            elif isinstance(agent, Microglia):
                if agent.state == params["microglia_state"]["active"]:
                    counts["microglia_active"] += 1
                else:
                    counts["microglia_resting"] += 1

        self.counts.number_of_healthy_neuron = counts["neuron_healthy"]
        self.counts.number_of_damaged_neuron = counts["neuron_damaged"]
        self.counts.number_of_dead_neuron = self.dead_neuron
        self.counts.number_of_cytokine_pro_inflammatory = self.pro_cytokine
        self.counts.number_of_cytokine_non_inflammatory = self.anti_cytokine
        self.counts.number_of_cleaved_alpha_syn = counts["alpha_cleaved"]
        self.counts.number_of_cleaved_tau = counts["tau_cleaved"]
        self.counts.number_alpha_syn_oligomer = counts["alpha_oligomer"]
        self.counts.number_tau_oligomer = counts["tau_oligomer"]
        self.counts.number_of_resting_microglia = counts["microglia_resting"]
        self.counts.number_of_active_microglia = counts["microglia_active"]

        self.data_set.log(tick)

    def at_end(self):
        self.data_set.close()

    def start(self):
        self.runner.execute()


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)

    run(params)
