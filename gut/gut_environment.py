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

from gut.aep import AEP
from gut.cleaved_protein import CleavedProtein
from gut.external_input import ExternalInput
from gut.normal_protein import Protein
from gut.oligomer import Oligomer
from gut.treatment import Treatment


# TODO understand why duplicate agents are created


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


agent_cache = {}


def restore_agent(agent_data: Tuple):
    # uid: 0 id, 1 type, 2 rank
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    if uid in agent_cache:
        agent = agent_cache[uid]
    else:
        if uid[1] == AEP.TYPE:
            agent = AEP(uid[0], uid[2], pt)
        elif uid[1] == Protein.TYPE:
            protein_name = agent_data[1]
            agent = Protein(uid[0], uid[2], protein_name, pt)
        elif uid[1] == CleavedProtein.TYPE:
            cleaved_protein_name = agent_data[1]
            agent = CleavedProtein(uid[0], uid[2], cleaved_protein_name, pt)
        elif uid[1] == Oligomer.TYPE:
            oligomer_name = agent_data[1]
            agent = Oligomer(uid[0], uid[2], oligomer_name, pt)
        elif uid[1] == ExternalInput.TYPE:
            agent = ExternalInput(uid[0], uid[2], pt)
        elif uid[1] == Treatment.TYPE:
            agent = Treatment(uid[0], uid[2], pt)
        agent_cache[uid] = agent

    agent.pt = pt

    if uid[1] == AEP.TYPE:
        agent.state = agent_data[1]
    elif uid[1] == Protein.TYPE:
        agent.toCleave = agent_data[3]
        agent.toRemove = agent_data[4]
    elif uid[1] == CleavedProtein.TYPE:
        agent.toAggregate = agent_data[3]
        agent.alreadyAggregate = agent_data[4]
        agent.toRemove = agent_data[5]
    elif uid[1] == ExternalInput.TYPE:
        agent.input_name = agent_data[1]
    elif uid[1] == Treatment.TYPE:
        agent.input_name = agent_data[1]
    return agent


class Model:

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        box = space.BoundingBox(0, params['world.width'] - 1, 0, params['world.height'] - 1, 0, 0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Single, buffer_size=1, comm=comm)
        self.context.add_projection(self.grid)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1, 2, self.microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 5, self.move_cleaved_protein_step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng

        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['gut_log_file'], buffer_size=1)

        self.microbiota_good_bacteria_class = params["microbiota_good_bacteria_class"]
        self.microbiota_pathogenic_bacteria_class = params["microbiota_pathogenic_bacteria_class"]
        self.microbiota_diversity_threshold = params["microbiota_diversity_threshold"]
        self.barrier_impermeability = params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = params["barrier_permeability_threshold_start"]

        world_size = self.comm.Get_size()
        self.added_agents_id = 0

        def distribute_agents(total_count, create_agent_fn):
            pp_count = int(total_count / world_size)
            if self.rank < total_count % world_size:
                pp_count += 1
            for i in range(pp_count):
                create_agent_fn(i)
            self.added_agents_id += pp_count

        def create_and_add_agent(agent_class, i, rank, name=None):
            pt = self.grid.get_random_local_pt(self.rng)
            if name:
                agent = agent_class(i + self.added_agents_id, rank, name, pt)
            else:
                agent = agent_class(i + self.added_agents_id, rank, pt)
            self.context.add(agent)
            self.move(agent, agent.pt)

        distribute_agents(params['aep_enzyme'], lambda i: create_and_add_agent(AEP, i, self.rank))
        distribute_agents(params['tau_proteins'],
                          lambda i: create_and_add_agent(Protein, i, self.rank, params["protein_name"]["tau"]))
        distribute_agents(params['alpha_syn_proteins'],
                          lambda i: create_and_add_agent(Protein, i, self.rank, params["protein_name"]["alpha_syn"]))
        distribute_agents(params['external_input_number'], lambda i: create_and_add_agent(ExternalInput, i, self.rank))

        if params['treatment']:
            distribute_agents(params['treatment_input_number'], lambda i: create_and_add_agent(Treatment, i, self.rank))

        distribute_agents(params['alpha_syn_oligomers'],
                          lambda i: create_and_add_agent(Oligomer, i, self.rank, params["protein_name"]["alpha_syn"]))
        distribute_agents(params['tau_oligomers'],
                          lambda i: create_and_add_agent(Oligomer, i, self.rank, params["protein_name"]["tau"]))

        self.context.synchronize(restore_agent)

    def move_cleaved_protein_step(self):
        for agent in self.context.agents():
            if type(agent) == CleavedProtein:
                if not agent.alreadyAggregate:
                    pt = self.grid.get_random_local_pt(self.rng)
                    self.move(agent, pt)

    def microbiota_dysbiosis_step(self):
        if self.microbiota_good_bacteria_class - self.microbiota_pathogenic_bacteria_class <= self.microbiota_diversity_threshold:
            value_decreased = int((params["barrier_impermeability"] * np.random.randint(0, 6)) / 100)
            if self.barrier_impermeability - value_decreased <= 0:
                self.barrier_impermeability = 0
            else:
                self.barrier_impermeability = self.barrier_impermeability - value_decreased
            number_of_aep_to_hyperactivate = value_decreased
            cont = 0
            for agent in self.context.agents(agent_type=0):
                if agent.state == params["aep_state"]["active"] and cont < number_of_aep_to_hyperactivate:
                    agent.state = params["aep_state"]["hyperactive"]
                    cont += 1
                elif cont == number_of_aep_to_hyperactivate:
                    break
        else:
            if self.barrier_impermeability < params["barrier_impermeability"]:
                value_increased = int((params["barrier_impermeability"] * np.random.randint(0, 4)) / 100)
                if (self.barrier_impermeability + value_increased) <= params["barrier_impermeability"]:
                    self.barrier_impermeability = self.barrier_impermeability + value_increased

    def step(self):
        self.context.synchronize(restore_agent)
        self.log_counts()

        def gather_agents_to_remove():
            return [agent for agent in self.context.agents() if
                    isinstance(agent, (Oligomer, CleavedProtein, Protein)) and agent.toRemove]

        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.context.synchronize(restore_agent)

        for agent in self.context.agents():
            agent.step()

        protein_to_remove = []
        all_true_cleaved_aggregates = []

        for agent in self.context.agents():
            if type(agent) == Protein and agent.toCleave == True:
                protein_to_remove.append(agent)
                agent.toRemove = True
            elif type(agent) == CleavedProtein and agent.toAggregate == True:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True

                # self.context.synchronize(restore_agent)

        for agent in protein_to_remove:
            if agent.uid in removed_ids:
                continue
            protein_name = agent.name
            self.remove_agent(agent)
            removed_ids.add(agent.uid)
            self.add_cleaved_protein(protein_name)
            self.add_cleaved_protein(protein_name)

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

        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid)

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_cleaved_protein(self, cleaved_protein_name):
        self.added_agents_id += 1
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

    def move(self, agent, pt):
        self.grid.move(agent, pt)
        agent.pt = self.grid.get_location(agent)

    def log_counts(self):
        tick = self.runner.schedule.tick

        aep_active = 0
        aep_hyperactive = 0
        alpha_protein = 0
        tau_protein = 0
        alpha_cleaved = 0
        tau_cleaved = 0
        alpha_oligomer = 0
        tau_oligomer = 0

        for agent in self.context.agents():
            if type(agent) == Oligomer:
                if agent.name == params["protein_name"]["alpha_syn"]:
                    alpha_oligomer += 1
                else:
                    tau_oligomer += 1
            if type(agent) == CleavedProtein:
                if agent.name == params["protein_name"]["alpha_syn"]:
                    alpha_cleaved += 1
                else:
                    tau_cleaved += 1
            elif type(agent) == Protein:
                if agent.name == params["protein_name"]["alpha_syn"]:
                    alpha_protein += 1
                else:
                    tau_protein += 1
            elif type(agent) == AEP:
                if agent.state == params["aep_state"]["active"]:
                    aep_active += 1
                else:
                    aep_hyperactive += 1

        self.counts.aep_active = aep_active
        self.counts.aep_hyperactive = aep_hyperactive
        self.counts.alpha_protein = alpha_protein
        self.counts.tau_protein = tau_protein
        self.counts.alpha_cleaved = alpha_cleaved
        self.counts.tau_cleaved = tau_cleaved
        self.counts.alpha_oligomer = alpha_oligomer
        self.counts.tau_oligomer = tau_oligomer
        self.counts.microbiota_good_bacteria_class = self.microbiota_good_bacteria_class
        self.counts.microbiota_pathogenic_bacteria_class = self.microbiota_pathogenic_bacteria_class
        self.counts.barrier_impermeability = self.barrier_impermeability
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
