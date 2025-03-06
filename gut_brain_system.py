from typing import Dict, List

import numpy as np
import pygame
import repast4py
import repast4py.random
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import space, schedule, logging, parameters
from repast4py.space import DiscretePoint as dpt

from utilities.graphic.gridNghFinder import GridNghFinder
from utilities.graphic.gui import GUI
from utilities.gutBrainInterface import GutBrainInterface
from utilities.log import Log

from environments import brain_environment, gut_environment

class Model:

    # Initialize the model
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.params = params #TODO remove this variable adding the missing parameters (ex. existing_meals and cyclic_menu)
        self.clock = pygame.time.Clock()  # Add a clock object
        self.FPS = 12  # Set max FPS limit
        self.comm = comm
        self.rank = comm.Get_rank()
        # Create shared contexts for the brain and the gut
        self.gut_context = ctx.SharedContext(comm)
        self.brain_context = ctx.SharedContext(comm)
        # Create shared grids for the brain and the gut
        box = space.BoundingBox(0, params['world.width'] - 1, 0, params['world.height'] - 1, 0, 0)
        self.gut_grid = self.init_grid('gut_grid', box, self.gut_context)
        self.brain_grid = self.init_grid('brain_grid', box, self.brain_context)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.gutBrainInterface = GutBrainInterface(self.gut_context, self.brain_context, self.params["seed"])
        # Initialize the schedule runner
        self.runner = schedule.init_schedule_runner(comm)
        self.init_schedule(params)
        # Set the seed for the random number generator
        repast4py.random.seed = params['seed']
        self.rng = repast4py.random.default_rng
        # Initialize the log
        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, params['log_file'], buffer_size=1)
        # Initialize the model parameters
        self.init_microbiota_params(params)
        self.world_size = self.comm.Get_size()
        self.added_agents_id = 0
        self.pro_cytokine = 0
        self.anti_cytokine = 0
        self.dead_neuron = self.calculate_partitioned_count(params['neuron_dead.count'])
        self.agent_cache = {}
        # Initialize the agents
        agent_types_gut = [('aep_enzyme.count', "AEP", None),
                           ('tau_proteins.count', "Protein", params["protein_name"]["tau"]),
                           ('alpha_syn_proteins.count', "Protein", params["protein_name"]["alpha_syn"]),
                           ('external_input.count', "ExternalInput", None), ('treatment_input.count', "Treatment", None),
                           ('alpha_syn_oligomers_gut.count', "Oligomer", params["protein_name"]["alpha_syn"]),
                           ('tau_oligomers_gut.count', "Oligomer", params["protein_name"]["tau"]), ]
        agent_types_brain = [('neuron_healthy.count', "Neuron", 'healthy'), ('neuron_damaged.count', "Neuron", 'damaged'),
                             ('neuron_dead.count', "Neuron", 'dead'), ('resting_microglia.count', "Microglia", 'resting'),
                             ('active_microglia.count', "Microglia", 'active'),
                             ('alpha_syn_cleaved_brain.count', "CleavedProtein", params["protein_name"]["alpha_syn"]),
                             ('tau_cleaved_brain.count', "CleavedProtein", params["protein_name"]["tau"]),
                             ('alpha_syn_oligomer_brain.count', "Oligomer", params["protein_name"]["alpha_syn"]),
                             ('tau_oligomer_brain.count', "Oligomer", params["protein_name"]["tau"]),
                             ('cytokine.count', "Cytokine", None)]
        self.distribute_all_agents(agent_types_gut, self.gut_context, self.gut_grid, 'gut')
        self.distribute_all_agents(agent_types_brain, self.brain_context, self.brain_grid, 'brain')
        
        # Synchronize the contexts
        self.gut_context.synchronize(gut_environment.restore_agent)
        self.brain_context.synchronize(brain_environment.restore_agent)

        # Initialize Pygame and gui object
        pygame.init()
        self.screen = GUI(width=1600, height=800, gut_context=self.gut_context, brain_context=self.brain_context,
                          grid_dimensions=(params['world.width'], params['world.height']))
        pygame.display.set_caption("Gut-Brain Axis Model")
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)

    # Function to initialize the shared grid
    def init_grid(self, name, box, context):
        grid = space.SharedGrid(name=name, bounds=box, borders=space.BorderType.Sticky,
                                occupancy=space.OccupancyType.Multiple, buffer_size=1, comm=self.comm)
        context.add_projection(grid)
        return grid

    # Function to initialize the schedule
    def init_schedule(self, params):
        self.runner.schedule_repeating_event(1, 1, self.gut_step)
        self.runner.schedule_repeating_event(1, 2, self.microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 5, self.move_cleaved_protein_step)
        self.runner.schedule_repeating_event(1, 1, self.brain_step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.pygame_update, priority_type=1)
        self.runner.schedule_repeating_event(1, 1, self.log_counts, priority_type=1)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

    # Function to initialize the microbiota parameters
    def init_microbiota_params(self, params):
        # Total count of the good and pathogenic bacteria in the microbiota
        self.good_bact_count = params["microbiota_good_bacteria_class"]
        self.pathogenic_bact_count = params["microbiota_pathogenic_bacteria_class"]

        self.microbiota_diversity_threshold = params["microbiota_diversity_threshold"]
        self.barrier_impermeability = params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = params["barrier_permeability_threshold_start"]

    # Function to distribute all agents through the different ranks
    def distribute_all_agents(self, agent_types, context, grid, region):
        for agent_type in agent_types:
            total_count = self.params[agent_type[0]]
            pp_count = self.calculate_partitioned_count(total_count)
            self.create_agents(agent_type[1], pp_count, agent_type[2], context, grid, region)

    # Function to create agents in the different ranks based on the total count
    def create_agents(self, agent_class, pp_count, state_key, context, grid, region):
        for j in range(pp_count):
            pt = grid.get_random_local_pt(self.rng)

            agent = _create_new_agent(agent_class, self.added_agents_id + j, self.rank, state_key, pt, region, self)

            context.add(agent)
            self.move(agent, pt, agent.context)
        self.added_agents_id += pp_count

    # Function to get the total count of agents to create in that rank
    def calculate_partitioned_count(self, total_count):
        pp_count = int(total_count / self.world_size)
        if self.rank < total_count % self.world_size:
            pp_count += 1
        return pp_count

    # Function to update the interface
    def pygame_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the 'X' button is clicked, stop the simulation
                print("Ending the simulation.")
                self.at_end()
                self.comm.Abort()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.screen.handle_button_click(event.pos)

        while self.screen.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # If the 'X' button is clicked, stop the simulation
                    print("Ending the simulation.")
                    self.at_end()
                    self.comm.Abort()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.screen.handle_button_click(event.pos)

        # Updates the Pygame GUI based on the current state of the Repast simulation
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)
        pygame.display.flip()

        self.clock.tick(self.FPS)

        # Brain steps

    def brain_step(self):
        self.brain_context.synchronize(brain_environment.restore_agent)

        def gather_agents_to_remove():
            return [agent for agent in self.brain_context.agents() if
                    agent.TYPE in {3, 2, 7} and agent.toRemove] # if agent is Oligomer, CleavedProtein or Neuron

        # Remove agents marked for removal
        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.brain_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.brain_context.synchronize(brain_environment.restore_agent)

        # Let each agent perform its step
        for agent in self.brain_context.agents():
            agent.step(self)

        # Collect data and perform operations based on agent states
        oligomer_to_remove = []
        active_microglia = 0
        damaged_neuron = 0
        all_true_cleaved_aggregates = []

        for agent in self.brain_context.agents():
            match agent.TYPE:
                case 2: #cleaved protein
                    if agent.toAggregate:
                        all_true_cleaved_aggregates.append(agent)
                        agent.toRemove = True
                case 3: #Oligomer
                    if agent.toRemove:
                        oligomer_to_remove.append(agent)
                case 6: #microglia
                    if agent.state == "active":
                        active_microglia += 1
                case 7: #neuron
                    if agent.state == "damaged":
                        damaged_neuron += 1



        for _ in range(active_microglia):
            self.add_cytokine()
        for _ in range(damaged_neuron):
            self.brain_add_cleaved_protein()
        for oligomer in oligomer_to_remove:
            if self.brain_context.agent(oligomer.uid) is not None:
                self.remove_agent(oligomer)
                removed_ids.add(oligomer.uid)

        self.brain_context.synchronize(brain_environment.restore_agent)

        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid(self):
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs(self)
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.brain_context.agent(x.uid) is not None:
                                self.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                self.add_oligomer_protein(agent.name, agent.context)
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.brain_context.synchronize(brain_environment.restore_agent)

        # Remove agents marked for removal after all processing
        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.brain_context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid)

    # Function to remove an agent from the context and the grid
    def remove_agent(self, agent):
        if agent.context == 'brain':
            self.brain_context.remove(agent)
        else:
            self.gut_context.remove(agent)

            # Function to add a cleaved protein agent to the brain context

    def brain_add_cleaved_protein(
            self):
        self.added_agents_id += 1
        possible_types: List[str] = list(self.params.get("protein_name").values())
        random_index = np.random.randint(0, len(possible_types))
        cleaved_protein_name = possible_types[random_index]
        pt = self.brain_grid.get_random_local_pt(self.rng)
        cleaved_protein = brain_environment.create_new_cleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt)
        self.brain_context.add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt, cleaved_protein.context)

    # Function to add a cleaved protein agent to the gut context
    def gut_add_cleaved_protein(self, cleaved_protein_name):
        self.added_agents_id += 1
        pt = self.gut_grid.get_random_local_pt(self.rng)
        cleaved_protein = gut_environment.create_new_cleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt)
        self.gut_context.add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt, 'gut')

    # Function to add an oligomer protein agent to the brain or gut context
    def add_oligomer_protein(self, oligomer_name, context):
        self.added_agents_id += 1
        if context == 'brain':
            pt = self.brain_grid.get_random_local_pt(self.rng)
            oligomer_protein = brain_environment.create_new_Oligomer(self.added_agents_id, self.rank, oligomer_name, pt)
            self.brain_context.add(oligomer_protein)
            self.move(oligomer_protein, oligomer_protein.pt, 'brain')
        else:
            pt = self.gut_grid.get_random_local_pt(self.rng)
            oligomer_protein = gut_environment.create_new_Oligomer(self.added_agents_id, self.rank, oligomer_name, pt)
            self.gut_context.add(oligomer_protein)
            self.move(oligomer_protein, oligomer_protein.pt, 'gut')

            # Function to move an agent to a new location

    def move(self, agent, pt: dpt, context):
        if context == 'brain':
            self.brain_grid.move(agent, pt)
        else:
            self.gut_grid.move(agent, pt)
        agent.pt = pt

    # Function to add a cytokine agent to the brain context
    def add_cytokine(self):
        self.added_agents_id += 1
        pt = self.brain_grid.get_random_local_pt(self.rng)
        cytokine = brain_environment.create_new_cytokine(self.added_agents_id, self.rank, pt, self)
        self.brain_context.add(cytokine)
        self.move(cytokine, cytokine.pt, 'brain')

    # Gut steps 
    # Function to move the cleaved protein agents 
    def move_cleaved_protein_step(self):
        for agent in self.gut_context.agents():
            if agent.TYPE == 2: #CleavedProtein
                if not agent.alreadyAggregate:
                    pt = self.gut_grid.get_random_local_pt(self.rng)
                    self.move(agent, pt, agent.context)
        for agent in self.brain_context.agents():
            if agent.TYPE == 2: #CleavedProtein
                if not agent.alreadyAggregate:
                    pt = self.brain_grid.get_random_local_pt(self.rng)
                    self.move(agent, pt, agent.context)

    # Function to check if the microbiota is dysbiotic and adjust the barrier impermeability 
    def microbiota_dysbiosis_step(self):
        if self.good_bact_count - self.pathogenic_bact_count <= self.microbiota_diversity_threshold:
            value_decreased = int((params["barrier_impermeability"] * np.random.randint(0, 6)) / 100)
            if self.barrier_impermeability - value_decreased <= 0:
                self.barrier_impermeability = 0
            else:
                self.barrier_impermeability = self.barrier_impermeability - value_decreased
            number_of_aep_to_hyperactivate = value_decreased
            cont = 0
            for agent in self.gut_context.agents(agent_type=0):
                if agent.state == "active" and cont < number_of_aep_to_hyperactivate:
                    agent.state = "hyperactive"
                    cont += 1
                elif cont == number_of_aep_to_hyperactivate:
                    break
        else:
            if self.barrier_impermeability < self.params["barrier_impermeability"]:
                value_increased = int((self.params["barrier_impermeability"] * np.random.randint(0, 4)) / 100)
                if (self.barrier_impermeability + value_increased) <= self.params["barrier_impermeability"]:
                    self.barrier_impermeability = self.barrier_impermeability + value_increased

    def gut_step(self):
        self.gut_context.synchronize(gut_environment.restore_agent)

        def gather_agents_to_remove():
            return [agent for agent in self.gut_context.agents() if
                    agent.TYPE in {3, 2, 1} and agent.toRemove] # if agent is Oligomer, CleavedProtein or Protein

        remove_agents = gather_agents_to_remove()
        removed_ids = set()
        for agent in remove_agents:
            if self.gut_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.gut_context.synchronize(gut_environment.restore_agent)

        for agent in self.gut_context.agents():
            agent.step(self)

        protein_to_remove = []
        all_true_cleaved_aggregates = []
        oligomers_to_move = []

        for agent in self.gut_context.agents():
            match agent.TYPE:
                case 1: #protein
                    if agent.toCleave:
                        protein_to_remove.append(agent)
                        agent.toRemove = True
                case 2: #CleavedProtein 
                    if agent.toAggregate:
                        all_true_cleaved_aggregates.append(agent)
                        agent.toRemove = True
                case 3: #Oligomer
                    if agent.toMove:
                        oligomers_to_move.append(agent)
                        agent.toRemove = True

        for agent in oligomers_to_move:
            self.gutBrainInterface.transfer_from_gut_to_brain(agent)

        for agent in protein_to_remove:
            if agent.uid in removed_ids:
                continue
            protein_name = agent.name
            self.remove_agent(agent)
            removed_ids.add(agent.uid)
            self.gut_add_cleaved_protein(protein_name)
            self.gut_add_cleaved_protein(protein_name)

        self.gut_context.synchronize(gut_environment.restore_agent)

        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid(self):
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs(self)
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.gut_context.agent(x.uid) is not None:
                                self.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                self.add_oligomer_protein(agent.name, 'gut')
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

        self.gut_context.synchronize(gut_environment.restore_agent)

        remove_agents = gather_agents_to_remove()
        for agent in remove_agents:
            if agent.uid not in removed_ids:
                if self.gut_context.agent(agent.uid) is not None:
                    self.remove_agent(agent)
                    removed_ids.add(agent.uid)

                    # Function to log the counts of the agents

    def log_counts(self):
        tick = self.runner.schedule.tick

        counts = {"aep_active": 0, "aep_hyperactive": 0, "alpha_protein_gut": 0, "tau_protein_gut": 0,
                  "alpha_cleaved_gut": 0, "tau_cleaved_gut": 0, "alpha_oligomer_gut": 0, "tau_oligomer_gut": 0,
                  "microglia_resting": 0, "microglia_active": 0, "neuron_healthy": 0, "neuron_damaged": 0,
                  "alpha_cleaved_brain": 0, "tau_cleaved_brain": 0, "alpha_oligomer_brain": 0, "tau_oligomer_brain": 0}

        for agent in self.brain_context.agents():
            match agent.TYPE:
                case 3: #Oligomer
                    if agent.name == "alpha_syn":
                        counts["alpha_oligomer_brain"] += 1
                    else:
                        counts["tau_oligomer_brain"] += 1
                case 2: #CleavedProtein
                    if agent.name == "alpha_syn":
                        counts["alpha_cleaved_brain"] += 1
                    else:
                        counts["tau_cleaved_brain"] += 1
                case 7: #Neuron
                    if agent.state == "healthy":
                        counts["neuron_healthy"] += 1
                    elif agent.state == "damaged":
                        counts["neuron_damaged"] += 1
                case 6: #Microglia
                    if agent.state == "active":
                        counts["microglia_active"] += 1
                    else:
                        counts["microglia_resting"] += 1

        for agent in self.gut_context.agents():
            match agent.type:
                case 3: # Oligomer
                    if agent.name == "alpha_syn":
                        counts["alpha_oligomer_gut"] += 1
                    else:
                        counts["tau_oligomer_gut"] += 1
                case 2: #CleavedProtein
                    if agent.name == "alpha_syn":
                        counts["alpha_cleaved_gut"] += 1
                    else:
                        counts["tau_cleaved_gut"] += 1
                case 7: #Protein
                    if agent.name == "alpha_syn":
                        counts["alpha_protein_gut"] += 1
                    else:
                        counts["tau_protein_gut"] += 1
                case 0: #AEP
                    if agent.state == "active":
                        counts["aep_active"] += 1
                    else:
                        counts["aep_hyperactive"] += 1

        # brain
        self.counts.healthy_neuron = counts["neuron_healthy"]
        self.counts.damaged_neuron = counts["neuron_damaged"]
        self.counts.dead_neuron = self.dead_neuron
        self.counts.cytokine_pro_inflammatory = self.pro_cytokine
        self.counts.cytokine_non_inflammatory = self.anti_cytokine
        self.counts.cleaved_alpha_syn_brain = counts["alpha_cleaved_brain"]
        self.counts.cleaved_tau_brain = counts["tau_cleaved_brain"]
        self.counts.alpha_syn_oligomer_brain = counts["alpha_oligomer_brain"]
        self.counts.tau_oligomer_brain = counts["tau_oligomer_brain"]
        self.counts.resting_microglia = counts["microglia_resting"]
        self.counts.active_microglia = counts["microglia_active"]

        # gut
        self.counts.aep_active = counts["aep_active"]
        self.counts.aep_hyperactive = counts["aep_hyperactive"]
        self.counts.alpha_protein_gut = counts["alpha_protein_gut"]
        self.counts.tau_protein_gut = counts["tau_protein_gut"]
        self.counts.alpha_cleaved_gut = counts["alpha_cleaved_gut"]
        self.counts.tau_cleaved_gut = counts["tau_cleaved_gut"]
        self.counts.alpha_oligomer_gut = counts["alpha_oligomer_gut"]
        self.counts.tau_oligomer_gut = counts["tau_oligomer_gut"]
        self.counts.microbiota_good_bacteria_class = self.good_bact_count
        self.counts.microbiota_pathogenic_bacteria_class = self.pathogenic_bact_count
        self.counts.barrier_impermeability = self.barrier_impermeability

        self.data_set.log(tick)

    # Function to close the data set and quit Pygame
    def at_end(self):
        self.data_set.close()
        pygame.quit()

    # Function to start the simulation
    def start(self):
        self.runner.execute()


def _create_new_agent(agent_class: str, local_id: int, rank: int, nameOrState: str, pt, region: str, model: Model):
    """Stopgap function created as universal constructor call for all defined agents."""
    if region == "brain":
        match agent_class:
            case "Cytokine":
                return brain_environment.create_new_cytokine(local_id, rank, nameOrState, pt)
            case "Microglia":
                return brain_environment.create_new_microglia(local_id, rank, nameOrState, pt)
            case "Neuron":
                return brain_environment.create_new_neuron(local_id, rank, nameOrState, pt)
            case "CleavedProtein":
                return brain_environment.create_new_cleavedProtein(local_id, rank, nameOrState, pt)
            case "Oligomer":
                return brain_environment.create_new_Oligomer(local_id, rank, nameOrState, pt)
            case _:
                raise ValueError("Not supported agent")
    elif region == "gut":
        match agent_class:
            case "AEP":
                return gut_environment.create_new_aep(local_id, rank, pt, model)
            case "ExternalInput":
                return gut_environment.create_new_externalInput(local_id, rank, pt, model)
            case "Treatment":
                return gut_environment.create_new_treatment(local_id, rank, pt, model)
            case "Protein":
                return gut_environment.create_new_protein(local_id, rank, nameOrState, pt, model)
            case "CleavedProtein":
                return gut_environment.create_new_cleavedProtein(local_id, rank, nameOrState, pt)
            case "Oligomer":
                return gut_environment.create_new_Oligomer(local_id, rank, nameOrState, pt)
            case _:
                raise ValueError("Not supported agent")
    else:
        raise ValueError("not supported region")

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)