from dataclasses import dataclass

@dataclass
class Log:
    # Gut variables
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein_gut: int = 0
    tau_protein_gut: int = 0
    alpha_cleaved_gut: int = 0
    tau_cleaved_gut: int = 0
    alpha_oligomer_gut: int = 0
    tau_oligomer_gut: int = 0
    barrier_impermeability: int = 0
    microbiota_good_bacteria_class: int = 0
    microbiota_pathogenic_bacteria_class: int = 0
    # Brain variables
    resting_microglia: int = 0
    active_microglia: int = 0
    healthy_neuron: int = 0
    damaged_neuron: int = 0
    dead_neuron: int = 0
    cleaved_alpha_syn_brain: int = 0
    alpha_syn_oligomer_brain: int = 0
    cleaved_tau_brain: int = 0
    tau_oligomer_brain: int = 0
    cytokine_pro_inflammatory: int = 0
    cytokine_non_inflammatory: int = 0