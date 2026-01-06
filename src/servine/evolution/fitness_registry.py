import numpy as np

from src.servine.evolution.fitness import PurifyingFitness, EpistaticFitness, FrequencyDependentFitness, \
    ExposureFitness, CategoricalFitness, NeutralFitness, SiteSpecificPurifyingFitness
from src.servine.color import fg


class FitnessRegistry:
    """A simple factory to fetch fitness models by name."""
    _models = {
        "purifying": PurifyingFitness,
        "epistatic": EpistaticFitness,
        "frequency": FrequencyDependentFitness,
        "exposure": ExposureFitness,
        "categorical": CategoricalFitness,
        "neutral": NeutralFitness,
        "site_specific": SiteSpecificPurifyingFitness
    }

    @classmethod
    def get(cls, name, **params):
        # Normalize name to lowercase to avoid case-sensitivity issues
        model_name = name.lower()

        if model_name in cls._models:
            # Dynamically instantiate the class from the dictionary
            return cls._models[model_name](**params)

        raise ValueError(fg.YELLOW, f"Unknown fitness model: {name}. Available: {list(cls._models.keys())}", fg.RESET)

    @classmethod
    def fitness_type_and_params(cls, e_conf, initial_seq, epoch_mutator):
        fitness_params = e_conf['fitness'].get('params', {}).copy()  # Use copy to avoid mutation issues
        # A. Global Reference handling: If initial sequence exists, inject it
        if initial_seq is not None:
            fitness_params['reference_sequence'] = initial_seq

        # B. Model-Specific Pre-processing
        fit_type = e_conf['fitness']['type'].lower()

        # --- Model-Specific Requirements ---
        # Exposure Fitness needs a mutator to drift the peak over time
        if fit_type == "exposure":
            fitness_params['mutator'] = epoch_mutator
        # Epistatic Fitness needs a 2D NumPy array for interactions
        elif fit_type == "epistatic" and "interaction_matrix" in fitness_params:
            fitness_params['interaction_matrix'] = np.array(fitness_params['interaction_matrix'])
        # Categorical Fitness needs a 1D NumPy array for site weights
        elif fit_type == "categorical" and "site_weights" in fitness_params:
            fitness_params['site_weights'] = np.array(fitness_params['site_weights'])
        elif fit_type == "site_specific":
            fitness_params['site_intensities'] = np.array(fitness_params['site_intensities'])
        return fitness_params, fit_type
