# Fitness landscapes (Purifying, Epistatic)

from abc import ABC, abstractmethod
import numpy as np
from ..population.container import Population


class FitnessModel(ABC):
    """
    The Base Template for all Fitness Landscapes.
    """
    def __init__(self, reference_sequence=None):
        self.reference_sequence = reference_sequence

    def set_reference(self, sequence: np.ndarray):
        """Allows locking the reference sequence from the CLI."""
        self.reference_sequence = sequence.copy()

    @abstractmethod
    def evaluate_population(self, population: Population) -> np.ndarray:
        """
        Must return a 1D NumPy array of fitness scores
        corresponding to each individual in the population.
        :param population: The Population to evaluate.
        :return: A NumPy array of fitness scores.
        """
        pass

    def update(self, generation: int):
        """Optional hook for models that change over time."""
        pass


class NeutralFitness(FitnessModel):
    """Naive fitness model where all individuals have equal fitness."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_population(self, population: Population) -> np.ndarray:
        # Everyone gets a fitness of 1.0
        return np.ones(population.get_count())


class PurifyingFitness(FitnessModel):
    """Fitness model where mutations reduce fitness exponentially."""
    def __init__(self, intensity: float = 0.1, reference_sequence=None):
        """
        :param intensity: How much each mutation hurts fitness (Selection Coefficient).
        """
        super().__init__(reference_sequence)
        self.intensity = intensity

    def evaluate_population(self, population: Population) -> np.ndarray:
        """
        Calculates fitness for every individual in the population.
        """
        matrix = population.get_matrix()

        # If no reference is set, use the first individual as the 'Wild Type'
        if self.reference_sequence is None:
            self.reference_sequence = matrix[0].copy()

        # 1. Count mutations relative to reference for every row
        # matrix != reference_sequence uses NumPy broadcasting
        mutation_counts = np.sum(matrix != self.reference_sequence, axis=1)

        # 2. Calculate fitness: w = (1 - s)^n
        # where s is intensity and n is number of mutations
        fitness_scores = np.power(1.0 - self.intensity, mutation_counts)

        # Ensure fitness never hits exactly zero to avoid math errors in selection
        return np.maximum(fitness_scores, 1e-10)


class EpistaticFitness(FitnessModel):
    """Fitness model with pairwise epistatic interactions."""
    def __init__(self, interaction_matrix: np.ndarray, reference_sequence=None):
        """
        :param interaction_matrix: A 2D matrix where matrix[i, j] is the
                                   fitness boost/penalty if both sites i and j are mutated.
        """
        super().__init__(reference_sequence)
        self.interaction_matrix = interaction_matrix

    def evaluate_population(self, population: Population) -> np.ndarray:
        matrix = population.get_matrix()
        if self.reference_sequence is None:
            self.reference_sequence = matrix[0].copy()

        # Identify where mutations exist (binary mask)
        mut_mask = (matrix != self.reference_sequence).astype(float)

        # Matrix multiplication calculates the sum of pairwise interactions
        # for every individual in one vectorized step.
        # Score = Mask * InteractionMatrix * Mask.T
        epistatic_effects = np.sum((mut_mask @ self.interaction_matrix) * mut_mask, axis=1)

        # Convert log-sum effects to fitness scores
        return np.exp(epistatic_effects)


class FrequencyDependentFitness(FitnessModel):
    """
    Fitness model where common genotypes are penalized.
    This simulates the immune system. The more "popular" a specific sequence becomes,
    the lower its fitness. This forces the virus to keep changing to "rare" genotypes to survive.
    """
    def __init__(self, pressure: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.pressure = pressure

    def evaluate_population(self, population: Population) -> np.ndarray:
        matrix = population.get_matrix()

        # Get counts of each unique individual
        _, inverse_indices, counts = np.unique(
            matrix, axis=0, return_inverse=True, return_counts=True
        )

        # Calculate frequencies for every individual in the population
        # inverse_indices maps each individual back to its count in the 'counts' array
        frequencies = counts[inverse_indices] / len(matrix)

        fitness_scores = 1.0 - (frequencies * self.pressure)
        return np.maximum(fitness_scores, 1e-10)


class ExposureFitness(PurifyingFitness):
    """
    Note: Inherits from PurifyingFitness.
    This models a changing environment.
    Every X generations, the "optimal" sequence changes, and the population must catch up or die.
    """
    def __init__(self, intensity: float, update_interval: int, mutator, **kwargs):
        super().__init__(intensity, **kwargs)
        self.update_interval = update_interval
        self.mutator = mutator          # Uses a mutator to "drift" the peak

    def update_peak(self, generation: int):
        if generation % self.update_interval == 0:
            # Shift the reference sequence slightly
            # Effectively "moving the goalposts" for the population
            tmp_pop = Population(self.reference_sequence.reshape(1, -1))
            self.mutator.apply(tmp_pop)
            self.reference_sequence = tmp_pop.get_matrix()[0]


class CategoricalFitness(FitnessModel):
    """
    Not all genes are equal.
    Some sites are "locked" (Lethal if mutated), while others are "flexible" (Neutral).
    """
    def __init__(self, site_weights: np.ndarray, reference_sequence=None, **kwargs):
        """
        :param site_weights: Array of length L, where 0.0 = Neutral and 1.0 = Lethal.
        """
        super().__init__(reference_sequence, **kwargs)
        self.site_weights = site_weights

    def evaluate_population(self, population: Population) -> np.ndarray:
        matrix = population.get_matrix()
        diffs = (matrix != self.reference_sequence)

        # If any difference occurs at a site where weight is 1.0,
        # that individual should likely have 0 fitness.
        # This is a bitwise 'any' check across lethal columns.
        lethal_mask = np.any(diffs[:, self.site_weights == 1.0], axis=1)

        fitness = np.exp(-np.sum(diffs * self.site_weights, axis=1))
        fitness[lethal_mask] = 1e-10  # Effectively dead
        return fitness


class FitnessRegistry:
    """A simple factory to fetch fitness models by name."""
    _models = {
        "purifying": PurifyingFitness,
        "epistatic": EpistaticFitness,
        "frequency": FrequencyDependentFitness,
        "exposure": ExposureFitness,
        "categorical": CategoricalFitness,
        "neutral": NeutralFitness
    }

    @classmethod
    def get(cls, name, **params):
        # Normalize name to lowercase to avoid case-sensitivity issues
        model_name = name.lower()

        if model_name in cls._models:
            # Dynamically instantiate the class from the dictionary
            return cls._models[model_name](**params)

        raise ValueError(f"Unknown fitness model: {name}. Available: {list(cls._models.keys())}")
