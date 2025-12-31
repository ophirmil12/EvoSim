# Fitness landscapes (Purifying, Epistatic)


import numpy as np
from ..population.container import Population


class FitnessRegistry:
    """A simple factory to fetch fitness models by name."""
    _models = {}

    @classmethod
    def get(cls, name, **params):
        if name == "purifying":
            return PurifyingFitness(**params)
        raise ValueError(f"Unknown fitness model: {name}")


class PurifyingFitness:
    def __init__(self, intensity: float = 0.1, reference_sequence=None):
        """
        :param intensity: How much each mutation hurts fitness (Selection Coefficient).
        """
        self.intensity = intensity
        self.reference_sequence = reference_sequence

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