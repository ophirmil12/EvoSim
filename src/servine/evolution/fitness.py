# Fitness landscapes (Purifying, Epistatic...)

from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

from ..population.container import Population

# TODO: for all the models - compare to the SANTA-SIM git to check that the math is correct(!!!)
class FitnessModel(ABC):
    """
    The Base Template for all Fitness Landscapes.
    """
    def __init__(self, reference_sequence=None, **kwargs):
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


# TODO: the purifying models need some testing and refinement
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
        self.update_reference_to_consensus(population)

        # 1. Count mutations relative to reference for every row
        # matrix != reference_sequence uses NumPy broadcasting
        mutation_counts = np.sum(matrix != self.reference_sequence, axis=1)

        # 2. Calculate fitness: w = (1 - s)^n
        # where s is intensity and n is number of mutations

        # fitness_scores = np.power(1.0 - self.intensity, mutation_counts)
        fitness_scores = np.exp(-self.intensity*mutation_counts)

        # Ensure fitness never hits exactly zero to avoid math errors in selection
        return np.maximum(fitness_scores, 1e-10)

    def update_reference_to_consensus(self, population: Population):
        """Sets the reference to the most common nucleotide at each site."""
        matrix = population.get_matrix()
        # Find the mode (most frequent value) for each column
        self.reference_sequence = np.array([np.bincount(matrix[:, col], minlength=4).argmax() for col in range(matrix.shape[1])])

        # axis=0 means operate on columns
        # consensus, _ = stats.mode(matrix, axis=0)
        # self.reference_sequence = consensus.flatten()


# TODO maybe create a sub-SiteSpecificPurifyingFitness that consider the 3-codon redundancy
#  (ignore the mutations in third letters in fitness calculations)
class SiteSpecificPurifyingFitness(FitnessModel):
    """
    By defining an array of "site_intensities", we can control on how a mutation
    in a specific site cause decrease in fitness.
    Higher "site_intensity" -> lower fitness when mutation caused.
    Note that a reference sequence is required (the most fit variant).
    """
    def __init__(self, site_intensities: np.ndarray, reference_sequence: np.ndarray):
        super().__init__(reference_sequence)
        # site_intensities is an array of length L (e.g., [0.5, 0.01, 0.01, 0.9...])
        self.site_intensities = site_intensities

    def evaluate_population(self, population: Population) -> np.ndarray:
        matrix = population.get_matrix()

        # 1. Find where mutations are (Boolean matrix)
        mutations = (matrix != self.reference_sequence)

        # 2. Multiply each mutation by its specific cost
        # We use log-space to handle the product (1-s1)*(1-s2)... efficiently
        weighted_mutation_costs = mutations * np.log(1.0 - self.site_intensities)

        # 3. Sum logs and exponentiate to get final fitness
        fitness_scores = np.exp(np.sum(weighted_mutation_costs, axis=1))

        return np.maximum(fitness_scores, 1e-10)


class EpistaticFitness(FitnessModel):
    """
    Fitness model with pairwise epistatic interactions.
    Note: an L*L matrix is required as input.
    """
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
        mut_mask = (matrix != self.reference_sequence)

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
    This models a changing environment.
    Every X generations, the "optimal" sequence changes,
        and the population must catch up or die.
    Note: Inherits from PurifyingFitness.
    """
    def __init__(self, intensity: float, update_interval: int, mutator, **kwargs):
        # TODO make a way to send a specific mutator to this model, not the general mutator
        #  (here we might want higher mutation rate than the entire population)
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


