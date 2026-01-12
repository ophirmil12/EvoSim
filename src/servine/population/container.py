# The Population holds the population matrix (pop_size*genome_len), and fitness values of the population

import numpy as np
from src.servine.genome.sequence import SequenceHandler, Genome


class Population:
    """
    Manages the pool of sequences using a NumPy matrix.
    Rows = Individuals
    Columns = Nucleotide Positions
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix                            # the main object
        self.size = matrix.shape[0]                     # size of population
        self.genome_length = matrix.shape[1]            # genome fixed length
        self.last_fitness = np.ones(self.size)          # the fitness values of each organism

    @classmethod
    def create_homogeneous(cls, size: int, genome: Genome, sequence: np.ndarray = None):
        """
        Initializes a population where everyone is identical.
        :param size: Integer, number of individuals in the population.
        :param genome: Genome object defining the sequence structure.
        :param sequence: (Optional) A NumPy array of integers.
                         If None, a random sequence is generated.
        """
        if sequence is None:
            # Generate random if none provided
            master = genome.get_random_sequence()
        else:
            # Use the provided sequence
            master = np.array(sequence, dtype=np.uint8)

        # Clone the master sequence across the whole population
        matrix = SequenceHandler.create_population_matrix(size, genome.length, master)
        return cls(matrix)

    @classmethod
    def create_heterogeneous(cls, matrix: np.ndarray):
        """
        Initializes a population from a pre-calculated 2D matrix.
        """
        return cls(matrix.astype(np.uint8))     # Ensure the matrix is of type uint8

    def select(self, fitness_values: np.ndarray):
        """
        Performs Wright-Fisher selection.
        Individuals are sampled with replacement proportional to their fitness.
        """
        # Normalize fitness to create a probability distribution
        prob = fitness_values / np.sum(fitness_values).astype(float)

        # Generate indices of the individuals who will 'parent' the next generation
        indices = np.random.choice(
            np.arange(self.size),
            size=self.size,
            replace=True,
            p=prob
        )

        # Update the matrix: This is a 'view' or 'fancy indexing' copy
        self.matrix = self.matrix[indices]

        # Update last fitness
        self.last_fitness = fitness_values[indices]
        return indices      # Return indices for ancestry tracking

    def get_count(self) -> int:
        return self.size

    def get_matrix(self) -> np.ndarray:
        return self.matrix