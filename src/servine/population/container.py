import numpy as np
from src.servine.genome.sequence import SequenceHandler, Genome
from src.servine.color import fg


class Population:
    """
    Manages the pool of sequences using a NumPy matrix.
    Rows = Individuals
    Columns = Nucleotide Positions
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix                            # the main object
        self.size = matrix.shape[0]                     # size of population TODO make it not fixed? no idea how
        self.genome_length = matrix.shape[1]            # genome fixed length TODO make it not fixed? no idea how
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
            size=self.size,     # TODO: Make size dynamic based on population control strategies?
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


class PopulationRegistry:
    """Registry for a Population object"""
    @classmethod
    def get(cls, conf, initial_seq, genome, **params):
        # Inside main() in cli.py
        pop_conf = conf['population']
        initial_size = pop_conf.get('initial_size', 50)
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        pop = None

        # CASE: Dictionary Distribution
        if 'initial_distribution' in pop_conf:
            dist = pop_conf['initial_distribution']
            all_seqs = []

            for seq_str, count in dist.items():
                # Convert string to numpy array
                seq_array = np.array([mapping[base.upper()] for base in seq_str], dtype=np.uint8)
                # Add 'count' copies of this array to our list
                for _ in range(count):
                    all_seqs.append(seq_array)

            # Safety Check: If the counts don't add up to initial_size,
            # either truncate or pad with the last sequence type
            if len(all_seqs) != initial_size:
                print(fg.YELLOW, f"Warning: Distribution total ({len(all_seqs)}) != initial_size ({initial_size}). Adjusting...", fg.RESET)
                if len(all_seqs) > initial_size:
                    all_seqs = all_seqs[:initial_size]
                else:
                    while len(all_seqs) < initial_size:
                        all_seqs.append(all_seqs[-1])

            matrix = np.array(all_seqs)
            pop = Population.create_heterogeneous(matrix)

        # FALLBACK: Single sequence (the old way)
        elif 'initial_sequence' in pop_conf:
            pop = Population.create_homogeneous(
                size=conf['population']['initial_size'],
                genome=genome,
                sequence=initial_seq
            )

        return pop