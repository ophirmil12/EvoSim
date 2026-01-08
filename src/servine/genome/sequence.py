# Sequence representation (NumPy arrays)


import numpy as np


class Genome:
    """
    Defines the structure and alphabet of the genetic material.
    """
    ALPHABET = np.array(['A', 'C', 'G', 'T'])

    def __init__(self, length: int):
        self.length = length
        # Map: 0:A, 1:C, 2:G, 3:T
        self.alphabet = Genome.ALPHABET

    def get_random_sequence(self) -> np.ndarray:
        """Generates a random initial sequence as an array of integers."""
        return np.random.randint(0, 4, size=self.length, dtype=np.uint8)

    def to_string(self, sequence_array: np.ndarray) -> str:
        """Converts a NumPy integer array back into a DNA string."""
        return "".join(self.alphabet[sequence_array])

    @staticmethod
    def get_alphabet():
        return Genome.ALPHABET


class SequenceHandler:
    """
    Utility to handle bulk operations on sequences.
    Using a 2D NumPy array [Population_Size, Genome_Length].
    """

    @staticmethod
    def create_population_matrix(pop_size: int, length: int, master_sequence=None):
        """
        Creates a 2D matrix representing the whole population.
        If master_sequence is provided, the whole population starts identical.
        """
        if master_sequence is None:
            # Random starting population
            return np.random.randint(0, 4, size=(pop_size, length), dtype=np.uint8)
        else:
            # Everyone starts as a clone of the master
            return np.tile(master_sequence, (pop_size, 1))