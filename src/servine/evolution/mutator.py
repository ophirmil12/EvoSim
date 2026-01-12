# Mutation models - how to mutate the genomes

import numpy as np
from abc import ABC, abstractmethod

from numpy.lib.stride_tricks import sliding_window_view


class Mutator(ABC):
    """
    Abstract Base Class for all mutation models.
    Ensures that any new mutator (Nucleotide, AminoAcid, etc.)
    implements the 'apply' method.
    """

    def __init__(self, rate: float):
        self.rate = rate

    @abstractmethod
    def apply(self, population):
        pass


class UnifyMutator(Mutator):
    """
    Simple mutator - equal chance for each mutation,
    and rate determined by user.
    Assumes alphabet size k = 4 (e.g. A, C, G, T).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, population):
        matrix = population.get_matrix()

        # Decide which sites mutate
        mutation_mask = np.random.random(matrix.shape) < self.rate
        num_mutations = np.count_nonzero(mutation_mask)
        if num_mutations == 0:
            return

        current = matrix[mutation_mask]

        # Pick uniformly from the other 3 nucleotides
        # offset ∈ {1,2,3} guarantees new != old
        offsets = np.random.randint(1, 4, size=num_mutations)
        new_vals = (current + offsets) % 4

        matrix[mutation_mask] = new_vals


class NucleotideMutator(Mutator):
    """
    Simple mutator that gets a rate of mutation, and a bias for
    transitions/transversions (transversions are rarer).
    """

    def __init__(self, transition_bias: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.transition_bias = transition_bias

        # 0:A, 1:C, 2:G, 3:T
        # We define a transition map: A->G (0->2), C->T (1->3), etc.
        self.transitions = {0: 2, 1: 3, 2: 0, 3: 1}

    def apply(self, population):
        matrix = population.get_matrix()
        mutation_mask = np.random.random(matrix.shape) < self.rate
        num_mutations = np.count_nonzero(mutation_mask)

        if num_mutations == 0:
            return

        # Get the current nucleotides that are about to mutate
        current_nucs = matrix[mutation_mask]

        # Calculate probabilities:
        # There is 1 possible transition and 2 possible transversions for any base.
        # Prob(Transition) = bias / (bias + 2)
        prob_transition = self.transition_bias / (self.transition_bias + 2.0)

        # Decide for each mutation if it's a Ts or Tv
        is_transition = np.random.random(num_mutations) < prob_transition

        new_nucs = np.zeros(num_mutations, dtype=np.uint8)

        # 1. Handle Transitions (A<->G, C<->T)
        # Vectorized lookup using a small helper array
        ts_lookup = np.array([2, 3, 0, 1])  # Index corresponds to nucleotide value
        new_nucs[is_transition] = ts_lookup[current_nucs[is_transition]]

        # 2. Handle Transversions (e.g., A -> C or T)
        # For transversions, we pick randomly from the two remaining options

        tv_mask = ~is_transition
        num_tvs = np.count_nonzero(tv_mask)
        if num_tvs > 0:
            tv_offsets = np.random.choice([1, 3], size=num_tvs)
            new_nucs[tv_mask] = (current_nucs[tv_mask] + tv_offsets) % 4

        matrix[mutation_mask] = new_nucs


class HotColdMutator(Mutator):
    def __init__(self, variable_kmers: list, k_high: float,
                 conserved_kmers: list, k_low: float, threshold: float = 0.8,
                 transition_bias: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.high_var_kmers = [np.array(km, dtype=np.uint8) for km in variable_kmers]
        self.k_high = k_high
        self.preserved_kmers = [np.array(km, dtype=np.uint8) for km in conserved_kmers]
        self.k_low = k_low
        self.threshold = threshold
        self.transition_bias = transition_bias
        self.ts_lookup = np.array([2, 3, 0, 1], dtype=np.uint8)

    def _get_rate_mask(self, matrix):
        pop_size, genome_len = matrix.shape
        mask = np.ones((pop_size, genome_len), dtype=np.float32)

        # print(f"first sequence {matrix[0]}")

        def find_kmers(kmers, factor):
            for kmer in kmers:
                # print(f"Searching for kmer: {kmer}")
                k_len = len(kmer)
                min_matches = int(np.ceil(self.threshold * k_len))

                # Create a sliding window view: shape (pop_size, num_windows, k_len)
                windows = sliding_window_view(matrix, window_shape=k_len, axis=1)

                # Compare all windows to the kmer at once
                # matches shape: (pop_size, num_windows)
                matches = np.sum(windows == kmer, axis=2) >= min_matches

                # Find indices where matches occur
                rows, start_cols = np.where(matches)

                # Apply the factor to the mask for the entire span of the kmer
                for offset in range(k_len):
                    mask[rows, start_cols + offset] = factor

        # Apply variability first, then preservation (priority)
        find_kmers(self.high_var_kmers, self.k_high)
        find_kmers(self.preserved_kmers, self.k_low)

        # print(f"mask after factor: {mask}")
        return mask

    def apply(self, population):
        matrix = population.get_matrix()

        # Vectorized Rate Calculation
        rate_matrix = self._get_rate_mask(matrix) * self.rate

        # Standard Mutation Logic
        mutation_mask = np.random.random(matrix.shape) < rate_matrix
        num_mutations = np.count_nonzero(mutation_mask)
        if num_mutations == 0: return

        current_nucs = matrix[mutation_mask]
        prob_ts = self.transition_bias / (self.transition_bias + 2.0)
        is_transition = np.random.random(num_mutations) < prob_ts

        new_nucs = np.zeros(num_mutations, dtype=np.uint8)
        new_nucs[is_transition] = self.ts_lookup[current_nucs[is_transition]]

        tv_indices = np.where(~is_transition)[0]
        if len(tv_indices) > 0:
            # Vectorized transversion choice helper
            tv_options = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])  # Options for nucs 0,1,2,3
            choices = np.random.randint(0, 2, size=len(tv_indices))
            new_nucs[tv_indices] = tv_options[current_nucs[tv_indices], choices]

        matrix[mutation_mask] = new_nucs