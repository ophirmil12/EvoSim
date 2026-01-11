# Mutation models - how to mutate the genomes

import numpy as np
from abc import ABC, abstractmethod


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

    def __init__(self, rate: float):
        super().__init__(rate)

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

    def __init__(self, rate: float, transition_bias: float = 2.0):
        super().__init__(rate)
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