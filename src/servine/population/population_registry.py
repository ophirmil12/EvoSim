import numpy as np

from src.servine.color import fg
from src.servine.population.container import Population


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
