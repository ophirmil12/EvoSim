import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.santa.io.sampler import Sampler


class TreeRecorder(Sampler):
    """
    A type of Sampler that Tracks the ancestry of every individual
     to reconstruct evolutionary history.
    """

    def __init__(self, interval: int, output_path: str, initial_size: int):
        super().__init__(interval, output_path)
        self.ancestry = []
        # Initialize IDs for the starting population (Generation 0)
        self.last_gen_ids = list(range(initial_size))
        self.next_id = initial_size

    def sample(self, population, generation: int):
        """
        The standard sample method is required by the base class.
        For TreeRecorder, we primarily use the custom record_generation method.
        """
        pass

    def record_generation(self, generation: int, parent_indices: np.ndarray):
        """
        Maps current individuals to their parents from the previous generation.
        This is called by the Simulator after selection.
        """
        current_gen_ids = []

        for p_idx in parent_indices:
            node_id = self.next_id
            # Identify which ID from the previous generation was the parent
            parent_id = self.last_gen_ids[p_idx]

            self.ancestry.append({
                "id": node_id,
                "parent_id": parent_id,
                "generation": generation
            })
            current_gen_ids.append(node_id)
            self.next_id += 1

        # Move current IDs to last_gen_ids for the next loop
        self.last_gen_ids = current_gen_ids

    def finalize(self):
        if not self.ancestry:
            return

        # 1. Save the raw data
        df = pd.DataFrame(self.ancestry)
        df.to_csv(self.output_path, index=False)

        # 2. Identify final survivors
        last_gen = df['generation'].max()
        survivor_ids = df[df['generation'] == last_gen]['id'].values

        # 3. Create a robust lookup for Generation
        # Map ID -> Generation (include generation 0 for initial pop)
        id_to_gen = dict(zip(df['id'], df['generation']))

        # 4. Trace back ancestors of ONLY the survivors
        parent_map = dict(zip(df['id'], df['parent_id']))

        ltt_counts = {gen: set() for gen in range(int(last_gen) + 1)}

        for s_id in survivor_ids:
            curr = s_id
            while curr in parent_map:
                # Look up the generation using our map
                # If ID < initial_size, it's Gen 0
                gen = id_to_gen.get(curr, 0)

                ltt_counts[gen].add(curr)
                curr = parent_map[curr]
                if curr == -1:
                    break

            # Don't forget to add the root (Gen 0) ancestor
            ltt_counts[0].add(curr if curr != -1 else 0)

        # 5. Plotting (Same as before)
        generations = sorted(ltt_counts.keys())
        counts = [len(ltt_counts[g]) for g in generations]

        plt.figure(figsize=(10, 6))
        plt.step(generations, counts, where='post', color='darkorange', linewidth=2)
        plt.yscale('log')
        plt.title("Lineages Through Time (LTT) - The History of Survivors")
        plt.xlabel("Generation")
        plt.ylabel("Number of Ancestral Lineages (Log Scale)")
        plt.grid(True, which="both", ls="-", alpha=0.2)

        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()