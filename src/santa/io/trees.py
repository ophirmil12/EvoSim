import pandas as pd
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
        """Main entry point for post-simulation analysis."""
        if not self.ancestry:
            return

        df = pd.DataFrame(self.ancestry)
        df.to_csv(self.output_path, index=False)

        # 1. Prepare data structures for efficient lookup
        analysis_data = self._prepare_analysis_data(df)

        # 2. Generate the Newick file (The "Paper" Tree)
        self._save_newick_tree(analysis_data)

        # 3. Generate the LTT Plot (Temporal Dynamics)
        self._save_ltt_plot(analysis_data)

        # 4. Generate the Tree Plot (Top-Down View)
        self._save_tree_plot(analysis_data)

    def _prepare_analysis_data(self, df):
        """Filters the ancestry to only include lineages of final survivors."""
        last_gen = df['generation'].max()
        survivors = set(df[df['generation'] == last_gen]['id'])

        parent_map = dict(zip(df['id'], df['parent_id']))
        id_to_gen = dict(zip(df['id'], df['generation']))

        # Trace upward from survivors to find the 'Active' skeleton
        active_ids = set(survivors)
        for s_id in survivors:
            curr = s_id
            while curr in parent_map:
                curr = parent_map[curr]
                active_ids.add(curr)
                if curr == -1: break

        return {
            "df": df,
            "parent_map": parent_map,
            "id_to_gen": id_to_gen,
            "active_ids": active_ids,
            "survivors": survivors,
            "last_gen": last_gen,
            "root_id": min(active_ids) if active_ids else -1
        }

    def _save_newick_tree(self, data):
        """Constructs and saves the pruned Newick string."""
        import sys
        sys.setrecursionlimit(max(2000, len(data['df'])))

        def build_node(node_id):
            # Only children that are part of a surviving lineage
            children = [cid for cid, pid in data['parent_map'].items()
                        if pid == node_id and cid in data['active_ids']]

            if not children:
                return f"Ind_{node_id}"

            child_parts = []
            for cid in children:
                # Branch length = Time passed
                dist = data['id_to_gen'].get(cid, 0) - data['id_to_gen'].get(node_id, 0)
                child_parts.append(f"{build_node(cid)}:{max(1, dist)}")

            return f"({','.join(child_parts)})Gen{data['id_to_gen'].get(node_id, 0)}"

        try:
            nwk = build_node(data['root_id']) + ";"
            with open(self.output_path.with_suffix('.nwk'), 'w') as f:
                f.write(nwk)
        except Exception as e:
            print(f"Failed to build Newick: {e}")

    def _save_ltt_plot(self, data):
        """Generates the Lineages Through Time visualization."""
        ltt_counts = {gen: set() for gen in range(int(data['last_gen']) + 1)}

        for s_id in data['survivors']:
            curr = s_id
            while curr in data['parent_map']:
                gen = data['id_to_gen'].get(curr, 0)
                ltt_counts[gen].add(curr)
                curr = data['parent_map'][curr]
                if curr == -1: break
            ltt_counts[0].add(curr if curr != -1 else 0)

        gens = sorted(ltt_counts.keys())
        counts = [len(ltt_counts[g]) for g in gens]

        plt.figure(figsize=(10, 6))
        plt.step(gens, counts, where='post', color='darkorange', linewidth=2)
        plt.yscale('log')
        plt.title("Lineages Through Time (History of Survivors)")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()

    def _save_tree_plot(self, data):
        """
        Step 4: Plots the pruned tree, coloring nodes/branches by generation.
        Explicitly creates fig and ax to solve the Colorbar ValueError.
        """
        # Initialize fig and ax
        fig, ax = plt.subplots(figsize=(12, 12))

        # 1. Map each survivor to a vertical position (0 to N)
        sorted_survivors = sorted(list(data['survivors']))
        y_positions = {s_id: i for i, s_id in enumerate(sorted_survivors)}

        # 2. Recursively calculate Y positions for internal nodes
        parent_map = data['parent_map']
        id_to_gen = data['id_to_gen']
        active_ids = data['active_ids']
        node_y = y_positions.copy()

        # Iterate backwards from late generations to early ones
        for node_id in sorted(list(active_ids), reverse=True):
            if node_id not in node_y:
                children = [cid for cid, pid in parent_map.items()
                            if pid == node_id and cid in active_ids]
                if children:
                    node_y[node_id] = np.mean([node_y.get(cid, 0) for cid in children])
                else:
                    node_y[node_id] = 0

        # 3. Draw the branches
        cmap = plt.get_cmap('viridis')
        max_gen = data['last_gen']

        for node_id in data['active_ids']:
            if node_id in data['parent_map']:
                parent_id = data['parent_map'][node_id]
                if parent_id in data['active_ids']:
                    # X = Generation, Y = Calculated vertical position
                    x_coords = [id_to_gen.get(parent_id, 0), id_to_gen.get(node_id, 0)]
                    y_coords = [node_y[parent_id], node_y[node_id]]

                    # Color intensity based on generation progress
                    color = cmap(id_to_gen.get(node_id, 0) / max_gen)
                    ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=1)

        ax.set_title("Phylogenetic Tree of Survivors (Colored by Generation)")
        ax.set_xlabel("Generation (Time)")
        ax.set_ylabel("Lineage Space (Survivors)")
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_gen))
        fig.colorbar(sm, ax=ax, label="Generation")

        output_file = str(self.output_path).replace(".csv", "") + "_tree.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)