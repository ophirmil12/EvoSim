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
        # The IDs of current individuals (relative to the last sample)
        self.current_lineage_ids = list(range(initial_size))

        self.next_id = initial_size

        # A fast lookup for backtracking (Child ID -> Parent ID)
        self.parent_map = {i: -1 for i in range(initial_size)}

    def record_intermediate_step(self, parent_indices: np.ndarray):
        """
        Updates the current lineage tracking.
        Instead of creating new IDs, we just track which 'ancestor ID'
        from the last sample point each current slot belongs to.
        """
        # Collapse: new slot i now points to the ancestor that the parent at parent_indices[i] pointed to
        self.current_lineage_ids = [self.current_lineage_ids[p] for p in parent_indices]

    def sample(self, population, generation: int, **kwargs):
        """
        Called only at intervals. Records the 'jump' from the last sample point.
        """
        new_ids = []
        for ancestor_id in self.current_lineage_ids:
            node_id = self.next_id

            self.ancestry.append({
                "id": node_id,
                "parent_id": ancestor_id,
                "generation": generation
            })

            self.parent_map[node_id] = ancestor_id
            new_ids.append(node_id)
            self.next_id += 1

        # Reset the trackers: These new IDs are now the 'roots' for the next interval
        self.current_lineage_ids = new_ids

        # We return the new IDs so the Simulator can update current_individual_ids
        return new_ids

    def get_ancestor(self, current_id: int, generations_back: int) -> int:
        """Walks up the tree N steps to find the ancestor ID."""
        curr = current_id
        for _ in range(generations_back):
            # If we hit the root (-1) or a missing link, stop
            if curr not in self.parent_map or self.parent_map[curr] == -1:
                break
            curr = self.parent_map[curr]
        return curr

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
                if curr == -1: break
                active_ids.add(curr)

        return {
            "df": df,
            "parent_map": parent_map,
            "id_to_gen": id_to_gen,
            "active_ids": active_ids,
            "survivors": survivors,
            "last_gen": last_gen,
            #"root_id": min(active_ids) if active_ids else -1
            "root_ids": [i for i in df['id'] if parent_map[i] == -1]  # Find all roots, for plotting also extinct lineages
        }

    def _save_newick_tree(self, data):
        """Constructs and saves the FULL Newick string (including extinct branches)."""
        import sys
        sys.setrecursionlimit(max(3000, len(data['df'])))

        def build_node(node_id):
            # INCLUDE ALL CHILDREN (Survivors + Extinct)
            children = [cid for cid, pid in data['parent_map'].items() if pid == node_id]

            if not children:
                return f"Ind_{node_id}"

            child_parts = []
            for cid in children:
                # Branch length = Time passed
                dist = data['id_to_gen'].get(cid, 0) - data['id_to_gen'].get(node_id, 0)
                child_parts.append(f"{build_node(cid)}:{max(1, dist)}")

            return f"({','.join(child_parts)})Gen{data['id_to_gen'].get(node_id, 0)}"

        try:
            # If multiple roots exist, wrap them in a pseudo-root or pick the first
            # For simplicity, we create a list of trees if there are multiple roots
            # or just map the primary root.
            # Here we assume a single coherent population or pick the first root found.
            root = min(data['active_ids']) if data['active_ids'] else data['df']['id'].min()

            # Better approach: Find the actual starting nodes (Generation 0 or 1)
            # Since our parent_map[root] == -1 or missing

            nwk = build_node(root) + ";"
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
        Plots the COMPLETE tree using a topological layout (DFS).
        Fixes the 'flat line' bug by correctly identifying roots that
        are outside the dataframe (e.g., Gen 0).
        """
        df = data['df']

        # 1. Dynamic Figure Size
        n_leaves = len(df) - len(data['parent_map'])
        fig_height = max(10, n_leaves * 0.15)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        # 2. Build Adjacency List & Identify Roots
        children_map = {}
        all_recorded_ids = set(df['id'])
        start_nodes = set()  # Nodes that act as roots for our drawing

        for _, row in df.iterrows():
            pid = row['parent_id']
            cid = row['id']

            # Build tree structure
            if pid not in children_map:
                children_map[pid] = []
            children_map[pid].append(cid)

            # If the parent isn't in our recorded IDs, it's a Root of this tree
            if pid not in all_recorded_ids:
                start_nodes.add(pid)

        # 3. Topological Layout (DFS)
        layout_state = {'next_y': 0, 'node_y': {}}
        active_ids = data['active_ids']

        def compute_y_positions(node_id):
            # Base Case: Leaf (no known children)
            if node_id not in children_map:
                y = layout_state['next_y']
                layout_state['node_y'][node_id] = y
                layout_state['next_y'] += 1
                return y

            # Recursive Step: Visit children
            child_ys = []
            children = children_map[node_id]

            # Sort: Active lineages last (bottom/center) to group them
            sorted_children = sorted(children, key=lambda x: (x in active_ids, x))

            for child in sorted_children:
                child_ys.append(compute_y_positions(child))

            # Parent Y is average of children
            my_y = np.mean(child_ys)
            layout_state['node_y'][node_id] = my_y
            return my_y

        # Run Layout starting from the invisible Gen 0 roots
        for root in sorted(list(start_nodes)):
            compute_y_positions(root)
            layout_state['next_y'] += 1  # Gap between trees

        node_y = layout_state['node_y']

        # 4. Draw Branches
        cmap = plt.get_cmap('viridis')
        max_gen = data['last_gen']
        id_to_gen = data['id_to_gen']

        for _, row in df.iterrows():
            node_id = row['id']
            parent_id = row['parent_id']

            # Safety: Ensure we have coordinates for both
            if parent_id not in node_y or node_id not in node_y:
                continue

            # X-Coords: Use 0 for the invisible root parents if missing
            px = id_to_gen.get(parent_id, 0)
            cx = row['generation']

            # Y-Coords
            py = node_y[parent_id]
            cy = node_y[node_id]

            x_coords = [px, cx]
            y_coords = [py, cy]

            # Styling
            if node_id in active_ids:
                color = cmap(row['generation'] / max_gen)
                alpha = 0.9
                linewidth = 1.5
                zorder = 10
            else:
                color = 'grey'
                alpha = 0.3
                linewidth = 0.5
                zorder = 1

            ax.plot(x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

        ax.set_title("Evolutionary History: Topological View")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Lineage Space (Sorted)")
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.invert_yaxis()

        # Legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=cmap(0.8), lw=2, label='Surviving Lineage'),
            Line2D([0], [0], color='grey', lw=1, label='Extinct Lineage')
        ]
        ax.legend(handles=custom_lines, loc='upper left')

        output_file = str(self.output_path).replace(".csv", "") + "_tree.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)