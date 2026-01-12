# Exporters for FASTA, Statistics, and more - collected during the simulation


import numpy as np
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from src.servine.genome.sequence import Genome


class Sampler(ABC):
    """
    The Base Template for all Samplers.
    """

    def __init__(self, interval: int, output_path: str, **kwargs):
        self.interval = interval
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def is_sampling_time(self, generation: int) -> bool:
        """Standard check for all samplers."""
        return generation % self.interval == 0

    @abstractmethod
    def sample(self, population, generation: int, **kwargs):
        """Must be implemented by child classes."""
        pass

    def finalize(self):
        """Optional hook for end-of-simulation tasks (like plotting)."""
        pass


class FastaSampler(Sampler):
    """
    Exports population sequences to a FASTA file.
    """

    def __init__(self, interval: int, output_path: str, backtrack_steps: int = 50):
        super().__init__(interval, output_path)
        # DNA mapping
        self.alphabet = Genome.ALPHABET
        self.backtrack_steps = backtrack_steps

        # Ensure the file is empty
        with open(self.output_path, 'w') as f:
            pass  # Just create/clear the file

    def sample(self, population, generation: int, ids: list = None, tree_provider=None, **kwargs):
        matrix = population.get_matrix()
        fasta_lines = []

        for i in range(len(matrix)):
            seq_str = "".join(self.alphabet[matrix[i]])
            curr_id = ids[i] if ids is not None else i

            # Ancestry Lookup Logic
            ancestor_tag = ""
            if tree_provider:
                # Ask the TreeRecorder for the ancestor N generations ago
                anc_id = tree_provider.get_ancestor(curr_id, self.backtrack_steps)
                ancestor_tag = f"|anc_gen_{max(0, generation - self.backtrack_steps)}:{anc_id}"

            # Format: >id:205|gen:100|idx:5|anc_gen_50:102
            header = f">id:{curr_id}|gen:{generation}|idx:{i}{ancestor_tag}\n"
            fasta_lines.append(header)
            fasta_lines.append(f"{seq_str}\n")

        # Open file once and write everything
        with open(self.output_path, 'a') as f:
            f.writelines(fasta_lines)

    def finalize(self):
        """A hook for any finalization if needed."""
        pass


class IdentitySampler(Sampler):
    """
    Plots/Records the average sequence identity relative to the initial sequence.
    """

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.reference_sequence = None
        self.history = []

    def sample(self, population, generation: int, **kwargs):
        matrix = population.get_matrix()

        # Capture the initial sequence from the first generation sampled
        if self.reference_sequence is None:
            self.reference_sequence = matrix[0].copy()

        # Calculate Identity:
        # 1. Compare entire matrix to reference (Broadcasting)
        # 2. Count matches (True = 1, False = 0)
        # 3. Average across genome length and then across population
        matches = (matrix == self.reference_sequence)
        identity_per_ind = np.mean(matches, axis=1)  # 1.0 = 100% match
        avg_identity = np.mean(identity_per_ind)

        self.history.append({
            "generation": generation,
            "avg_identity": avg_identity
        })
        self._write_to_disk()

    def _write_to_disk(self):
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["generation", "avg_identity"])
            writer.writeheader()
            writer.writerows(self.history)

    def finalize(self):
        """
        Called once at the end of the simulation to generate the plot.
        """
        if not self.history:
            return

        # Load the data we just wrote
        df = pd.read_csv(self.output_path)

        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'], df['avg_identity'], color='red', linewidth=2)

        plt.title("Sequence Identity to Initial Sequence Over Time")
        plt.xlabel("Generation")
        plt.ylabel("Avg Identity (Mean) to Initial Sequence")
        plt.ylim(0, 1.05)  # Keeps scale consistent
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot to same folder with .png extension
        plot_path = self.output_path.with_suffix('.png')
        plt.savefig(plot_path)
        plt.close()  # Close to free up memory


class FitnessSampler(Sampler):
    """Tracks Average Fitness (Genetic Health) over time."""

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.history = []

    def sample(self, population, generation: int, **kwargs):
        # Capturing fitness from the population object
        fitness_vals = getattr(population, 'last_fitness', np.array([1.0]))
        avg_fit = fitness_vals.mean()

        self.history.append({"generation": generation, "avg_fitness": avg_fit})
        pd.DataFrame(self.history).to_csv(self.output_path, index=False)

    def finalize(self):
        if not self.history: return
        df = pd.read_csv(self.output_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df['generation'], df['avg_fitness'], color='green', linewidth=2)
        plt.title("Population Genetic Health (Avg Fitness)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()


class DiversitySampler(Sampler):
    """Tracks Population Diversity (Unique Genotypes) over time."""

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.history = []

    def sample(self, population, generation: int, **kwargs):
        matrix = population.get_matrix()
        unique_strains = len(np.unique(matrix, axis=0))
        pop_count = population.get_count()

        self.history.append({
            "generation": generation,
            "unique_strains": unique_strains,
            "population_size": pop_count,
            "diversity_ratio": unique_strains / pop_count
        })
        pd.DataFrame(self.history).to_csv(self.output_path, index=False)

    def finalize(self):
        if not self.history: return
        df = pd.read_csv(self.output_path)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        # Primary Axis: Absolute Unique Genotypes
        color_strains = 'tab:purple'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Unique Genotypes', color=color_strains, fontweight='bold')
        ax1.fill_between(df['generation'], df['unique_strains'], color=color_strains, alpha=0.2)
        ax1.plot(df['generation'], df['unique_strains'], color=color_strains, linewidth=2, label='Unique Strains')
        ax1.tick_params(axis='y', labelcolor=color_strains)
        ax1.grid(True, alpha=0.3, linestyle='--')
        # Secondary Axis: Diversity Ratio (Normalized)
        ax2 = ax1.twinx()
        color_ratio = 'tab:cyan'
        ax2.set_ylabel('Diversity Ratio (Unique/Total)', color=color_ratio, fontweight='bold')
        ax2.plot(df['generation'], df['diversity_ratio'], color=color_ratio, linestyle='--', linewidth=1.5,
                 label='Diversity Ratio')
        ax2.tick_params(axis='y', labelcolor=color_ratio)
        ax2.set_ylim(0, 1.05)  # Ratio is always between 0 and 1
        # Title and Layout
        plt.title("Viral Population Diversity Dynamics", fontsize=14)
        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_path.with_suffix('.png'), dpi=300)
        plt.close()


class PairwiseIdentitySampler(Sampler):
    """Calculates Average Pairwise Distance using the Jukes-Cantor (JC69) model."""

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.history = []

    def sample(self, population, generation: int, **kwargs):
        matrix = population.get_matrix()
        # Sub-sample to maintain performance
        n = min(50, len(matrix))
        idx = np.random.choice(len(matrix), n, replace=False)
        sub_matrix = matrix[idx]

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                # 1. Calculate p (proportion of differing sites)
                p = np.mean(sub_matrix[i] != sub_matrix[j])

                # 2. Apply JC69 formula
                # We cap p at 0.75 to avoid log(0) or log(negative)
                if p < 0.75:
                    d = -0.75 * np.log(1 - (4 / 3) * p)
                else:
                    d = 3.0  # Maximum theoretical distance for JC69
                distances.append(d)

        avg_distance = np.mean(distances) if distances else 0.0
        self.history.append({
            "generation": generation,
            "avg_jc_distance": avg_distance
        })
        pd.DataFrame(self.history).to_csv(self.output_path, index=False)

    def finalize(self):
        if not self.history: return
        df = pd.read_csv(self.output_path)
        plt.figure(figsize=(10, 5))
        plt.plot(df['generation'], df['avg_jc_distance'], color='blue', linewidth=2)

        plt.title("Population Genetic Divergence (Jukes-Cantor Distance)")
        plt.xlabel("Generation")
        plt.ylabel("Estimated Substitutions per Site")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()


class HaplotypeFrequencySampler(Sampler):
    """Tracks the frequency of the most common genotypes (Haplotypes)."""

    def __init__(self, interval: int, output_path: str, top_n: int = 10):
        super().__init__(interval, output_path)
        self.history = []
        self.top_n = top_n

    def sample(self, population, generation: int, **kwargs):
        matrix = population.get_matrix()
        # Identify unique sequences and their counts
        unique_seqs, counts = np.unique(matrix, axis=0, return_counts=True)
        # Calculate frequencies and sort descending
        freqs = np.sort(counts / len(matrix))[::-1]

        row = {"generation": generation}
        for i in range(self.top_n):
            # Record the frequency of the i-th most common strain
            row[f"haplo_{i}"] = freqs[i] if i < len(freqs) else 0.0

        self.history.append(row)
        pd.DataFrame(self.history).to_csv(self.output_path, index=False)

    def finalize(self):
        if not self.history: return
        df = pd.read_csv(self.output_path).fillna(0)

        plt.figure(figsize=(12, 6))
        # Create a stackplot: shows how lineages compete for dominance
        y = [df[f"haplo_{i}"] for i in range(self.top_n)]
        plt.stackplot(df['generation'], y, labels=[f"Strain {i + 1}" for i in range(self.top_n)])

        plt.title("Haplotype Frequency Dynamics (Selective Sweeps)")
        plt.xlabel("Generation")
        plt.ylabel("Cumulative Frequency")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()


class InitialAlleleFrequencySampler(Sampler):
    """
    Tracks the frequency of only the initial unique genotypes over time.
    """

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.tracked_alleles = None
        self.history = []

    def sample(self, population, generation: int, **kwargs):
        matrix = population.get_matrix()

        # Capture the unique genotypes at the very first sampling point
        if self.tracked_alleles is None:
            self.tracked_alleles = np.unique(matrix, axis=0)
        total_pop = len(matrix)

        # For each initial allele, count how many individuals still carry it
        for i, allele in enumerate(self.tracked_alleles):
            # Find rows that exactly match the initial sequence
            matches = np.all(matrix == allele, axis=1)
            count = np.count_nonzero(matches)
            self.history.append({
                "generation": generation,
                "allele_id": f"Initial_{i}",
                "frequency": count / total_pop
            })

    def finalize(self):
        if not self.history:
            return

        # Prepare data for plotting
        df = pd.DataFrame(self.history)
        pivot_df = df.pivot(index="generation", columns="allele_id", values="frequency").fillna(0)

        plt.figure(figsize=(10, 6))
        # Plot each initial allele as a separate line
        for col in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[col], linewidth=2, alpha=0.8, label=col)

        plt.title("Frequency of Initial Alleles Over Time")
        plt.xlabel("Generation")
        plt.ylabel("Frequency")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.legend(title="Allele IDs", loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small')
        plt.tight_layout()

        # Save the plot
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()
