# Exporters for FASTA, NEXUS, and Stats


import numpy as np
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    The Base Template for all Samplers.
    """
    def __init__(self, interval: int, output_path: str):
        self.interval = interval
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def is_sampling_time(self, generation: int) -> bool:
        """Standard check for all samplers."""
        return generation % self.interval == 0

    @abstractmethod
    def sample(self, population, generation: int):
        """Must be implemented by child classes."""
        pass

    def finalize(self):
        """Optional hook for end-of-simulation tasks (like plotting)."""
        pass


class StatisticsSampler(Sampler):
    """
    Records population-level metrics to a CSV file.
    """

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.history = []

    def sample(self, population, generation: int):
        """
        Calculates and stores metrics for the current generation.
        """
        matrix = population.get_matrix()

        # Calculate Genetic Diversity
        # (Average number of differences between individuals)
        # For speed, we can estimate this via the number of unique sequences
        unique_count = len(np.unique(matrix, axis=0))

        # In a real scenario, you'd pass fitness values here too
        # For now, let's just track the count and diversity
        row = {
            "generation": generation,
            "population_size": population.get_count(),
            "unique_sequences": unique_count
        }

        self.history.append(row)
        self._write_to_disk()

    def _write_to_disk(self):
        """Writes the accumulated history to a CSV file."""
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)


class FastaSampler(Sampler):
    """
    Exports population sequences to a FASTA file.
    """

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        # DNA mapping
        self.alphabet = np.array(['A', 'C', 'G', 'T'])

        # Ensure the file is empty
        with open(self.output_path, 'w') as f:
            pass  # Just create/clear the file

    def sample(self, population, generation: int):
        matrix = population.get_matrix()

        # We'll save a subset or the whole population
        # Format: >gen_[gen]_ind_[index]
        with open(self.output_path, 'a') as f:  # 'a' for append
            for i in range(len(matrix)):
                # Convert numbers to characters
                seq_str = "".join(self.alphabet[matrix[i]])
                # generation and individual index in header
                f.write(f">gen_{generation}_ind_{i}\n")
                f.write(f"{seq_str}\n")



class IdentitySampler(Sampler):
    """
    Plots/Records the average sequence identity relative to the initial sequence.
    """

    def __init__(self, interval: int, output_path: str):
        super().__init__(interval, output_path)
        self.reference_sequence = None
        self.history = []

    def sample(self, population, generation: int):
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
        # TODO we also need to do as the paper calculates identity - between all pairs
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

    def sample(self, population, generation: int):
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

    def sample(self, population, generation: int):
        matrix = population.get_matrix()
        unique_strains = len(np.unique(matrix, axis=0))

        self.history.append({
            "generation": generation,
            "unique_strains": unique_strains,
            "diversity_ratio": unique_strains / population.get_count()
        })
        pd.DataFrame(self.history).to_csv(self.output_path, index=False)

    def finalize(self):
        if not self.history: return
        df = pd.read_csv(self.output_path)
        plt.figure(figsize=(10, 5))
        plt.fill_between(df['generation'], df['unique_strains'], color='purple', alpha=0.3)
        plt.plot(df['generation'], df['unique_strains'], color='purple', linewidth=2)
        plt.title("Viral Population Diversity (Unique Strains)")
        plt.xlabel("Generation")
        plt.ylabel("Number of Unique Genotypes")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path.with_suffix('.png'))
        plt.close()