# A simple test run


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.santa.genome.sequence import Genome
from src.santa.population.container import Population
from src.santa.evolution.mutator import NucleotideMutator
from src.santa.evolution.fitness import PurifyingFitness
from src.santa.simulator import Simulator, Epoch
from src.santa.io.sampler import StatisticsSampler

def run_test():
    print("🚀 Initializing Test Run...")

    # 1. Setup Genome (1000 nucleotides)
    genome = Genome(length=1000)

    # 2. Setup Population (500 identical individuals)
    # Everyone starts as a clone of a random sequence
    initial_pop = Population.create_homogeneous(size=500, genome=genome)

    # 3. Setup Evolutionary Logic
    # High mutation rate so we see results quickly in a short test
    mutator = NucleotideMutator(rate=0.0001, transition_bias=2.0)
    fitness_model = PurifyingFitness(intensity=0.05)

    # 4. Define an Epoch
    epoch = Epoch(
        name="Test_Outbreak",
        generations=100,
        mutator=mutator,
        fitness_model=fitness_model
    )

    # 5. Setup Sampling
    sampler = StatisticsSampler(interval=10, output_path="test_results.csv")

    # 6. Initialize and Run Simulator
    sim = Simulator(
        population=initial_pop,
        epochs=[epoch],
        samplers=[sampler]
    )

    sim.run()

    # 7. Visualization
    df = pd.read_csv("test_results.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(df['generation'], df['unique_sequences'], marker='o', color='teal', linestyle='-')
    plt.title("Population Diversity Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Unique Sequences (Diversity)")
    plt.grid(True, alpha=0.3)
    plt.savefig("test_diversity_plot.png")
    print("📈 Plot saved as 'diversity_plot.png'")
    plt.show()

    print("\n✅ Simulation Complete!")
    print("Check 'test_results.csv' to see the population metrics.")

    # Quick Verification: Check if the population is still the same or changed
    final_matrix = initial_pop.get_matrix()
    unique_count = len(np.unique(final_matrix, axis=0))
    print(f"Final Unique Sequences: {unique_count} (out of 500)")

if __name__ == "__main__":
    run_test()