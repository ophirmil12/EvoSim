import argparse
import numpy as np
import yaml
import sys

# Internal imports
from src.santa.simulator import Simulator, Epoch
from src.santa.genome.sequence import Genome
from src.santa.population.container import Population
from src.santa.evolution.mutator import NucleotideMutator
from src.santa.evolution.fitness import FitnessRegistry
from src.santa.io.sampler import StatisticsSampler, FastaSampler, IdentitySampler, FitnessSampler, DiversitySampler

"""
How the components interact:
1. cli.py: Takes your config.yaml and assembles the machine.
2. simulator.py: Runs the clock.
3. population.py: Holds the "DNA Matrix."
4. mutator.py: Uses a random mask to flip bits in that matrix.
5. fitness.py: Scores the rows of the matrix.
6. sampler.py: Records the results to results.csv.
"""

SAMPLER_MAP = {
    'stats': StatisticsSampler,
    'fasta': FastaSampler,
    'identity': IdentitySampler,
    'fitness': FitnessSampler,
    'diversity': DiversitySampler
}

def main():
    """
    Run with:
         <.../EvoSim> $env:PYTHONPATH = "src"
         <.../EvoSim> py -m santa.cli src/config.yaml
    in the command line (checked working in Windows PowerShell).
    """
    # TODO better name and description
    parser = argparse.ArgumentParser(description="SANTA-SIM Python: Evolutionary Simulator")

    # Simple positional argument for the config file
    parser.add_argument("config", help="Path to the YAML configuration file")

    args = parser.parse_args()

    # 1. Load Configuration
    try:
        with open(args.config, 'r') as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Could not read config file. {e}")
        sys.exit(1)

    # 2. Initialize Core Components
    genome = Genome(length=conf['genome']['length'])

    initial_seq = None
    if 'initial_sequence' in conf['population']:
        seq_string = conf['population']['initial_sequence'].upper()
        if len(seq_string) != genome.length:
            print("Error: Initial sequence length does not match genome length.")
            sys.exit(1)

        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        try:
            initial_seq = np.array([mapping[base] for base in seq_string], dtype=np.uint8)
        except KeyError as e:
            print(f"Error: Invalid nucleotide in initial sequence: {e}")
            sys.exit(1)

    pop = Population.create_homogeneous(
        size=conf['population']['initial_size'],
        genome=genome,
        sequence=initial_seq
    )

    # 3. Setup Epochs
    epochs = []
    for e_conf in conf['epochs']:
        # Initialize Mutator
        epoch_mutator = NucleotideMutator(rate=float(e_conf['mutator']['rate']))

        # Setup Fitness Model
        fitness_params = e_conf['fitness'].get('params', {}).copy()  # Use copy to avoid mutation issues
        # A. Global Reference handling: If initial sequence exists, inject it
        if initial_seq is not None:
            fitness_params['reference_sequence'] = initial_seq

        # B. Model-Specific Pre-processing (Converting lists to NumPy arrays)
        fit_type = e_conf['fitness']['type'].lower()

        # --- Model-Specific Requirements ---
        # Exposure Fitness needs a mutator to drift the peak over time
        if fit_type == "exposure":
            fitness_params['mutator'] = epoch_mutator
        # Epistatic Fitness needs a 2D NumPy array for interactions
        elif fit_type == "epistatic" and "interaction_matrix" in fitness_params:
            fitness_params['interaction_matrix'] = np.array(fitness_params['interaction_matrix'])
        # Categorical Fitness needs a 1D NumPy array for site weights
        elif fit_type == "categorical" and "site_weights" in fitness_params:
            fitness_params['site_weights'] = np.array(fitness_params['site_weights'])

        # C. Instantiate via Registry
        fitness = FitnessRegistry.get(
            fit_type,
            **fitness_params
        )

        epochs.append(Epoch(
            name=e_conf['name'],
            generations=e_conf['generations'],
            mutator=epoch_mutator,
            fitness_model=fitness
        ))

    # 4. Setup Samplers
    samplers = []
    for s_conf in conf.get('sampling', []):
        sampler_class = SAMPLER_MAP.get(s_conf['type'])
        if sampler_class:
            samplers.append(sampler_class(s_conf['interval'], s_conf['file']))
        else:
            print(f"Warning: Unknown sampler type '{s_conf['type']}'")

    # 5. Run
    sim = Simulator(population=pop, epochs=epochs, samplers=samplers)

    print(f"--- Starting Simulation: {args.config} ---")
    sim.run()
    print("--- Done ---")


if __name__ == "__main__":
    main()