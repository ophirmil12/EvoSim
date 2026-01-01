import argparse
import numpy as np
import yaml
import sys

# Internal imports
from src.santa.simulator import Simulator, Epoch
from src.santa.genome.sequence import Genome
from src.santa.population.container import PopulationRegistry
from src.santa.evolution.mutator import NucleotideMutator
from src.santa.evolution.fitness import FitnessRegistry
from src.santa.io.sampler_registry import SamplerRegistry



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
    initial_seq = None
    if 'initial_sequence' in conf['population']:
        seq_string = conf['population']['initial_sequence'].upper()
        if 'genome' in conf and 'length' in conf['genome']:
            genome_length = conf['genome']['length']
        else:
            genome_length = len(seq_string)
        if len(seq_string) != genome_length:
            print("Error: Initial sequence length does not match genome length.")
            sys.exit(1)
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        try:
            initial_seq = np.array([mapping[base] for base in seq_string], dtype=np.uint8)
        except KeyError as e:
            print(f"Error: Invalid nucleotide in initial sequence: {e}")
            sys.exit(1)
        genome = Genome(length=genome_length)
    else:
        if 'genome' in conf and 'length' in conf['genome']:
            genome = Genome(length=conf['genome']['length'])
        else:
            print("Error: Genome length must be specified if no initial sequence is provided.")
            sys.exit(1)

    # Create Population     TODO: Add support for heterogeneous populations, and so on extensions
    pop = PopulationRegistry.get(conf, initial_seq, genome)

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
        elif fit_type == "site_specific":
            fitness_params['site_intensities'] = np.array(fitness_params['site_intensities'])

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
    samplers = SamplerRegistry.get_samplers(conf=conf)

    # 5. Run
    sim = Simulator(population=pop, epochs=epochs, samplers=samplers)

    print(f"--- Starting Simulation: {args.config} ---")
    sim.run()
    print("--- Done ---")


if __name__ == "__main__":
    main()