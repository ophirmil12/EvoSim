import argparse
import numpy as np
import yaml
import sys
import os

from src.servine.simulator import Simulator, Epoch
from src.servine.genome.sequence import Genome
from src.servine.population.population_registry import PopulationRegistry
from src.servine.evolution.mutator_registry import MutatorRegistry
from src.servine.evolution.fitness_registry import FitnessRegistry
from src.servine.io.sampler_registry import SamplerRegistry
from src.servine.color import fg, bg


def run_simulation_from_config(conf):
    """
    Runs the simulation based on a configuration dictionary.
    This function is reusable for both CLI and Streamlit UI.
    """
    print(bg.MAGENTA, "Simulation has initiated form CLI.", bg.RESET)

    # 1. Initialize Core Components
    print("Initializing Core Components...")
    initial_seq = None
    if 'initial_sequence' in conf['population']:
        seq_string = conf['population']['initial_sequence'].upper()
        if 'genome' in conf and 'length' in conf['genome']:
            genome_length = conf['genome']['length']
        else:
            genome_length = len(seq_string)

        if len(seq_string) != genome_length:
            print(fg.RED, "Error: Initial sequence length does not match genome length.", fg.RESET)
            # In UI context, raising an error is better than sys.exit
            raise ValueError("Initial sequence length does not match genome length.")

        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        try:
            initial_seq = np.array([mapping[base] for base in seq_string], dtype=np.uint8)
        except KeyError as e:
            print(fg.RED, f"Error: Invalid nucleotide in initial sequence: {e}", fg.RESET)
            raise ValueError(f"Invalid nucleotide in initial sequence: {e}")

        genome = Genome(length=genome_length)
    else:
        if 'genome' in conf and 'length' in conf['genome']:
            genome = Genome(length=conf['genome']['length'])
        else:
            print(fg.RED, "Error: Genome length must be specified if no initial sequence is provided.", fg.RESET)
            raise ValueError("Genome length must be specified if no initial sequence is provided.")

    # Create Population
    print("Initiating population...")
    pop = PopulationRegistry.get(conf, initial_seq, genome)

    # 2. Setup Epochs
    print("Setting up the epoches...")
    epochs = []
    for e_conf in conf['epochs']:
        # Initialize Mutator
        mutator_params = MutatorRegistry.mutator_params(e_conf)
        epoch_mutator = MutatorRegistry.get(
            e_conf['mutator']['type'],
            **mutator_params
        )

        # Setup Fitness Model
        fitness_params, fit_type = FitnessRegistry.fitness_type_and_params(e_conf, initial_seq, epoch_mutator)
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

    print("Initiating samplers...")
    samplers = SamplerRegistry.get_samplers(conf=conf)

    # 4. Run
    print("Building simulation (population + epochs + samplers = simulation!)...")
    sim = Simulator(population=pop, epochs=epochs, samplers=samplers)

    print(fg.GREEN, "--- Starting Simulation ---", fg.RESET)
    sim.run()
    print(fg.GREEN, "--- Done ---", fg.RESET)
    return True


def main():
    """
    CLI Entry point
    """
    parser = argparse.ArgumentParser(description="SERVINE: Evolutionary Simulator")
    parser.add_argument("config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load Configuration from file
    try:
        with open(args.config, 'r') as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        print(fg.RED, f"Error: Could not read config file. {e}", fg.RESET)
        sys.exit(1)

    # Run the common logic
    try:
        if run_simulation_from_config(conf):
            print(bg.GREEN, "Success! Simulation ran successfully.", bg.RESET)
        else:
            print(fg.RED, "Error: Simulation ran failed.", fg.RESET)
    except ValueError as e:
        print(fg.RED, f"Error: fatal error crash! {e}", fg.RESET)
        sys.exit(1)


if __name__ == "__main__":
    main()