import argparse
import numpy as np
import yaml
import sys

# Internal imports
from src.servine.simulator import Simulator, Epoch
from src.servine.genome.sequence import Genome
from src.servine.population.container import PopulationRegistry
from src.servine.evolution.mutator import NucleotideMutator
from src.servine.evolution.fitness import FitnessRegistry
from src.servine.io.sampler_registry import SamplerRegistry



def main():
    """
    Run with:
         <.../EvoSim> $env:PYTHONPATH = "src"
         <.../EvoSim> py -m servine.cli src/config.yaml
    in the command line (checked working in Windows PowerShell).
    """
    parser = argparse.ArgumentParser(description="SERVINE: Evolutionary Simulator")

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

    # Create Population
    pop = PopulationRegistry.get(conf, initial_seq, genome)

    # 3. Setup Epochs
    epochs = []
    for e_conf in conf['epochs']:
        # Initialize Mutator
        epoch_mutator = NucleotideMutator(rate=float(e_conf['mutator']['rate']))

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

    # 4. Setup Samplers
    samplers = SamplerRegistry.get_samplers(conf=conf)

    # 5. Run
    sim = Simulator(population=pop, epochs=epochs, samplers=samplers)

    print(f"--- Starting Simulation: {args.config} ---")
    sim.run()
    print("--- Done ---")


if __name__ == "__main__":
    main()