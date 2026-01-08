# The main file - run the simulation from CLI

import argparse
import numpy as np
import yaml
import sys


from src.servine.simulator import Simulator, Epoch
from src.servine.genome.sequence import Genome
from src.servine.population.population_registry import PopulationRegistry
from src.servine.evolution.mutator import NucleotideMutator
from src.servine.evolution.fitness_registry import FitnessRegistry
from src.servine.io.sampler_registry import SamplerRegistry
from src.servine.color import fg


# TODO-s:
#  0. Understand the code
#  1. Complete all TODOs (recombination?)
#  2. Make some cool examples
#  3. Finalise README


def main():
    """
    Run with:
         <.../EvoSim>    $env:PYTHONPATH = "src"
         <.../EvoSim>    py -m servine.cli src/config.yaml
    in the command line (checked working in Windows PowerShell).

    For cluster (Linux SLURM):
        export PYTHONPATH="src"
        python -m servine.cli src/hiv_simulation.yaml
    See full script example in 'slurm.sh'.
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
        print(fg.RED, f"Error: Could not read config file. {e}", fg.RESET)
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
            print(fg.RED, "Error: Initial sequence length does not match genome length.", fg.RESET)
            sys.exit(1)
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        try:
            initial_seq = np.array([mapping[base] for base in seq_string], dtype=np.uint8)
        except KeyError as e:
            print(fg.RED, f"Error: Invalid nucleotide in initial sequence: {e}", fg.RESET)
            sys.exit(1)
        genome = Genome(length=genome_length)
    else:
        if 'genome' in conf and 'length' in conf['genome']:
            genome = Genome(length=conf['genome']['length'])
        else:
            print(fg.RED, "Error: Genome length must be specified if no initial sequence is provided.", fg.RESET)
            sys.exit(1)

    # Create Population
    pop = PopulationRegistry.get(conf, initial_seq, genome)

    # 3. Setup Epochs
    epochs = []
    for e_conf in conf['epochs']:
        # Initialize Mutator
        epoch_mutator = NucleotideMutator(rate=float(e_conf['mutator']['rate']))

        # Setup Fitness Model
        # TODO: we need to add functionality where a user can select multiple fitness models per
        #  epoch, and the final fitness of an organism is the product (*) of all fitness's models
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

    print(fg.GREEN, f"--- Starting Simulation: {args.config} ---", fg.RESET)
    sim.run()
    print(fg.GREEN, "--- Done ---", fg.RESET)


if __name__ == "__main__":
    main()