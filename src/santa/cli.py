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
from src.santa.io.sampler import StatisticsSampler, FastaSampler, IdentitySampler

"""
How the components interact:
1. cli.py: Takes your config.yaml and assembles the machine.
2. simulator.py: Runs the clock.
3. population.py: Holds the "DNA Matrix."
4. mutator.py: Uses a random mask to flip bits in that matrix.
5. fitness.py: Scores the rows of the matrix.
6. sampler.py: Records the results to results.csv.
"""

def main():
    """
    Run with:
         <...\EvoSim> $env:PYTHONPATH = "src"
         <...\EvoSim> py -m santa.cli src/config.yaml
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
        mutator = NucleotideMutator(rate=float(e_conf['mutator']['rate']))

        # Setup Fitness Model - if initial sequence provided, use it as reference
        #  (used in purifying selection)
        fitness_params = e_conf['fitness'].get('params', {})
        if initial_seq is not None:
            fitness_params['reference_sequence'] = initial_seq

        fitness = FitnessRegistry.get(
            e_conf['fitness']['type'],
            **fitness_params
        )

        epochs.append(Epoch(
            name=e_conf['name'],
            generations=e_conf['generations'],
            mutator=mutator,
            fitness_model=fitness
        ))

    # 4. Setup Samplers
    samplers = []
    for s_conf in conf.get('sampling', []):
        if s_conf['type'] == 'stats':
            samplers.append(StatisticsSampler(s_conf['interval'], s_conf['file']))
        elif s_conf['type'] == 'fasta':
            samplers.append(FastaSampler(s_conf['interval'], s_conf['file']))
        elif s_conf['type'] == 'identity':
            samplers.append(IdentitySampler(s_conf['interval'], s_conf['file']))

    # 5. Run
    sim = Simulator(population=pop, epochs=epochs, samplers=samplers)

    print(f"--- Starting Simulation: {args.config} ---")
    sim.run()
    print("--- Done ---")


if __name__ == "__main__":
    main()