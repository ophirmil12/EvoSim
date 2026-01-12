# The Evolution Simulator - "SERVINE"

## Overview
The Evolution Simulator, **"SERVINE"**, is a sophisticated simulation program designed to model
the processes of evolution in a virtual environment.
It allows users to create and observe digital genomes, track evolutionary changes over time, and analyze the effects of various environmental factors on species development.


## How to run the simulator?
To run the Evolution Simulator, follow these steps:
1. Clone the Repository
2. In the terminal, navigate to the project directory
3. $env:PYTHONPATH = "src"
4. py -m servine.cli config.yaml


To run the graphical interface:
1. py -m pip install streamlit
2. py -m streamlit run UI.py

To customize the simulation parameters, edit the `config.yaml` file before running the simulator.
All outputs are saved in the directory specified in the configuration file.


## File Structure

### ./src/servine/
- cli.py
  - main: read config file, initialize core components, set up epochs, create samplers, run simulation loop
- simulator.py
  - Epoch: A time period with specific evolutionary rules
  - Simulator: Manages the overall simulation process.
        run(): calculate fitness, select survivors, collect data, apply mutations
- \__init\__.py

- evolution/
  - fitness.py
    - FitnessModel: Base class for fitness calculation models
    - FitnessRegistry: A simple factory to fetch fitness models by name
    - NeutralFitness(FitnessModel): Naive fitness model where all individuals have equal fitness
    - PurifyingFitness(FitnessModel): Fitness model where mutations reduce fitness exponentially
    - EpistaticFitness(FitnessModel): itness model with pairwise epistatic interactions
    - FrequencyDependentFitness(FitnessModel): Fitness model where common genotypes are penalized. This simulates the immune system. The more "popular" a specific sequence becomes,
      the lower its fitness. This forces the virus to keep changing to "rare" genotypes to survive.
    - ExposureFitness(PurifyingFitness): models a changing environment. Every X generations, the "optimal" sequence changes, and the population must catch up or die.
    - CategoricalFitness(FitnessModel): Some sites are "locked" (Lethal if mutated), while others are "flexible" (Neutral).

  - mutators.py
    - Mutator: Base class for mutation models
    - NucleotideMutator(Mutator): Implements point mutations for nucleotide sequences. Considers transition bias 

- genome/
  - sequence.py
    - Genome: Defines the structure and alphabet of the genetic material.
    - SequenceHandler: Utility to handle bulk operations on sequences. Using a 2D NumPy array [Population_Size, Genome_Length].

- io/
  - sampler.py
    - Sampler: Base class for data samplers
    - FastaSampler(Sampler): Exports population sequences to a FASTA file. Also records ancestors to reconstruct phylogenetic trees.
    - IdentitySampler(Sampler): Plots/Records the average sequence identity relative to the initial sequence.
    - PairwiseIdentitySampler(Sampler): Calculates Average Pairwise Distance using the Jukes-Cantor (JC69) model.
    - FitnessSampler(Sampler): Tracks Average Fitness (Genetic Health) over time.
    - DiversitySampler(Sampler): Tracks Population Diversity (Unique Genotypes) over time.
    - HaplotypeFrequencySampler(Sampler): Tracks the frequency of the most common genotypes (Haplotypes).
    - InitialAlleleFrequencySampler(Sampler): Tracks the frequency of the initial genotypes.
  - trees.py
    - TreeRecorder(Sampler): A type of Sampler that Tracks the ancestry of every individual to reconstruct evolutionary history.

- population/
  - container.py
    - Population: Manages the pool of sequences using a NumPy matrix. Rows = Individuals, Columns = Nucleotide positions.

### ./output/
- example_out/
- genetic_health_example/
- low_selection_example/

### ./tests/
- test_diversity_plot.py
