# The Evolution Simulator - "Cool Name"

## Overview
The Evolution Simulator, codenamed **"Cool Name"**, is a sophisticated simulation program designed to model
the processes of evolution in a virtual environment.
It allows users to create and observe digital genomes, track evolutionary changes over time, and analyze the effects of various environmental factors on species development.


## How to run the simulator?
To run the Evolution Simulator, follow these steps:
1. Clone the Repository
2. In the terminal, navigate to the project directory
3. $env:PYTHONPATH = "src"
4. py -m santa.cli config.yaml

To customize the simulation parameters, edit the `config.yaml` file before running the simulator.
All outputs are saved in the directory specified in the configuration file.


## Future Plans
- More **mutations models** - JC69 for example
- Option for simulation to **start from a single organism**
- Options for different genomes at the start of the simulation
- **Recombination** models, Insertion and Deletion mutations (need to change genome lengths first)
- More detailed environmental factors affecting evolution
- Visualization tools for better analysis of evolutionary trends (+ explanatory **trees**)
- Changing the codebase to match **different lengths of genomes**