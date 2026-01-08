# Main simulation loop and Epoch management
# for epoch (time period in simulation) in epochs:
#   for generation in len(epoch):
#       Calculate Fitness
#       Select survivors (Wright-Fisher)
#       Data Collection for graphs and analysis
#       Mutate (Variation) the population


import logging

from src.servine.io.trees import TreeRecorder
from src.servine.color import fg

logger = logging.getLogger(__name__)


class Epoch:
    """
    A time period with specific evolutionary rules.
    For example - a time period where the virus spreads, or a vaccine found.
    """

    def __init__(self, name, generations, mutator, fitness_model):
        self.name = name
        self.generations = generations
        self.mutator = mutator
        self.fitness_model = fitness_model


class Simulator:
    """The engine that runs the generations."""

    def __init__(self, population, epochs, samplers):
        self.population = population
        self.epochs = epochs
        self.samplers = samplers
        self.tree_recorder = next((s for s in samplers if isinstance(s, TreeRecorder)), None)
        self.current_generation = 0
        self.current_individual_ids = []

    def run(self):
        """The main simulation loop"""
        # Start with the founding IDs (0, 1, 2... N-1)
        self.current_individual_ids = list(range(len(self.population.get_matrix())))

        for epoch in self.epochs:
            print(fg.BLUE, f"Running Epoch: {epoch.name}...", fg.RESET)
            for g in range(epoch.generations):
                self.current_generation += 1

                # Call the update hook (e.g., for ExposureFitness to move the peak)
                epoch.fitness_model.update(self.current_generation)

                # 1. Calculate Fitness
                fitness_values = epoch.fitness_model.evaluate_population(self.population)

                # 2. Select survivors (Wright-Fisher) + Record Ancestry
                parent_indices = self.population.select(fitness_values)

                # 3. Data Collection for graphs and analysis
                if self.tree_recorder:
                    # We always need to know who the parents were to update the 'jump' map
                    self.tree_recorder.record_intermediate_step(parent_indices)
                self.collect_data()

                # 4. Mutate (Variation)
                epoch.mutator.apply(self.population)

            print(fg.MAGENTA, f"Finished Epoch: {epoch.name} (index: {self.epochs.index(epoch)})", fg.RESET)

        print(fg.GREEN, "Finalizing samplers...", fg.GREEN)
        for sampler in self.samplers:
            sampler.finalize()

    def collect_data(self):
        """Collect data from samplers in the simulation."""
        # Sampling - Now passing 'tree_provider' kwarg
        for sampler in self.samplers:
            if sampler.is_sampling_time(self.current_generation):
                result = sampler.sample(
                    self.population,
                    self.current_generation,
                    ids=self.current_individual_ids,
                    tree_provider=self.tree_recorder  # The "history book"
                )
                # Update our global IDs if the TreeRecorder just minted new ones
                if isinstance(sampler, TreeRecorder) and result:
                    self.current_individual_ids = result