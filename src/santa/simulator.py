# Main simulation loop and Epoch management


import logging

logger = logging.getLogger(__name__)


class Epoch:
    """A time period with specific evolutionary rules."""

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
        self.current_generation = 0

    def run(self):
        for epoch in self.epochs:
            print(f"Running Epoch: {epoch.name}...")
            for g in range(epoch.generations):
                self.current_generation += 1

                # Call the update hook (e.g., for ExposureFitness to move the peak)
                epoch.fitness_model.update(self.current_generation)

                # 1. Calculate Fitness
                fitness_values = epoch.fitness_model.evaluate_population(self.population)

                # 2. Select survivors (Wright-Fisher)
                self.population.select(fitness_values)

                # 3. Mutate (Variation)
                epoch.mutator.apply(self.population)

                # 4. Data Collection
                for sampler in self.samplers:
                    if sampler.is_sampling_time(self.current_generation):
                        sampler.sample(self.population, self.current_generation)

            print(f"Finished Epoch: {epoch.name} (index: {self.epochs.index(epoch)})")

        print("Finalizing samplers...")
        for sampler in self.samplers:
            if hasattr(sampler, 'finalize'):
                sampler.finalize()