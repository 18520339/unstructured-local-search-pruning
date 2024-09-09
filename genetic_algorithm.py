import numpy as np
import pandas as pd
from typing import List, Tuple
from copy import deepcopy
from tqdm.notebook import tqdm
from IPython.display import clear_output
from base import Pruner


class Chromosome:
    def __init__(self, pruner, layer_index, layer_mask=None, init_rate=0.1, disable_prune=False):
        self.pruner = deepcopy(pruner) # Create a copy of the pruner
        self.init_rate = init_rate
        self.layer_index = layer_index
        self.layer_mask = self._initialize_layer_mask() if layer_mask is None else layer_mask
        self.obj_dict = {}

        if not disable_prune: # Avoid applying mask in Crossover to make it faster
            self.pruner.apply_layer_mask(self.layer_index, self.layer_mask)
            self.obj_dict = self.pruner.calculate_objective()

    def _initialize_layer_mask(self):
        layer = self.pruner.pruned_model.layers[self.layer_index]
        return np.random.choice([0, 1], size=layer.get_weights()[0].shape, p=[self.init_rate, 1-self.init_rate])

    def crossover(self, other: 'Chromosome', crossover_rate=0.9) -> Tuple['Chromosome', 'Chromosome']:
        if np.random.random() < crossover_rate:
            crossover_point = np.random.randint(0, self.layer_mask.size - 1)
            mask1_flat, mask2_flat = self.layer_mask.flatten(), other.layer_mask.flatten()
            child1_mask = np.concatenate([mask1_flat[:crossover_point], mask2_flat[crossover_point:]])
            child2_mask = np.concatenate([mask2_flat[:crossover_point], mask1_flat[crossover_point:]])
            child1_mask = child1_mask.reshape(self.layer_mask.shape)
            child2_mask = child2_mask.reshape(other.layer_mask.shape)
            return (
                Chromosome(self.pruner, self.layer_index, child1_mask, disable_prune=True),
                Chromosome(self.pruner, self.layer_index, child2_mask, disable_prune=True)
            )
        return deepcopy(self), deepcopy(other)

    def mutate(self, mutation_rate=0.05):
        self.layer_mask = np.array([
            np.random.choice([0, 1], p=[1-mutation_rate, mutation_rate])
            if np.random.random() < mutation_rate else gene for gene in self.layer_mask.flatten()
        ]).reshape(self.layer_mask.shape)
        self.pruner.apply_layer_mask(self.layer_index, self.layer_mask)
        self.obj_dict = self.pruner.calculate_objective()
        
        
class GeneticAlgorithmPruner(Pruner):
    def __init__(
            self, baseline, num_generations=15, population_size=10,
            tournament_size=5, crossover_rate=0.9, mutation_rate=0.1,
            elite_size=2, target_loss=1.0, max_loss_penalty=1e8
    ):
        super().__init__(baseline, target_loss, max_loss_penalty)
        self.num_generations = num_generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.target_loss = target_loss


    def initialize_layer_population(self, layer_index) -> List[Chromosome]:
        init_rate = 1.0
        while True:
            init_rate *= self.baseline.layer_importances[layer_index] * self.mutation_rate
            population = [
                Chromosome(self, layer_index, init_rate=init_rate)
                for _ in tqdm(range(self.population_size), desc=f'[LAYER {layer_index}] Initialize Population')
            ]
            if all(chromosome.obj_dict['cost'] > self.max_loss_penalty for chromosome in population): continue
            break
        return population


    def tournament_select(self, population: List[Chromosome]) -> List[Chromosome]:
        selected_chromosomes = []
        for _ in range(2):
            tournament = np.random.choice(population, size=self.tournament_size, replace=False)
            winner = min(tournament, key=lambda chromosome: chromosome.obj_dict['cost'])
            selected_chromosomes.append(winner)
        return selected_chromosomes


    def evolve_layer_population(self, layer_index, population: List[Chromosome]) -> Chromosome:
        best_chromosome = None
        for generation in range(self.num_generations):
            # Apply mutation by decreasing mutation rate over time to fine-tune solutions
            adaptive_mutation_rate = self.mutation_rate * (1 -generation / self.num_generations)
            population.sort(key=lambda chromosome: chromosome.obj_dict['cost'])
            new_population = population[:self.elite_size] # Apply elitism

            while True:
                pbar = tqdm(
                    total=self.population_size - self.elite_size, initial=self.elite_size,
                    desc=f'[LAYER {layer_index}] Evolving Generation {generation}'
                )
                while len(new_population) < self.population_size: # Generate new population
                    parent1, parent2 = self.tournament_select(population)
                    child1, child2 = parent1.crossover(parent2, self.crossover_rate)
                    child1.mutate(adaptive_mutation_rate)
                    child2.mutate(adaptive_mutation_rate)
                    new_population.extend([child1, child2])
                    pbar.update(2)
                    
                pbar.close()
                population = new_population
                new_population = []
                if all(chromosome.obj_dict['cost'] > self.max_loss_penalty for chromosome in population): continue
                break

            current_best_chromosome = min(population, key=lambda chromosome: chromosome.obj_dict['cost'])
            if best_chromosome is None or current_best_chromosome.obj_dict['cost'] < best_chromosome.obj_dict['cost']:
                best_chromosome = current_best_chromosome # Update best solution

            self.history.append({
                'layer': layer_index, 'generation': generation,
                'mutation_rate': adaptive_mutation_rate, **current_best_chromosome.obj_dict,
            })
            clear_output(wait=True)
            display(pd.DataFrame(self.history))
        return best_chromosome


    def prune(self):
        for layer_index, layer in enumerate(self.baseline.model.layers):
            if len(layer.get_weights()) <= 0 : continue
            self.max_loss = self.target_loss * (layer_index + 1) / len(self.baseline.model.layers) # Adaptive max_loss

            old_layer_mask = self.get_layer_mask(layer_index)
            population = self.initialize_layer_population(layer_index)
            best_chromosome = self.evolve_layer_population(layer_index, population)

            self.reset_layer_weights(layer_index)
            if best_chromosome.obj_dict['cost'] < self.max_loss_penalty:
                self.apply_layer_mask(layer_index, best_chromosome.layer_mask)
            else:
                self.apply_layer_mask(layer_index, old_layer_mask)
        self.apply_all_masks()
        return self.masks, best_chromosome.obj_dict