import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output
from ipywidgets import Output
from base import Pruner


class SimulatedAnnealingPruner(Pruner):
    def __init__(
        self, baseline, initial_temperature=1.0, iterations=200,
        mutation_rate=0.05, max_loss=1.0, max_loss_penalty=1e8
    ):
        super().__init__(baseline, max_loss, max_loss_penalty)
        self.initial_temperature = initial_temperature
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.log = Output()

    def _generate_neighbor(self):
        layer_index = np.random.choice(
            self.baseline.prunable_layers,
            p=[self.baseline.layer_importances[i] for i in self.baseline.prunable_layers]
        )
        layer_mask = self.masks[layer_index] * ~(np.random.rand(*self.masks[layer_index].shape) < self.mutation_rate)
        return layer_index, layer_mask

    def _acceptance_probability(self, deltaE, temperature):
        prob = np.exp(-deltaE / temperature)
        if prob > 1: return False
        return np.random.uniform(0, 1) < prob

    def prune(self):
        current_obj_dict = self.calculate_objective()
        best_obj_dict = current_obj_dict
        pbar = tqdm(range(self.iterations))
        display(self.log)

        for step in pbar:
            temperature = self.initial_temperature / (1 + np.log(1 + step))
            if temperature <= 0: break

            layer_index, layer_mask = self._generate_neighbor()
            old_layer_mask = self.get_layer_mask(layer_index)
            self.apply_layer_mask(layer_index, layer_mask)

            new_obj_dict = self.calculate_objective()
            deltaE = new_obj_dict['cost'] - current_obj_dict['cost']

            if deltaE < 0 or self._acceptance_probability(deltaE, temperature):
                current_obj_dict = new_obj_dict
                self.history.append({'layer': layer_index, 'temperature': temperature, **current_obj_dict})

                with self.log:
                    clear_output(wait=True)
                    display(pd.DataFrame(self.history))

                if current_obj_dict['cost'] < best_obj_dict['cost']:
                    best_masks = [mask.copy() if mask is not None else None for mask in self.masks]
                    best_obj_dict = current_obj_dict
            else:
                self.reset_layer_weights(layer_index)
                self.apply_layer_mask(layer_index, old_layer_mask)

        self.masks = best_masks
        self.apply_all_masks()
        return best_masks, best_obj_dict