import numpy as np
from tqdm.notebook import tqdm
from base import Pruner

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class SimulatedAnnealingPruner(Pruner):
    def __init__(
        self, baseline, initial_temperature=1.0, iterations=200,
        mutation_rate=0.05, loss_to_warmup=1.0, max_loss_penalty=1e8
    ):
        super().__init__(baseline, loss_to_warmup, max_loss_penalty)
        self.initial_temperature = initial_temperature
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.loss_to_warmup = loss_to_warmup
        self.display_handle = display('', display_id=True)

    def _acceptance_probability(self, deltaE, temperature):
        prob = np.exp(-deltaE / temperature)
        if prob > 1: return False
        return np.random.uniform(0, 1) < prob

    def prune(self):
        for layer_index, layer in enumerate(self.baseline.model.layers):
            if len(layer.get_weights()) <= 0 : continue
            self.max_loss = self.loss_to_warmup * (layer_index + 1) / len(self.baseline.model.layers) # Adaptive max_loss
            current_obj_dict = self.calculate_objective()
            best_obj_dict = current_obj_dict

            for step in tqdm(range(self.iterations), desc=f'[LAYER {layer_index}]'):
                adaptive_mutation_rate = self.mutation_rate * (1 - step / self.iterations)
                temperature = self.initial_temperature / (1 + np.log(1 + step))
                if temperature <= 0: break

                layer_index, layer_mask = self.masks[layer_index] * ~(np.random.rand(*self.masks[layer_index].shape) < adaptive_mutation_rate)
                old_layer_mask = self.get_layer_mask(layer_index)
                self.apply_layer_mask(layer_index, layer_mask)

                new_obj_dict = self.calculate_objective()
                deltaE = new_obj_dict['cost'] - current_obj_dict['cost']

                if deltaE < 0 or self._acceptance_probability(deltaE, temperature):
                    current_obj_dict = new_obj_dict
                    self.history.append({
                        'layer': layer_index, 'temperature': temperature, 
                        'mutation_rate': adaptive_mutation_rate, **current_obj_dict
                    })
                    self.display_handle.update(pd.DataFrame(self.history))

                    if current_obj_dict['cost'] < best_obj_dict['cost']:
                        best_masks = [mask.copy() if mask is not None else None for mask in self.masks]
                        best_obj_dict = current_obj_dict
                else:
                    self.reset_layer_weights(layer_index)
                    self.apply_layer_mask(layer_index, old_layer_mask)

        self.masks = best_masks
        self.apply_all_masks()
        return best_masks, best_obj_dict