import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from base import Pruner


class SimulatedAnnealingPruner(Pruner):
    def __init__(
        self, baseline, initial_temperature=1.0, iterations=200,
        mutation_rate=0.05, loss_to_warmup=1.0, max_loss_penalty=1e8
    ):
        super().__init__(baseline, loss_to_warmup, max_loss_penalty)
        self.initial_temperature = initial_temperature
        self.iterations = iterations
        self.mutation_rate = mutation_rate # The algorithm will calculate its adaptive version to fine-tune solutions over time
        self.loss_to_warmup = loss_to_warmup # Warmup the max_loss to the final value for more aggressive pruning at the end
        self.display_handle = display('', display_id=True)

    def _acceptance_probability(self, deltaE, temperature): # Decide whether to accept or reject a new solution
        prob = np.exp(-deltaE / temperature)
        ''' Since the probability is a function of e^-x:
        - If the change in the objective is negative, the function will keep increasing to infinity
        - If the change in the objective >= 0, the function will have a range of (0, 1], making it suitable for probability calculation
        As I researched this acceptance probability in SA is governed by an optimization rule called Metropolis criterion
        '''
        if prob > 1: return False
        return np.random.uniform(0, 1) < prob # If the probability <= 1, accept the new mask with a certain probability

    def prune(self):
        for layer_index, layer in enumerate(self.baseline.model.layers):
            if len(layer.get_weights()) <= 0 : continue # Skip layers with no weights (e.g., activation layers)
            self.max_loss = self.loss_to_warmup * (layer_index + 1) / len(self.baseline.model.layers) # Adaptive max_loss
            current_obj_dict = self.calculate_objective() # Calculate initial objective for the current layer
            best_obj_dict = current_obj_dict # Initialize best objective for the current layer

            for step in tqdm(range(self.iterations), desc=f'[LAYER {layer_index}]'):
                # Decrease mutation rate over time to fine-tune solutions
                adaptive_mutation_rate = self.mutation_rate * (1 - step / self.iterations)

                # Logarithmic annealing schedule to decrease the temperature as the step increases
                temperature = self.initial_temperature / (1 + np.log(1 + step))
                if temperature <= 0: break

                # Randomly flip "adaptive_mutation_rate"% of the mask values
                layer_mask = self.masks[layer_index] * ~(np.random.rand(*self.masks[layer_index].shape) < adaptive_mutation_rate)
                old_layer_mask = self.get_layer_mask(layer_index) # Save the old mask to revert the changes if the new mask is not accepted
                self.apply_layer_mask(layer_index, layer_mask)

                new_obj_dict = self.calculate_objective()
                deltaE = new_obj_dict['cost'] - current_obj_dict['cost'] # Calculate the change in the objective function, lower is better

                # If the new objective is better or the acceptance probability is met, update the current objective and save the history
                if deltaE < 0 or self._acceptance_probability(deltaE, temperature):
                    current_obj_dict = new_obj_dict
                    self.history.append({
                        'layer': layer_index, 'temperature': temperature,
                        'mutation_rate': adaptive_mutation_rate, **current_obj_dict
                    })
                    self.display_handle.update(pd.DataFrame(self.history)) # Display the history in a table

                    if current_obj_dict['cost'] < best_obj_dict['cost']: # Update best solution found so far for the layer
                        best_masks = [mask.copy() if mask is not None else None for mask in self.masks]
                        best_obj_dict = current_obj_dict
                else:
                    self.reset_layer_weights(layer_index) # Revert the changes if the new mask is not accepted
                    self.apply_layer_mask(layer_index, old_layer_mask)

        self.masks = best_masks
        self.apply_all_masks() # Update best pruning masks for all layers
        return best_masks, best_obj_dict