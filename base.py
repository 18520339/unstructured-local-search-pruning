import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, AvgPool2D, Flatten, Dense


class MNIST:
    def __init__(self):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.y_train, self.y_test = y_train, y_test

        # Add channel dimmension and normalize pixel values between 0 and 1
        self.x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        self.x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        self.input_shape = self.x_train.shape[1:]
        self.num_classes = len(set(self.y_train))

    def display_sample_images(self, num_images=5):
        fig, axes = plt.subplots(1, num_images)
        for i in range(num_images):
            axes[i].imshow(self.x_train[i], cmap='gray')
            axes[i].set_title(f"Label: {self.y_train[i]}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return (
            f'Training data shape: {self.x_train.shape}\n'
            f'Training labels shape: {self.y_train.shape}\n'
            f'Test data shape: {self.x_test.shape}\n'
            f'Test labels shape: {self.y_test.shape}\n'
            f'Input shape: {self.input_shape}\n'
            f'Number of classes: {self.num_classes}'
        )
        
        
class Baseline:
    def __init__(self, data, model_path='LeNet.keras'):
        self.data = data
        self.model_path = model_path

    def build_and_compile_lenet(self, optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model = Sequential([
            Input(shape=data.input_shape),
            Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', name='Conv1'),
            AvgPool2D(pool_size=2, name='AvgPool1'),
            Conv2D(filters=16, kernel_size=5, activation='sigmoid', name='Conv2'),
            AvgPool2D(pool_size=2, name='AvgPool2'),
            Flatten(name='Flatten'),
            Dense(120, activation='sigmoid', name='FC1'),
            Dense(84, activation='sigmoid', name='FC2'),
            Dense(data.num_classes, activation='softmax', name='Output')
        ], name='LeNet')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model.summary(show_trainable=True)

    def train(self, val_ratio=0.2, epochs=50, batch_size=128):
        # Split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.data.x_train, self.data.y_train, test_size=val_ratio)

        # Train model and save it, then obtain necessary information for pruning process
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs, batch_size=batch_size,
            verbose=1
        ).history

        self.model.save(self.model_path)
        self.calculate_pruning_info()

    def calculate_pruning_info(self):
        # Calculate the loss on test set, metrics like accuracy, prunable layers, and total weights
        self.loss, self.metrics = self.model.evaluate(self.data.x_test, self.data.y_test, verbose=0)
        self.prunable_layers, self.total_weights = [], 0

        for layer_index, layer in enumerate(self.model.layers):
            params = layer.get_weights()
            if len(params) > 0: # Skip layers with no weights (e.g., activation layers)
                self.prunable_layers.append(layer_index)
                self.total_weights += params[0].size # Add number of weights in the layer

    def plot_training_history(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        
class Pruner(ABC):
    def __init__(self, baseline, max_loss=1.0, max_loss_penalty=1e8):
        self.baseline = baseline
        self.pruned_model = load_model(baseline.model_path)
        self.masks = self._initialize_masks() # Initialize masks for all layers
        self.max_loss = max_loss # Maximum allowed loss
        self.max_loss_penalty = max_loss_penalty # Penalty for exceeding that maximum loss
        self.history = []

    @abstractmethod
    def prune(self):
        raise NotImplementedError('Method not implemented')

    def _initialize_masks(self):
        masks = [] # Initialize masks as all ones (no pruning)
        for layer in self.baseline.model.layers:
            if len(layer.get_weights()) > 0: # Skip layers with no weights (e.g., activation layers)
                weight_mask = np.ones_like(layer.get_weights()[0]) # Weight-level pruning mask
                masks.append(weight_mask)
            else: masks.append(None)
        return masks

    def apply_all_masks(self):
        self.pruned_model = load_model(self.baseline.model_path)
        for layer, mask in zip(self.pruned_model.layers, self.masks):
            if mask is not None:  # Skip layers with no weights (e.g., activation layers)
                weights, biases = layer.get_weights()
                weights *= mask # Apply mask to weights
                layer.set_weights([weights, biases])

    def get_layer_mask(self, layer_index):
        if 0 <= layer_index < len(self.masks): return self.masks[layer_index]
        raise ValueError('Invalid layer index')

    def apply_layer_mask(self, layer_index, layer_mask):
        if 0 <= layer_index < len(self.pruned_model.layers) and self.masks[layer_index] is not None:
            self.masks[layer_index] = layer_mask # Update mask for the layer
            layer = self.pruned_model.layers[layer_index]
            weights, biases = layer.get_weights()
            weights *= layer_mask # Apply mask to weights of the layer
            layer.set_weights([weights, biases])
        else: raise ValueError('Invalid layer index')

    def get_layer_weights(self, layer_index):
        if 0 <= layer_index < len(self.pruned_model.layers):
            return self.pruned_model.layers[layer_index].get_weights()[0]
        raise ValueError('Invalid layer index')

    def reset_layer_weights(self, layer_index):
        if 0 <= layer_index < len(self.pruned_model.layers):
            weights, biases = self.baseline.model.layers[layer_index].get_weights() # Get original weights
            self.pruned_model.layers[layer_index].set_weights([weights, biases]) # Reset weights
        else: raise ValueError('Invalid layer index')

    def calculate_objective(self):
        # Calculate cost as a combination of relative loss, sparsity, and a penalty for exceeding the maximum allowed loss
        loss, metrics = self.pruned_model.evaluate(self.baseline.data.x_test, self.baseline.data.y_test, verbose=0)
        sparsity = np.mean([np.mean(mask == 0) for mask in self.masks if mask is not None]) # The ratio of 0 masks in all layers

        # The loss difference is squared to penalize high loss more than low loss
        loss_diff = (self.baseline.loss - loss) ** 2 if loss < self.max_loss else self.max_loss_penalty

        # Balance between maintaining loss and achieving target sparsity
        cost = loss_diff + 1 / (sparsity + 1e-8) # # A small epsilon value is added to avoid division by 0
        return {'cost': cost, 'metrics': metrics, 'loss': loss, 'sparsity': sparsity}