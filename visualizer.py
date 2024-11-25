import matplotlib.pyplot as plt


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
        

def visualize_pruning_results(ga_df, sa_df, ga_by_layers, sa_by_layers):
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    markers = {
        'ga_name': 'Genetic Algorithm', 'sa_name': 'Simulated Annealing',
        'ga_cost': '#1f77b4'          , 'sa_cost': '#aec7e8', # Blue and Light blue
        'ga_loss': '#d62728'          , 'sa_loss': '#ff9896', # Red and Light red
        'ga_accuracy': '#2ca02c'      , 'sa_accuracy': '#98df8a', # Green and Light green
    }

    # 1st subplot: Cost vs Sparsity, Accuracy vs Sparsity, Loss vs Sparsity (detailed line plot)
    ax1 = axs[0][0]
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60)) # Offset the third axis

    # Cost vs Sparsity
    ax1.plot(ga_df['sparsity'], ga_df['cost'], label=f"{markers['ga_name']} Cost", color=markers['ga_cost'], marker='o', linestyle='--')
    ax1.plot(sa_df['sparsity'], sa_df['cost'], label=f"{markers['sa_name']} Cost",  color = markers['sa_cost'])
    ax1.set_xlabel('Sparsity', fontsize=14)
    ax1.set_ylabel('Cost', color=markers['ga_cost'], fontsize=14)
    ax1.tick_params(axis='y', labelcolor=markers['ga_cost'], labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_title('Cost, Accuracy, and Loss vs Sparsity', fontsize=18, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper center', fontsize=14)
    ax1.annotate(
        'Initial Point', fontsize=14, # Annotations for key points
        xy = (ga_df['sparsity'].iloc[0], ga_df['cost'].iloc[0]),
        xytext = (ga_df['sparsity'].iloc[0] + 0.02, ga_df['cost'].iloc[0] + 5),
        arrowprops = dict(facecolor='black', arrowstyle='->')
    )

    # Metrics (Accuracy) vs Sparsity
    ax2.plot(ga_df['sparsity'], ga_df['metrics'], label=f"{markers['ga_name']} Accuracy", color=markers['ga_accuracy'], marker='o', linestyle='--')
    ax2.plot(sa_df['sparsity'], sa_df['metrics'], label=f"{markers['sa_name']} Accuracy", color=markers['sa_accuracy'])
    ax2.set_ylabel('Metrics (Accuracy)', color=markers['ga_accuracy'], fontsize=14)
    ax2.tick_params(axis='y', labelcolor=markers['ga_accuracy'], labelsize=14)

    # Loss vs Sparsity
    ax3.plot(ga_df['sparsity'], ga_df['loss'], label=f"{markers['ga_name']} Loss", color=markers['ga_loss'], marker='o', linestyle='--')
    ax3.plot(sa_df['sparsity'], sa_df['loss'], label=f"{markers['sa_name']} Loss", color=markers['sa_loss'])
    ax3.set_ylabel('Loss', color=markers['ga_loss'], fontsize=14)
    ax3.tick_params(axis='y', labelcolor=markers['ga_loss'], labelsize=14)

    # 2nd subplot: Final Cost per layer (grouped bar chart)
    layers = ga_by_layers['layer']
    axs[0][1].bar(layers - 0.15, ga_by_layers['cost'], label=f"{markers['ga_name']} Cost", color=markers['ga_cost'], width=0.3)
    axs[0][1].bar(layers + 0.15, sa_by_layers['cost'], label=f"{markers['sa_name']} Cost", color=markers['sa_cost'], width=0.3)
    axs[0][1].set_title('Final Cost per Layer', fontsize=18, fontweight='bold')
    axs[0][1].set_xlabel('Layer', fontsize=14)
    axs[0][1].set_ylabel('Cost', fontsize=14)
    axs[0][1].tick_params(axis='x', labelsize=14)
    axs[0][1].tick_params(axis='y', labelsize=14)
    axs[0][1].legend(loc='upper center', fontsize=14)
    axs[0][1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3rd subplot: Final Accuracy per layer (grouped bar chart)
    axs[1][0].bar(layers - 0.15, ga_by_layers['metrics'], label=f"{markers['ga_name']} Accuracy", color=markers['ga_accuracy'], width=0.3)
    axs[1][0].bar(layers + 0.15, sa_by_layers['metrics'], label=f"{markers['sa_name']} Accuracy", color=markers['sa_accuracy'], width=0.3)
    axs[1][0].set_title('Final Accuracy per Layer', fontsize=18, fontweight='bold')
    axs[1][0].set_xlabel('Layer', fontsize=14)
    axs[1][0].set_ylabel('Metrics (Accuracy)', fontsize=14)
    axs[1][0].tick_params(axis='x', labelsize=14)
    axs[1][0].tick_params(axis='y', labelsize=14)
    axs[1][0].legend(loc='upper center', fontsize=14)
    axs[1][0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4th subplot: Final Loss per layer (grouped bar chart)
    axs[1][1].bar(layers - 0.15, ga_by_layers['loss'], label=f"{markers['ga_name']} Loss", color=markers['ga_loss'], width=0.3)
    axs[1][1].bar(layers + 0.15, sa_by_layers['loss'], label=f"{markers['sa_name']} Loss", color=markers['sa_loss'], width=0.3)
    axs[1][1].set_title('Final Loss per Layer', fontsize=18, fontweight='bold')
    axs[1][1].set_xlabel('Layer', fontsize=14)
    axs[1][1].set_ylabel('Loss', fontsize=14)
    axs[1][1].tick_params(axis='x', labelsize=14)
    axs[1][1].tick_params(axis='y', labelsize=14)
    axs[1][1].legend(loc='upper center', fontsize=14)
    axs[1][1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()