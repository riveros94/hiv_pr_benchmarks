import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_comparison_plot(data, output_filename='comparison_plot.png'):
    """
    Create a publication-quality scatter plot comparing AUC values across datasets.
    
    Parameters:
    -----------
    data : list of dictionaries
        Each dictionary contains model performance data for a dataset
    output_filename : str
        Name of the output file
    """
    # Setup
    DPI = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    drugs = ['ATV', 'DRV', 'LPV', 'TPV', 'NFV', 'IDV', 'FPV', 'SQV']
    models = ['MLP - Int encod', 'CNN - Int encod', 'BRNN - Int encod', 
              'RF - Triang', 'KNN - Triang', 'zScales LR', 'Rosetta LR']
    colors = sns.color_palette('hsv', len(models))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16,9))  
    
    # Plot for each dataset
    for d, dataset in enumerate(data):
        for i, model in enumerate(models):
            x_positions = [j + d * len(drugs) for j in range(len(drugs))]
            ax.scatter(x_positions, dataset[model], 
                      color=colors[i], 
                      label=f"{model}" if d == 0 else "", 
                      s=110,  
                      alpha=0.8)
    
    # Add vertical lines between datasets
    for i in range(1, len(data)):
        ax.axvline(x=len(drugs) * i - 0.5, 
                   color='black', 
                   linestyle='--', 
                   linewidth=0.5)  # Thinner line for publication
    
    # Configure x-axis
    all_x_positions = [i + j * len(drugs) for j in range(len(data)) 
                      for i in range(len(drugs))]
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(drugs * len(data), rotation=90, fontsize=22)
    
    # Add dataset labels
    dataset_labels = ["Steiner's dataset", "Shen's dataset", "In-house"]
    for i, label in enumerate(dataset_labels):
        ax.text(len(drugs) * i + len(drugs) / 2 - 0.5, 1.04,
                label, ha='center', va='bottom', 
                fontsize=26)
    
    # Customize axes
    ax.set_ylim(0.65, 1.03)
    plt.yticks(fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    plt.xlabel('Drugs', fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    
    # Configure spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set visible spines to black with width 1
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.0)
    
    # Configure legend
    ax.legend(title='Models', 
             loc='lower left', 
             fontsize=20, 
             title_fontsize=22,
             frameon=True,
             edgecolor='black',
             fancybox=False)
    

    # Remove grid
    ax.grid(False)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
    plt.close()

data = [
    {   # Steiner
        'MLP - Int encod':               [0.892, 0.895, 0.946, 0.863, 0.904, 0.943, 0.883, 0.911],
        'CNN - Int encod':               [0.972, 0.988, 0.959, 0.969, 0.981, 0.984, 0.961, 0.987],
        'BRNN - Int encod':              [0.972, 0.983, 0.984, 0.934, 0.981, 0.981, 0.945, 0.979],
        'RF - Triang':                   [0.967, 0.946, 0.963, 0.869, 0.977, 0.973, 0.930, 0.974],
        'KNN - Triang':                  [0.933, 0.926, 0.973, 0.879, 0.945, 0.979, 0.922, 0.956],
        'zScales LR':                    [0.985, 0.989, 0.980, 0.969, 0.987, 0.994, 0.970, 0.994],
        'Rosetta LR':                    [0.985, 0.991, 0.980, 0.970, 0.985, 0.997, 0.964, 0.988],

    },
    {   # TRIANG
        'MLP - Int encod':               [0.9987, 0.976, 0.9892, 0.9968, 0.9705084, 0.9953, 0.9975, 0.9919],
        'CNN - Int encod':               [0.9984, 0.995, 0.998, 0.9979, 0.9992, 0.9919, 0.997, 0.9961],
        'BRNN - Int encod':              [1.0000, 0.9936834, 0.9998, 1.000, 1.000, 0.999, 0.999, 0.999],
        'RF - Triang':                   [0.999918, 0.999404, 0.99986, 0.999843, 0.999884, 0.999836, 0.99988, 0.999788],
        'KNN - Triang':                  [0.999447, 0.998413, 0.999846, 0.999508,0.999426, 0.999133, 0.999325, 0.999379],
        'zScales LR':                    [0.999584572030688, 0.999584572030688, 0.999573644659319, 0.999472219173551, 0.999578016361671, 0.999810492875013, 0.999810492875013, 0.99966764112216],
        'Rosetta LR':                    [0.999966622544834, 0.99316735514285, 0.999544337868138, 0.998637076582586, 0.999925410753346, 0.999544337868138, 0.999897353984293,0.999855130272215],
    },
    {   #In house
        'MLP - Int encod':               [0.861, 0.876, 0.913, 0.835, 0.871, 0.877, 0.881, 0.915],
        'CNN - Int encod':               [0.963, 0.949, 1.000, 0.940, 0.974, 0.962, 0.978, 0.980],
        'BRNN - Int encod':              [0.975, 0.848, 0.978, 0.884, 0.989, 0.943, 0.955, 0.967],
        'RF - Triang':                   [0.968, 0.879, 0.960, 0.866, 0.979, 0.945, 0.955, 0.967],
        'KNN - Triang':                  [0.900, 0.884, 0.962, 0.832, 0.939, 0.956, 0.943, 0.955],
        'zScales LR':                    [0.985, 0.986, 0.987, 0.914, 0.991, 0.982, 0.991, 0.983],
        'Rosetta LR':                    [0.984, 0.963, 0.978, 0.890, 0.985, 0.967, 0.978, 0.983],                        
                                #drugs = ['ATV', 'DRV', 'LPV', 'TPV', 'NFV', 'IDV', 'FPV', 'SQV']
    }
]

create_comparison_plot(data, output_filename='auc.png')
