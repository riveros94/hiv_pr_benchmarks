import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def create_combined_correlation_plots(blue_dfs, orange_dfs, column_name='MI_Scores', axis_linewidth=2.0):
    """
    Create a 2x3 combined figure with correlation plots and additional labels.
    """
    # Set up the figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 16))
    plt.rcParams['font.family'] = 'sans-serif'
    
    def plot_correlation(ax, df1, df2, label1, label2, color, subplot_label, comparison_title):
        corr, p_value = pearsonr(df1[column_name], df2[column_name])
        
        # Create scatter plot
        ax.scatter(df1[column_name], df2[column_name], 
                  color=color, 
                  s=120,
                  edgecolor='black',
                  alpha=0.7)
        
        # Configure labels
        ax.set_xlabel(label1, fontsize=22, labelpad=15)
        ax.set_ylabel(label2, fontsize=22, labelpad=15)
        
        # Add title (comparison) above correlation value
        ax.text(0.5, 1.06, comparison_title, 
                transform=ax.transAxes,
                fontsize=20,
                fontweight='bold',
                ha='center')
        
        # Add correlation value below title
        ax.text(0.5, 0.99, f'Pearson Correlation: {corr:.2f}', 
                transform=ax.transAxes,
                fontsize=20,
                ha='center')
        
        # Add subplot label outside the plot
        ax.text(-0.15, -0.15, subplot_label, transform=ax.transAxes, 
                fontsize=20,
                fontweight='bold',
                ha='center',
                va='center')
        
        # Configure ticks
        ax.tick_params(axis='both', which='major', labelsize=20, width=axis_linewidth, length=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set remaining spines properties
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(axis_linewidth)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(axis_linewidth)
        
        # Remove grid
        ax.grid(False)
        
        return corr, p_value

    # Comparison titles
    titles = [
        'In-house dataset vs Steiner dataset',
        'In-house dataset vs Shen dataset',
        'Steiner dataset vs Shen dataset'
    ]
    
    # Add descriptive labels for each method
    fig.text(0.055, 0.66, 'zScales descriptors', fontsize=20, fontweight='bold', rotation=90)
    fig.text(0.055, 0.165, 'Rosetta energy terms', fontsize=20, fontweight='bold', rotation=90)
    
    # Blue plots (top row)
    in_house_blue, steiner_blue, triang_blue = blue_dfs
    
    # Plot a: In-house vs Steiner (Blue)
    corr1, p1 = plot_correlation(ax1, in_house_blue, steiner_blue, 
                               'In-house dataset', 'Steiner dataset',
                               '#1f77b4', 'a', titles[0])
    
    # Plot b: In-house vs Shen (Blue)
    corr2, p2 = plot_correlation(ax2, in_house_blue, triang_blue,
                               'In-house dataset', 'Shen dataset',
                               '#1f77b4', 'b', titles[1])
    
    # Plot c: Steiner vs Shen (Blue)
    corr3, p3 = plot_correlation(ax3, steiner_blue, triang_blue,
                               'Steiner dataset', 'Shen dataset',
                               '#1f77b4', 'c', titles[2])
    
    # Orange plots (bottom row)
    in_house_orange, steiner_orange, triang_orange = orange_dfs
    
    # Plot d: In-house vs Steiner (Orange)
    corr4, p4 = plot_correlation(ax4, in_house_orange, steiner_orange,
                               'In-house dataset', 'Steiner dataset',
                               'orange', 'd', titles[0])
    
    # Plot e: In-house vs Shen (Orange)
    corr5, p5 = plot_correlation(ax5, in_house_orange, triang_orange,
                               'In-house dataset', 'Shen dataset',
                               'orange', 'e', titles[1])
    
    # Plot f: Steiner vs Shen (Orange)
    corr6, p6 = plot_correlation(ax6, steiner_orange, triang_orange,
                               'Steiner dataset', 'Shen dataset',
                               'orange', 'f', titles[2])
    
    # Adjust layout with more padding
    plt.tight_layout(rect=[0.05, 0, 1, 1], pad=4.0)
    
    # Print correlation values
    print("Blue correlations:")
    print(f"a) In-house vs Steiner - Correlation: {corr1:.2f}, p-value: {p1:.4f}")
    print(f"b) In-house vs Shen - Correlation: {corr2:.2f}, p-value: {p2:.4f}")
    print(f"c) Steiner vs Shen - Correlation: {corr3:.2f}, p-value: {p3:.4f}")
    
    print("\nOrange correlations:")
    print(f"d) In-house vs Steiner - Correlation: {corr4:.2f}, p-value: {p4:.4f}")
    print(f"e) In-house vs Shen - Correlation: {corr5:.2f}, p-value: {p5:.4f}")
    print(f"f) Steiner vs Shen - Correlation: {corr6:.2f}, p-value: {p6:.4f}")
    
    return fig

# z-scales
in_house_blue = pd.read_csv("nfv/mi_norm_values_nfv.csv")
steiner_blue = pd.read_csv("nfv/mi_norm_values_nfv.csv")
triang_blue = pd.read_csv("nfv/mi_norm_values_nfv.csv")

# rosetta
in_house_orange = pd.read_csv("nfv/mi_norm_values_nfv.csv")
steiner_orange = pd.read_csv("nfv/mi_norm_values_nfv.csv")
triang_orange = pd.read_csv("nfv/mi_norm_values_nfv.csv")

# Create the combined plot
fig = create_combined_correlation_plots(
    (in_house_blue, steiner_blue, triang_blue),
    (in_house_orange, steiner_orange, triang_orange),
    axis_linewidth=2.0
)

# Save the figure
plt.savefig('combined_correlations_2x3.png', dpi=300, bbox_inches='tight')
plt.show()
