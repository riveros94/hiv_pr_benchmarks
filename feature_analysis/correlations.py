def create_combined_correlation_plots(blue_dfs, orange_dfs, column_name='MI_Scores', axis_linewidth=2.0):
    """
    Create a 2x3 combined figure with correlation plots and updated layout.
    """
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 14))
    plt.rcParams['font.family'] = 'sans-serif'
    
    def plot_correlation(ax, df1, df2, color, subplot_label, show_title=False, title=""):
        corr, p_value = pearsonr(df1[column_name], df2[column_name])
        
        
        x_min, x_max = df1[column_name].min(), df1[column_name].max()
        y_min, y_max = df2[column_name].min(), df2[column_name].max()
        x_margin = (x_max - x_min) * 0.10  
        y_margin = (y_max - y_min) * 0.10  
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Create scatter plot
        ax.scatter(df1[column_name], df2[column_name], 
                  color=color, 
                  s=120,
                  edgecolor='black',
                  alpha=0.7)
        
        # Configure labels
        ax.set_xlabel('Residue-wise norm mutual inf', fontsize=22, labelpad=15)
        ax.set_ylabel('Residue-wise norm mutual inf', fontsize=22, labelpad=15)
        
        # Add title only for top row if show_title is True
        if show_title:
            ax.text(0.5, 1.15, title, 
                    transform=ax.transAxes,
                    fontsize=24,
                    fontweight='bold',
                    ha='center')
        
        # Add r value in gray box at top center
        bbox_props = dict(boxstyle="round,pad=0.3", fc='lightgray', ec="gray", alpha=0.8)
        ax.text(0.16, 1.05, f'r = {corr:.2f}', 
                transform=ax.transAxes,
                fontsize=20,
                ha='center',
                va='top',
                bbox=bbox_props)
        
        ax.text(-0.14, 1.05, subplot_label.upper(), 
                transform=ax.transAxes, 
                fontsize=34,
                fontweight='bold',
                ha='right',
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

    # Comparison titles - only for top row
    titles = [
        "In-house vs Steiner's datasets",
        "In-house vs Shen's datasets",
        "Steiner's vs Shen's datasets"
    ]
    
    # Add descriptive labels for each method
    # Ajustado o posicionamento dos labels do eixo Y
    fig.text(0.025, 0.69, 'zScales descriptors', fontsize=24, fontweight='bold', rotation=90)
    fig.text(0.025, 0.15, 'Rosetta energy terms', fontsize=24, fontweight='bold', rotation=90)
    
    # Blue plots (top row)
    in_house_blue, steiner_blue, triang_blue = blue_dfs
    
    # Plot a: In-house vs Steiner (Blue)
    corr1, p1 = plot_correlation(ax1, in_house_blue, steiner_blue, 
                               '#1f77b4', 'a', True, titles[0])
    
    # Plot b: In-house vs Shen (Blue)
    corr2, p2 = plot_correlation(ax2, in_house_blue, triang_blue,
                               '#1f77b4', 'b', True, titles[1])
    
    # Plot c: Steiner vs Shen (Blue)
    corr3, p3 = plot_correlation(ax3, steiner_blue, triang_blue,
                               '#1f77b4', 'c', True, titles[2])
    
    # Orange plots (bottom row) - no titles
    in_house_orange, steiner_orange, triang_orange = orange_dfs
    
    # Plot d: In-house vs Steiner (Orange)
    corr4, p4 = plot_correlation(ax4, in_house_orange, steiner_orange,
                               'orange', 'd')
    
    # Plot e: In-house vs Shen (Orange)
    corr5, p5 = plot_correlation(ax5, in_house_orange, triang_orange,
                               'orange', 'e')
    
    # Plot f: Steiner vs Shen (Orange)
    corr6, p6 = plot_correlation(ax6, steiner_orange, triang_orange,
                               'orange', 'f')
    
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=1, wspace=0.3, hspace=0.4)

# z-scales
in_house_blue = pd.read_csv("nfv/inhouse_mi_norm_values_nfv.csv")
steiner_blue = pd.read_csv("nfv/steiner_mi_norm_values_nfv.csv")
triang_blue = pd.read_csv("nfv/shen_mi_norm_values_nfv.csv")

# rosetta
in_house_orange = pd.read_csv("nfv/inhouse_mi_norm_values_nfv.csv")
steiner_orange = pd.read_csv("nfv/steiner_mi_norm_values_nfv.csv")
triang_orange = pd.read_csv("nfv/shen_mi_norm_values_nfv.csv")

# Create the combined plot
fig = create_combined_correlation_plots(
    (in_house_blue, steiner_blue, triang_blue),
    (in_house_orange, steiner_orange, triang_orange),
    axis_linewidth=2.0
)

# Save the figure
plt.savefig('combined_correlations_2x3.png', dpi=300, bbox_inches='tight')
plt.show()
