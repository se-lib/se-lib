import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .model import DecisionMatrix

def plot_decision_matrix(matrix: DecisionMatrix, title="Decision Matrix"):
    """Plots a heatmap of the decision matrix scores."""
    df = matrix.to_dataframe()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_radar_comparison(matrix: DecisionMatrix, alternatives_to_plot: list[str] = None):
    """Plots a radar chart comparing selected alternatives."""
    df = matrix.to_dataframe()
    
    # Simple Min-Max normalization for plotting to keep on same scale 0-1
    norm_df = df.copy()
    for col in norm_df.columns:
        norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())
        norm_df[col] = norm_df[col].fillna(0.0) # Handle constant columns

    if alternatives_to_plot:
        plot_df = norm_df.loc[alternatives_to_plot]
    else:
        plot_df = norm_df

    labels = plot_df.columns.tolist()
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += [angles[0]]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for idx, row in plot_df.iterrows():
        values = row.tolist()
        values += [values[0]] # Close the loop
        ax.plot(angles, values, label=idx, linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Alternatives Comparison (Normalized)")
    plt.show()
