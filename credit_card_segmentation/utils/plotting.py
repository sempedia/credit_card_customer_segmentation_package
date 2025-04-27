"""Visualization utilities for credit card customer segmentation."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

def set_plotting_style():
    """Set consistent style for all plots."""
    sns.set_theme()  # Use seaborn's default theme instead of style.use
    sns.set_palette('Set2')

def plot_cluster_distributions(df: pd.DataFrame, numeric_columns: List[str], cluster_col: str = 'CLUSTER'):
    """Plot average values of numeric features for each cluster.
    
    Args:
        df: Input dataframe with cluster assignments
        numeric_columns: List of numeric columns to plot
        cluster_col: Name of the cluster column
    """
    fig = plt.figure(figsize=(20, 20))
    for i, column in enumerate(numeric_columns):
        df_plot = df.groupby(cluster_col)[column].mean()
        ax = fig.add_subplot(5, 2, i+1)
        ax.bar(df_plot.index, df_plot, color=sns.color_palette('Set1'), alpha=0.6)
        ax.set_title(f'Average {column.title()} per Cluster', alpha=0.5)
        ax.xaxis.grid(False)
    plt.tight_layout()
    return fig

def plot_cluster_relationships(df: pd.DataFrame, cluster_col: str = 'CLUSTER'):
    """Create scatter plots showing relationships between key features.
    
    Args:
        df: Input dataframe with cluster assignments
        cluster_col: Name of the cluster column
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    
    sns.scatterplot(x='age', y='months_on_book', hue=cluster_col, 
                    data=df, palette='tab10', alpha=0.4, ax=ax1)
    
    sns.scatterplot(x='estimated_income', y='credit_limit', hue=cluster_col,
                    data=df, palette='tab10', alpha=0.4, ax=ax2, legend=False)
    
    sns.scatterplot(x='credit_limit', y='avg_utilization_ratio', hue=cluster_col,
                    data=df, palette='tab10', alpha=0.4, ax=ax3)
    
    sns.scatterplot(x='total_trans_count', y='total_trans_amount', hue=cluster_col,
                    data=df, palette='tab10', alpha=0.4, ax=ax4, legend=False)
    
    plt.tight_layout()
    return fig

def plot_categorical_distributions(df: pd.DataFrame, cat_columns: List[str], cluster_col: str = 'CLUSTER'):
    """Plot distribution of categorical variables within clusters.
    
    Args:
        df: Input dataframe with cluster assignments
        cat_columns: List of categorical columns to plot
        cluster_col: Name of the cluster column
    """
    fig = plt.figure(figsize=(18, 6))
    
    for i, col in enumerate(cat_columns):
        plot_df = pd.crosstab(
            index=df[cluster_col], 
            columns=df[col], 
            values=df[col], 
            aggfunc='size', 
            normalize='index'
        )
        
        ax = fig.add_subplot(1, len(cat_columns), i+1)
        plot_df.plot.bar(stacked=True, ax=ax, alpha=0.6)
        ax.set_title(f'% {col.title()} per Cluster', alpha=0.5)
        ax.set_ylim(0, 1.4)
        ax.legend(frameon=False)
        ax.xaxis.grid(False)
    
    plt.tight_layout()
    return fig

def plot_elbow_curve(inertias: List[float]):
    """Plot elbow curve for K-means clustering.
    
    Args:
        inertias: List of inertia values for different numbers of clusters
    """
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(inertias) + 1), inertias, marker='o')
    plt.xticks(ticks=range(1, len(inertias) + 1))
    plt.title('Inertia vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.tight_layout()
    return plt.gcf()