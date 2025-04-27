"""Clustering module for credit card customer segmentation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple, List

def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10) -> List[float]:
    """Calculate inertia for different numbers of clusters.
    
    Args:
        X: Input features array
        max_clusters: Maximum number of clusters to try
        
    Returns:
        List[float]: List of inertia values for each number of clusters
    """
    inertias = []
    for k in range(1, max_clusters + 1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)
    return inertias

def perform_clustering(X: np.ndarray, n_clusters: int = 8) -> Tuple[np.ndarray, KMeans]:
    """Perform K-means clustering on the data.
    
    Args:
        X: Input features array
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple[np.ndarray, KMeans]: Cluster labels and fitted model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def get_cluster_statistics(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Calculate statistics for each cluster.
    
    Args:
        df: Original dataframe with features
        labels: Cluster labels from KMeans
        
    Returns:
        pd.DataFrame: Dataframe with cluster statistics
    """
    df = df.copy()
    df['Cluster'] = labels + 1  # Add 1 to make clusters 1-based
    
    # Calculate basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cluster_stats = df.groupby('Cluster')[numeric_cols].agg(['mean', 'std', 'count'])
    
    # Calculate mode for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            cluster_stats[col, ''] = df.groupby('Cluster')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
    return cluster_stats