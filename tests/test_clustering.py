"""Tests for clustering module."""
import pytest
import numpy as np
import pandas as pd
from credit_card_segmentation.src.clustering import (
    find_optimal_clusters,
    perform_clustering,
    get_cluster_statistics
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    # Create 3 distinct clusters
    cluster1 = np.random.normal(0, 1, (20, 2))
    cluster2 = np.random.normal(5, 1, (20, 2))
    cluster3 = np.random.normal(-5, 1, (20, 2))
    X = np.vstack([cluster1, cluster2, cluster3])
    return X

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing cluster statistics."""
    return pd.DataFrame({
        'customer_id': range(60),
        'age': np.random.normal(40, 10, 60),
        'income': np.random.normal(50000, 10000, 60),
        'gender': np.random.choice(['M', 'F'], 60)
    })

def test_find_optimal_clusters(sample_data):
    """Test finding optimal number of clusters."""
    inertias = find_optimal_clusters(sample_data, max_clusters=5)
    assert len(inertias) == 5
    assert all(isinstance(x, (int, float)) for x in inertias)
    assert all(x > 0 for x in inertias)
    # Inertia should decrease with more clusters
    assert all(inertias[i] > inertias[i+1] for i in range(len(inertias)-1))

def test_perform_clustering(sample_data):
    """Test K-means clustering."""
    labels, model = perform_clustering(sample_data, n_clusters=3)
    assert len(labels) == len(sample_data)
    assert all(label in [0, 1, 2] for label in labels)
    assert hasattr(model, 'cluster_centers_')
    assert model.cluster_centers_.shape == (3, 2)

def test_get_cluster_statistics(sample_df):
    """Test cluster statistics calculation."""
    # Create sample cluster labels
    labels = np.random.choice([0, 1, 2], len(sample_df))
    stats = get_cluster_statistics(sample_df, labels)
    
    # Check if statistics are calculated correctly
    assert isinstance(stats, pd.DataFrame)
    assert len(stats.index.unique()) == 3  # 3 clusters
    assert ('age', 'mean') in stats.columns
    assert ('income', 'mean') in stats.columns
    assert ('gender', '') in stats.columns  # Mode for categorical