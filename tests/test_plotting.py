"""Tests for visualization utilities."""
import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from credit_card_segmentation.utils.plotting import (
    set_plotting_style,
    plot_cluster_distributions,
    plot_cluster_relationships,
    plot_categorical_distributions,
    plot_elbow_curve
)

@pytest.fixture
def sample_data():
    """Create sample data for testing visualizations."""
    np.random.seed(42)
    return pd.DataFrame({
        'customer_id': range(100),
        'age': np.random.normal(40, 10, 100),
        'months_on_book': np.random.normal(36, 12, 100),
        'credit_limit': np.random.normal(5000, 1000, 100),
        'estimated_income': np.random.normal(50000, 10000, 100),
        'avg_utilization_ratio': np.random.uniform(0, 1, 100),
        'total_trans_count': np.random.poisson(30, 100),
        'total_trans_amount': np.random.normal(3000, 500, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'education_level': np.random.choice(['Graduate', 'College', 'High School'], 100),
        'CLUSTER': np.random.randint(1, 5, 100)
    })

def test_set_plotting_style():
    """Test setting plotting style."""
    set_plotting_style()
    # No assertion needed - function should run without errors

def test_plot_cluster_distributions(sample_data):
    """Test plotting cluster distributions."""
    numeric_cols = ['age', 'credit_limit', 'estimated_income']
    fig = plot_cluster_distributions(sample_data, numeric_cols)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(numeric_cols)

def test_plot_cluster_relationships(sample_data):
    """Test plotting cluster relationships."""
    fig = plot_cluster_relationships(sample_data)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # 2x2 grid of plots

def test_plot_categorical_distributions(sample_data):
    """Test plotting categorical distributions."""
    cat_cols = ['gender', 'education_level']
    fig = plot_categorical_distributions(sample_data, cat_cols)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == len(cat_cols)

def test_plot_elbow_curve():
    """Test plotting elbow curve."""
    inertias = [100, 80, 60, 45, 35, 30, 25, 22, 20, 18]
    fig = plot_elbow_curve(inertias)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1