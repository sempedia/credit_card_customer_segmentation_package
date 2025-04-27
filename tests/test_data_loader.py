"""Tests for data loading utilities."""
import pytest
import pandas as pd
import numpy as np
from credit_card_segmentation.utils.data_loader import (
    load_customer_data,
    get_numeric_features,
    get_categorical_features,
    validate_customer_data
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3],
        'gender': ['M', 'F', 'M'],
        'education_level': ['Graduate', 'College', 'High School'],
        'marital_status': ['Single', 'Married', 'Single'],
        'age': [30, 45, 25],
        'months_on_book': [24, 36, 12],
        'credit_limit': [5000, 7000, 3000],
        'total_trans_amount': [1000, 2000, 500],
        'avg_utilization_ratio': [0.2, 0.3, 0.1]
    })

def test_get_numeric_features(sample_data):
    """Test getting numeric feature columns."""
    numeric_cols = get_numeric_features(sample_data)
    expected_cols = ['customer_id', 'age', 'months_on_book', 
                    'credit_limit', 'total_trans_amount', 
                    'avg_utilization_ratio']
    assert set(numeric_cols) == set(expected_cols)

def test_get_categorical_features(sample_data):
    """Test getting categorical feature columns."""
    cat_cols = get_categorical_features(sample_data)
    expected_cols = ['gender', 'education_level', 'marital_status']
    assert set(cat_cols) == set(expected_cols)

def test_validate_customer_data_valid(sample_data):
    """Test data validation with valid data."""
    assert validate_customer_data(sample_data) == True

def test_validate_customer_data_invalid():
    """Test data validation with invalid data."""
    invalid_data = pd.DataFrame({
        'customer_id': ['A', 'B', 'C'],  # Should be numeric
        'gender': [1, 2, 3],  # Should be object/string
        'age': [30, 45, 25]
    })
    assert validate_customer_data(invalid_data) == False

def test_load_customer_data(tmp_path):
    """Test loading customer data from CSV."""
    # Create a temporary CSV file
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'age': [30, 45, 25]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Test loading the file
    loaded_df = load_customer_data(str(csv_path))
    pd.testing.assert_frame_equal(df, loaded_df)