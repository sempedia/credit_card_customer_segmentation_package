"""Tests for feature engineering module."""
import pytest
import pandas as pd
import numpy as np
from credit_card_segmentation.src.feature_engineering import (
    encode_gender,
    encode_education,
    encode_marital_status,
    scale_features,
    prepare_features
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
        'income': [50000, 60000, 45000]
    })

def test_encode_gender(sample_data):
    """Test gender encoding."""
    result = encode_gender(sample_data)
    assert all(result['gender'].isin([0, 1]))
    assert result['gender'].tolist() == [1, 0, 1]

def test_encode_education(sample_data):
    """Test education level encoding."""
    result = encode_education(sample_data)
    assert result['education_level'].dtype == np.int64
    assert result['education_level'].tolist() == [3, 2, 1]

def test_encode_marital_status(sample_data):
    """Test marital status encoding."""
    result = encode_marital_status(sample_data)
    assert 'marital_status' not in result.columns
    assert 'marital_status_Single' in result.columns
    assert 'marital_status_Married' in result.columns

def test_scale_features(sample_data):
    """Test feature scaling."""
    numeric_cols = ['age', 'income']
    result, scaler = scale_features(sample_data[numeric_cols])
    
    # Check if scaled data has mean close to 0 and std close to 1
    assert np.abs(result['age'].mean()) < 1e-10
    assert np.abs(result['income'].mean()) < 1e-10
    assert np.abs(result['age'].std(ddof=0) - 1) < 1e-10  # Use ddof=0 to match sklearn
    assert np.abs(result['income'].std(ddof=0) - 1) < 1e-10  # Use ddof=0 to match sklearn

def test_prepare_features(sample_data):
    """Test complete feature preparation pipeline."""
    result = prepare_features(sample_data)
    
    # Check if categorical variables are properly encoded
    assert 'gender' in result.columns
    assert all(result['gender'].isin([0, 1]))
    assert 'education_level' in result.columns
    assert 'marital_status_Single' in result.columns
    
    # Check if numeric features are scaled
    numeric_cols = ['age', 'income']
    for col in numeric_cols:
        assert np.abs(result[col].mean()) < 1e-10
        assert np.abs(result[col].std(ddof=0) - 1) < 1e-10  # Use ddof=0 to match sklearn