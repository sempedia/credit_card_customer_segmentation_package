"""Data loading utilities for credit card customer segmentation."""
import pandas as pd
import numpy as np
from typing import List, Union

def load_customer_data(file_path: str) -> pd.DataFrame:
    """Load customer data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing customer data
        
    Returns:
        pd.DataFrame: Loaded customer data
    """
    df = pd.read_csv(file_path)
    return df

def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """Get list of numeric feature columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        List[str]: List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """Get list of categorical feature columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        List[str]: List of categorical column names
    """
    return df.select_dtypes(include=['object']).columns.tolist()

def validate_customer_data(df: pd.DataFrame) -> bool:
    """Validate that the customer data has required columns and formats.
    
    Args:
        df: Input dataframe to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    required_columns = {
        'customer_id': np.number,
        'gender': 'object',
        'education_level': 'object',
        'marital_status': 'object',
        'age': np.number,
        'months_on_book': np.number,
        'credit_limit': np.number,
        'total_trans_amount': np.number,
        'avg_utilization_ratio': np.number
    }
    
    # Check if all required columns exist
    missing_cols = set(required_columns.keys()) - set(df.columns)
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Check column types
    for col, dtype in required_columns.items():
        if dtype == np.number:
            if not np.issubdtype(df[col].dtype, np.number):
                print(f"Column {col} should be numeric but is {df[col].dtype}")
                return False
        elif df[col].dtype != dtype:
            print(f"Column {col} should be {dtype} but is {df[col].dtype}")
            return False
    
    return True