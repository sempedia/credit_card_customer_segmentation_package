"""Feature engineering module for credit card customer segmentation."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def encode_gender(df: pd.DataFrame, column: str = 'gender') -> pd.DataFrame:
    """Encode gender column to numeric values."""
    df = df.copy()
    df[column] = df[column].map({'M': 1, 'F': 0})
    return df

def encode_education(df: pd.DataFrame, column: str = 'education_level') -> pd.DataFrame:
    """Encode education levels using ordinal encoding."""
    df = df.copy()
    education_mapping = {
        'Uneducated': 0, 
        'High School': 1, 
        'College': 2,
        'Graduate': 3, 
        'Post-Graduate': 4, 
        'Doctorate': 5
    }
    df[column] = df[column].map(education_mapping)
    return df

def encode_marital_status(df: pd.DataFrame, column: str = 'marital_status') -> pd.DataFrame:
    """Encode marital status using one-hot encoding."""
    df = df.copy()
    dummies = pd.get_dummies(df[[column]], prefix=column)
    df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return df

def scale_features(df: pd.DataFrame, exclude_cols: list = None) -> tuple:
    """Scale numeric features using StandardScaler."""
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    df_scaled = df.copy()
    if not cols_to_scale:
        return df_scaled, None
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    df_scaled[cols_to_scale] = pd.DataFrame(
        scaler.fit_transform(df[cols_to_scale]),
        columns=cols_to_scale,
        index=df.index
    )
    
    return df_scaled, scaler

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare all features for clustering."""
    df = df.copy()
    
    # First encode categorical variables (these should not be scaled)
    df = encode_gender(df)
    df = encode_education(df)
    df = encode_marital_status(df)
    
    # Get encoded categorical columns to exclude from scaling
    categorical_cols = ['gender', 'education_level'] + \
                      [col for col in df.columns if col.startswith('marital_status_')]
    
    # Scale only numeric features, excluding customer_id and encoded categoricals
    exclude_from_scaling = ['customer_id'] + categorical_cols
    df_scaled, _ = scale_features(df, exclude_cols=exclude_from_scaling)
    
    return df_scaled