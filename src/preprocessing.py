"""Feature preprocessing pipeline for NSL-KDD dataset.

Implements Min-Max scaling for numerical features and One-Hot Encoding for
categorical features, strictly fitting only on training data to prevent data leakage.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from src.data import FEATURE_NAMES, CLASS_NAMES

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [col for col in FEATURE_NAMES if col not in CATEGORICAL_FEATURES]


class Preprocessor:
    """Encapsulates numerical scaling and categorical encoding for NSL-KDD."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CLASS_NAMES)  # Consistent class index ordering
        self.is_fitted = False
        self.encoded_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'Preprocessor':
        """Fit scaler on numerical columns and encoder on categorical columns."""
        # Ensure input has the expected columns
        X_df = X[FEATURE_NAMES].copy()

        # Fit numerical scaler
        self.scaler.fit(X_df[NUMERICAL_FEATURES].astype(float))

        # Fit categorical encoder
        self.encoder.fit(X_df[CATEGORICAL_FEATURES].astype(str))

        # Determine feature names out
        cat_feature_names = self.encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.encoded_feature_names_ = NUMERICAL_FEATURES + cat_feature_names

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform numerical and categorical features using fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        X_df = X[FEATURE_NAMES].copy()

        num_transformed = self.scaler.transform(X_df[NUMERICAL_FEATURES].astype(float))
        cat_transformed = self.encoder.transform(X_df[CATEGORICAL_FEATURES].astype(str))

        return np.hstack([num_transformed, cat_transformed])

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(X, y).transform(X)

    def encode_labels(self, y: Union[pd.Series, np.ndarray, List[str]]) -> np.ndarray:
        """Encode string labels (Normal, DoS, Probe, R2L, U2R) to integer indices."""
        return self.label_encoder.transform(y)

    def decode_labels(self, y_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        """Decode integer indices back to string class labels."""
        return self.label_encoder.inverse_transform(y_indices)
