"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Preprocesses raw data for machine learning models"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor
        
        Args:
            test_size: Proportion of dataset to include in test split
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        return pd.read_csv(filepath)
    
    def handle_missing_values(self, X, strategy='mean'):
        """Handle missing values in dataset"""
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = imputer.fit_transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns)
    
    def encode_categorical(self, X, categorical_columns):
        """Encode categorical variables"""
        X_encoded = X.copy()
        for col in categorical_columns:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le
        return X_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Standardize features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def preprocess(self, X, y, categorical_columns=None):
        """
        Complete preprocessing pipeline
        
        Args:
            X: Feature matrix
            y: Target variable
            categorical_columns: List of categorical column names
            
        Returns:
            X_train, X_test, y_train, y_test: Split and processed datasets
        """
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical variables
        if categorical_columns:
            X = self.encode_categorical(X, categorical_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
