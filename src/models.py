"""
Machine Learning Models Module
Implements Linear Regression, Decision Trees, and Random Forest
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np


class ModelTrainer:
    """Trains and evaluates ML models"""
    
    def __init__(self, model_type='regression'):
        """
        Initialize model trainer
        
        Args:
            model_type: 'regression' or 'classification'
        """
        self.model_type = model_type
        self.models = {}
        self.results = {}
        
    def linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def decision_tree(self, X_train, y_train, **kwargs):
        """
        Train Decision Tree model
        
        Args:
            **kwargs: Parameters like max_depth, min_samples_split, etc.
        """
        if self.model_type == 'regression':
            model = DecisionTreeRegressor(**kwargs)
        else:
            model = DecisionTreeClassifier(**kwargs)
        
        model.fit(X_train, y_train)
        self.models['decision_tree'] = model
        return model
    
    def random_forest(self, X_train, y_train, n_estimators=100, **kwargs):
        """
        Train Random Forest model
        
        Args:
            n_estimators: Number of trees in forest
            **kwargs: Additional parameters
        """
        if self.model_type == 'regression':
            model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def evaluate_regression(self, y_true, y_pred, model_name):
        """Evaluate regression model"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_classification(self, y_true, y_pred, model_name):
        """Evaluate classification model"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Confusion_Matrix': cm
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def predict(self, model_name, X):
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")
        
        return self.models[model_name].predict(X)
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            return None
