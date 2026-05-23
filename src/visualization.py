"""
Visualization Module
Creates performance visualizations including ROC curves and confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


class ModelVisualizer:
    """Visualizes model performance metrics"""
    
    def __init__(self, figsize=(10, 6)):
        """Initialize visualizer"""
        self.figsize = figsize
        sns.set_style("whitegrid")
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, save_path=None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return roc_auc
    
    def plot_feature_importance(self, feature_importance, feature_names, 
                                model_name, save_path=None, top_n=10):
        """
        Plot feature importance
        
        Args:
            feature_importance: Array of feature importances
            feature_names: List of feature names
            model_name: Name of the model
            save_path: Path to save figure
            top_n: Number of top features to display
        """
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(self.figsize[0], 6))
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.bar(range(top_n), feature_importance[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results, metric='Accuracy', save_path=None):
        """
        Plot comparison of multiple models
        
        Args:
            results: Dictionary of model results
            metric: Metric to compare
            save_path: Path to save figure
        """
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        plt.figure(figsize=self.figsize)
        plt.bar(models, values, color='skyblue', edgecolor='navy')
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curve(self, model, X_train, y_train, cv=5, save_path=None):
        """
        Plot learning curve
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            cv: Cross-validation splits
            save_path: Path to save figure
        """
        from sklearn.model_selection import learning_curve
        from sklearn.metrics import mean_squared_error
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )
        
        train_mean = -train_scores.mean(axis=1)
        val_mean = -val_scores.mean(axis=1)
        
        plt.figure(figsize=self.figsize)
        plt.plot(train_sizes, train_mean, 'o-', label='Training error')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation error')
        plt.xlabel('Training Set Size')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
