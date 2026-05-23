"""
Main Script - Machine Learning Predictive Modeling Pipeline
Demonstrates complete workflow: load data, preprocess, train, evaluate, visualize
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from data_preprocessing import DataPreprocessor
from models import ModelTrainer
from visualization import ModelVisualizer


def create_sample_data():
    """Create sample dataset using Iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Save to CSV
    df = X.copy()
    df['target'] = y
    df.to_csv('../data/sample_data.csv', index=False)
    
    return X, y, iris.feature_names.tolist()


def run_regression_pipeline():
    """Run regression pipeline on Boston Housing-like data"""
    print("\n" + "="*60)
    print("REGRESSION PIPELINE")
    print("="*60)
    
    # Load sample data
    X, y, feature_names = create_sample_data()
    
    # Preprocess data
    print("\n[1] Preprocessing data...")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train models
    print("\n[2] Training models...")
    trainer = ModelTrainer(model_type='regression')
    
    # Linear Regression
    print("  - Training Linear Regression...")
    lr_model = trainer.linear_regression(X_train, y_train)
    y_pred_lr = trainer.predict('linear_regression', X_test)
    lr_metrics = trainer.evaluate_regression(y_test, y_pred_lr, 'Linear Regression')
    print(f"    R² Score: {lr_metrics['R2']:.4f}, RMSE: {lr_metrics['RMSE']:.4f}")
    
    # Decision Tree
    print("  - Training Decision Tree...")
    dt_model = trainer.decision_tree(X_train, y_train, max_depth=5)
    y_pred_dt = trainer.predict('decision_tree', X_test)
    dt_metrics = trainer.evaluate_regression(y_test, y_pred_dt, 'Decision Tree')
    print(f"    R² Score: {dt_metrics['R2']:.4f}, RMSE: {dt_metrics['RMSE']:.4f}")
    
    # Random Forest
    print("  - Training Random Forest...")
    rf_model = trainer.random_forest(X_train, y_train, n_estimators=100)
    y_pred_rf = trainer.predict('random_forest', X_test)
    rf_metrics = trainer.evaluate_regression(y_test, y_pred_rf, 'Random Forest')
    print(f"    R² Score: {rf_metrics['R2']:.4f}, RMSE: {rf_metrics['RMSE']:.4f}")
    
    # Visualize results
    print("\n[3] Visualizing results...")
    visualizer = ModelVisualizer()
    
    # Feature importance
    try:
        dt_importance = trainer.get_feature_importance('decision_tree')
        if dt_importance is not None:
            visualizer.plot_feature_importance(
                dt_importance, X_train.columns, 'Decision Tree',
                save_path='../models/feature_importance_dt.png'
            )
    except:
        print("  - Could not plot feature importance")
    
    # Model comparison
    results = {
        'Linear Regression': lr_metrics,
        'Decision Tree': dt_metrics,
        'Random Forest': rf_metrics
    }
    visualizer.plot_model_comparison(results, metric='R2', save_path='../models/model_comparison.png')
    
    print("\n✓ Regression pipeline completed!")
    return trainer, preprocessor, visualizer


def run_classification_pipeline():
    """Run classification pipeline"""
    print("\n" + "="*60)
    print("CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Load sample data
    X, y, feature_names = create_sample_data()
    
    # Preprocess data
    print("\n[1] Preprocessing data...")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train models
    print("\n[2] Training models...")
    trainer = ModelTrainer(model_type='classification')
    
    # Decision Tree
    print("  - Training Decision Tree Classifier...")
    dt_model = trainer.decision_tree(X_train, y_train, max_depth=5)
    y_pred_dt = trainer.predict('decision_tree', X_test)
    dt_metrics = trainer.evaluate_classification(y_test, y_pred_dt, 'Decision Tree')
    print(f"    Accuracy: {dt_metrics['Accuracy']:.4f}, F1-Score: {dt_metrics['F1-Score']:.4f}")
    
    # Random Forest
    print("  - Training Random Forest Classifier...")
    rf_model = trainer.random_forest(X_train, y_train, n_estimators=100)
    y_pred_rf = trainer.predict('random_forest', X_test)
    rf_metrics = trainer.evaluate_classification(y_test, y_pred_rf, 'Random Forest')
    print(f"    Accuracy: {rf_metrics['Accuracy']:.4f}, F1-Score: {rf_metrics['F1-Score']:.4f}")
    
    # Visualize results
    print("\n[3] Visualizing results...")
    visualizer = ModelVisualizer()
    
    # Confusion matrices
    visualizer.plot_confusion_matrix(y_test, y_pred_dt, 'Decision Tree', 
                                    save_path='../models/cm_dt.png')
    visualizer.plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest',
                                    save_path='../models/cm_rf.png')
    
    # Feature importance
    try:
        rf_importance = trainer.get_feature_importance('random_forest')
        if rf_importance is not None:
            visualizer.plot_feature_importance(
                rf_importance, X_train.columns, 'Random Forest',
                save_path='../models/feature_importance_rf.png'
            )
    except:
        print("  - Could not plot feature importance")
    
    print("\n✓ Classification pipeline completed!")
    return trainer, preprocessor, visualizer


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PREDICTIVE MODELING USING MACHINE LEARNING")
    print("="*60)
    
    # Run both pipelines
    print("\nRunning regression and classification pipelines...")
    
    # Regression
    try:
        reg_trainer, reg_prep, reg_viz = run_regression_pipeline()
    except Exception as e:
        print(f"Error in regression pipeline: {e}")
    
    # Classification
    try:
        clf_trainer, clf_prep, clf_viz = run_classification_pipeline()
    except Exception as e:
        print(f"Error in classification pipeline: {e}")
    
    print("\n" + "="*60)
    print("ALL PIPELINES COMPLETED SUCCESSFULLY!")
    print("="*60)
