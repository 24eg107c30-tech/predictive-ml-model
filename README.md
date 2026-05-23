# Machine Learning Predictive Modeling Project

A comprehensive machine learning project demonstrating predictive modeling using multiple algorithms, including Linear Regression, Decision Trees, and Random Forest. This project includes complete workflows for data preprocessing, model training, evaluation, and advanced visualization.

## Features

✨ **Machine Learning Algorithms**
- Linear Regression for continuous prediction
- Decision Trees for interpretable classification and regression
- Random Forest for robust ensemble learning

📊 **Comprehensive Evaluation**
- Regression metrics: MSE, RMSE, MAE, R²
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for classification analysis
- ROC curves for binary classification
- Feature importance analysis

📈 **Advanced Visualization**
- Confusion matrices with heatmaps
- ROC curves with AUC scores
- Feature importance plots
- Learning curves for model analysis
- Model comparison charts

🔧 **Complete Pipeline**
- Data loading and preprocessing
- Missing value handling
- Categorical variable encoding
- Feature scaling and normalization
- Train-test split with stratification

## Project Structure

```
.
├── data/                     # Sample datasets
│   └── sample_data.csv      # Generated training data
├── notebooks/               # Jupyter notebooks for exploration
├── models/                  # Trained models and visualizations
│   ├── feature_importance_*.png
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── model_comparison.png
├── src/                     # Core modules
│   ├── main.py             # Main execution script
│   ├── data_preprocessing.py # DataPreprocessor class
│   ├── models.py            # ModelTrainer class
│   └── visualization.py     # ModelVisualizer class
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation & Setup

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import sklearn, pandas, numpy, matplotlib; print('All dependencies installed!')"
```

## Quick Start

### Run the Complete Pipeline

```bash
cd src
python main.py
```

This will:
1. Load and preprocess sample data
2. Train all models (Linear Regression, Decision Tree, Random Forest)
3. Evaluate model performance with comprehensive metrics
4. Generate visualization plots
5. Compare model performance

### Use Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Create new notebooks or use existing ones for interactive exploration and experimentation.

## Usage Examples

### Basic Usage: Data Preprocessing

```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(test_size=0.2)

# Load data
X = pd.read_csv('data.csv').drop('target', axis=1)
y = pd.read_csv('data.csv')['target']

# Preprocess
X_train, X_test, y_train, y_test = preprocessor.preprocess(
    X, y, 
    categorical_columns=['category_col1', 'category_col2']
)
```

### Train Models

```python
from models import ModelTrainer

# Initialize trainer for regression
trainer = ModelTrainer(model_type='regression')

# Train models
lr_model = trainer.linear_regression(X_train, y_train)
dt_model = trainer.decision_tree(X_train, y_train, max_depth=10)
rf_model = trainer.random_forest(X_train, y_train, n_estimators=100)

# Make predictions
predictions = trainer.predict('random_forest', X_test)

# Evaluate
metrics = trainer.evaluate_regression(y_test, predictions, 'Random Forest')
print(metrics)
```

### Visualize Results

```python
from visualization import ModelVisualizer

visualizer = ModelVisualizer()

# Plot confusion matrix (classification)
visualizer.plot_confusion_matrix(y_test, predictions, 'Random Forest')

# Plot ROC curve (binary classification)
visualizer.plot_roc_curve(y_test, pred_proba, 'Random Forest')

# Plot feature importance
importance = trainer.get_feature_importance('random_forest')
visualizer.plot_feature_importance(importance, X_train.columns, 'Random Forest')

# Compare models
visualizer.plot_model_comparison(trainer.results, metric='Accuracy')
```

## Key Components

### DataPreprocessor Class
Handles all data preprocessing tasks:
- Loading data from CSV
- Handling missing values (mean, median, most_frequent)
- Encoding categorical variables
- Feature scaling and normalization
- Train-test split

### ModelTrainer Class
Manages model training and evaluation:
- Linear Regression
- Decision Tree (classification & regression)
- Random Forest (classification & regression)
- Prediction generation
- Comprehensive metrics calculation
- Feature importance extraction

### ModelVisualizer Class
Creates publication-quality visualizations:
- Confusion matrices with annotations
- ROC curves with AUC scoring
- Feature importance bar plots
- Learning curves
- Multi-model comparison charts

## Configuration

Customize model training by modifying parameters:

```python
# Decision Tree with specific parameters
trainer.decision_tree(
    X_train, y_train,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    criterion='gini'  # or 'entropy' for classification
)

# Random Forest with custom settings
trainer.random_forest(
    X_train, y_train,
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1  # Use all available processors
)
```

## Performance Metrics

### Regression Metrics
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (0-1, higher is better)

### Classification Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of predictions vs actual values
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.0 | ML algorithms and metrics |
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computations |
| matplotlib | 3.7.2 | Plotting and visualization |
| seaborn | 0.12.2 | Statistical visualizations |
| jupyter | 1.0.0 | Interactive notebooks |

## Tips & Best Practices

1. **Always preprocess data** before training models
2. **Normalize features** for algorithms sensitive to scale (Linear Regression, KNN)
3. **Use cross-validation** for more reliable model evaluation
4. **Check feature importance** to understand model decisions
5. **Monitor for overfitting** using learning curves
6. **Tune hyperparameters** systematically (grid search, random search)
7. **Save trained models** for production use
8. **Document preprocessing steps** for reproducibility

## Troubleshooting

### Issue: ImportError for sklearn
**Solution**: Ensure scikit-learn is installed: `pip install scikit-learn`

### Issue: Out of memory with large datasets
**Solution**: Use Random Forest with `n_jobs=-1` or reduce dataset size for initial testing

### Issue: Poor model performance
**Solution**: 
- Check data preprocessing
- Verify feature scaling
- Tune hyperparameters
- Increase training data
- Check for data leakage

## Next Steps

1. **Load your own data** into the `data/` folder
2. **Modify preprocessing** for your specific needs
3. **Experiment with hyperparameters** in main.py
4. **Create custom notebooks** for specific analyses
5. **Integrate additional algorithms** (SVM, Gradient Boosting, Neural Networks)

## References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or suggestions, please refer to the project structure and update files as needed.


## Publishing to GitHub

To initialize a git repository and push this project to GitHub, run the following commands:

```bash
git init
git add .
git commit -m "Initial commit: predictive modeling project"
# Create a new repository on GitHub (use the GitHub UI or gh CLI), then:
git remote add origin <YOUR_REMOTE_URL>
git branch -M main
git push -u origin main
```

Replace `<YOUR_REMOTE_URL>` with your repository URL (for example `https://github.com/username/repo.git`).

## License

This project is provided under the MIT License. See the LICENSE file for details.

**Last Updated**: 2026-05-23
**Python Version**: 3.8+
