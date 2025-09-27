# Data-Insight-Engine
A Python tool for automatically generating data insights from any CSV dataset. This engine performs data exploration, preprocessing, outlier detection, correlation analysis, and optionally builds a classification model with feature importance—ideal for analysts, data scientists, or anyone who wants actionable insights quickly.

# Features
- Automatic package installation if dependencies are missing.
- Supports any CSV dataset with numeric and categorical columns.
- Cleans missing values automatically.
- Generates basic statistics for numeric features.
- Creates correlation heatmaps and highlights strong relationships.
- Detects outliers using Isolation Forest.
- Optional classification modeling if a target column is provided.
- Shows confusion matrix, classification report, and SHAP-based feature importance.
- Optimized for speed with tree-based models (RandomForest) for SHAP.

# Getting Started
- Python 3.8+
- Internet connection to install missing packages

# Installation
Clone the repository or download the script directly:

```
git clone https://github.com/munashez98/data-insight-engine.git
cd data-insight-engine
```

The script will:

1. Prompt for your CSV dataset path.
2. Detect numeric and categorical columns.
3. Handle missing values automatically.
4. Display basic statistics and correlation heatmaps.
5. Detect potential outliers.
6. Optionally, prompt for a target column to perform classification.
7. Generate a confusion matrix, classification report, and feature importance plot.

# Example
```
Enter the path to your CSV dataset: data/iris.csv
Enter the target column for classification: species
```

The script will output:

- First 5 rows of the dataset
- Numeric feature statistics
- Correlation heatmap
- Detected outliers
- Classification test accuracy
- Confusion matrix
- Feature importance using SHAP

## Datasets to Try:
[Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) \
[Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset/data)

# Notes
- Works best with structured tabular data.
- For very large datasets, SHAP feature importance may take longer to compute.
- If no target column is provided, the script still generates exploratory insights like correlations and outliers.

