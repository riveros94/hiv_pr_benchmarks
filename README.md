# Benchmarking Machine Learning Approaches for HIV-1 Protease Inhibitor Resistance Prediction: Impact of Dataset Construction and Feature Representation

## Overview

This repository contains a comprehensive benchmarking framework for evaluating machine learning models that predict HIV-1 protease inhibitor resistance from sequence data. The project systematically compares how different dataset construction strategies impact model performance across diverse machine learning approaches. By standardizing evaluation methodologies, we provide insights into which combination of data processing and modeling approaches yields optimal prediction capabilities for this critical clinical application.

## Repository Structure

```
.
├── data/                     # Dataset processing pipelines for three approaches
├── model_scripts/            # Model implementations organized by test dataset
├── feature_analysis/         # Feature importance analysis tools
└── time_test/                # Computational performance benchmarking
```

## Dataset Approaches

We implement three distinct strategies for processing HIV-1 sequence data:

1. **In-house Dataset (Clustering-based)**: 
   - Strict filtering to exclude sequences with ambiguities and ensure data quality
   - Test set construction using t-SNE and K-means clustering to ensure diverse sequence representation
   - Emphasizes comprehensive evaluation across sequence diversity

2. **Shen Dataset (Sequence Expansion)**: 
   - Expands sequences with ambiguous amino acids into multiple individual sequences
   - Creates larger datasets with identical resistance values for expanded sequences
   - Uses random 80:20 train/test split

3. **Steiner Dataset (Filtered)**: 
   - Excludes sequences with ambiguities at major drug resistance mutation positions
   - For remaining ambiguities, selects the first listed amino acid
   - Uses random 80:20 train/test split

## Models Evaluated

- **Multilayer Perceptron (MLP)**
- **Bidirectional Recurrent Neural Network (BRNN)**
- **Convolutional Neural Network (CNN)**
- **Random Forest (RF)**
- **K-Nearest Neighbors (KNN)**
- **zScales Logistic Regression**
- **Rosetta Logistic Regression**

## Feature Analysis

Tools for analyzing the importance of specific features:
- Mutual information calculations for both zScales descriptors and Rosetta energy terms
- Correlation analysis between feature importance across different dataset preparations
- Visualization of important positions on protein structures

## Performance Benchmarking

Comprehensive evaluation of both predictive and computational performance:
- Prediction accuracy, precision, recall, and AUC across all models and datasets
- Computational efficiency measurements (prediction time per sequence)
- Memory usage requirements for each approach

## Requirements

### For R-based models:
- R v3.6.1+
- Keras v2.11.1+
- TensorFlow v2.12.0+
- Additional packages: tidyverse, caret, pROC

### For Python-based models:
- Python 3.10.8+
- scikit-learn v1.5.2+
- NumPy v1.24.4+
- pandas
- BioPython
- pyRosetta v4.2+ (for Rosetta energy calculations)

## Usage

### Data Processing
```bash
# Process data using the in-house approach
cd data/inhouse_data
Rscript data_processing.R

# Process data using the Shen approach
cd data/shen_data
bash run_sequence_expansion.sh

# Process data using the Steiner approach
cd data/steiner_data
python process_steiner_data.py
```

### Model Training and Evaluation
```bash
# Train and evaluate MLP model
cd models/deep_learning/mlp
Rscript mlp_run_script.R

# Train and evaluate zScales LR model
cd models/interpretable_models/zscales_lr
python zscales_lr.py
```

### Feature Analysis
```bash
# Calculate mutual information for zScales descriptors
cd feature_analysis
python zscales_mutual_information.py
```
