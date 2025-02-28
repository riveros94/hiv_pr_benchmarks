# HIV-1 Protease Inhibitor Resistance Prediction Models

This repository contains the implementation of various machine learning models for predicting HIV-1 protease inhibitor resistance. The models are evaluated on three different dataset preprocessing approaches to assess their impact on prediction performance.

## Repository Structure

The repository is organized into three main directories, each corresponding to a different test dataset construction approach:

```
.
├── inhouse_test/       # Models evaluated on the clustering-based test set
├── shen_test/          # Models evaluated on Shen's expanded ambiguities test set
└── steiner_test/       # Models evaluated on Steiner's filtered test set
```

Each directory contains identical model implementations adapted for the specific test set:

## Models Implemented

### Deep Learning Models (R/Keras)
- **mlp_run_script.R**: Multilayer Perceptron implementation with embedding and 4 feed-forward hidden layers
- **brnn_run_script.R**: Bidirectional Recurrent Neural Network with LSTM implementation
- **cnn_run_script.R**: Convolutional Neural Network implementation with 2 convolutional layers

### Traditional Machine Learning Models (Python)
- **rf.py**: Random Forest implementation using Delaunay triangulation for feature extraction
- **knn.py**: K-Nearest Neighbors implementation (K=6) using Delaunay triangulation for feature extraction

### Interpretable Models (Python)
- **zscales_lr.py**: Logistic Regression using zScales amino acid descriptors
- **rosetta_lr.py**: Logistic Regression using Rosetta Energy Function terms

## Dataset Approaches

1. **In-house test**: Sequences with ambiguities were excluded. Test set construction used t-SNE clustering to ensure diverse sequence representation.

2. **Shen test**: Sequences with ambiguities were expanded into all possible combinations. Test set was created using random 80:20 split.

3. **Steiner test**: Sequences with ambiguities at major drug resistance mutation positions were excluded. For remaining ambiguities, the first listed amino acid was selected. Test set was created using random 80:20 split.

## Requirements

### For R models:
- R v3.6.1 
- Keras v2.11.1
- TensorFlow v2.12.0
- Additional packages: tidyverse, caret, pROC

### For Python models:
- Python 3.10.8
- scikit-learn v1.5.2
- NumPy v1.24.4
- pandas
- BioPython (for sequence processing)
- pyRosetta v4.2 (for Rosetta energy calculations)


