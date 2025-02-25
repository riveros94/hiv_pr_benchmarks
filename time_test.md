# Performance Benchmarking of Machine Learning Models for HIV-1 Protease Inhibitor Resistance Prediction

## README

This repository contains the code used to benchmark the computational performance of different machine learning approaches for predicting HIV-1 protease inhibitor resistance. The analysis focuses on comparing prediction times across various model architectures when evaluating a single HIV-1 protease sequence against NFV (Nelfinavir) resistance.

### Computational Performance Analysis

We evaluated the computational efficiency of each resistance prediction model by measuring prediction time for a single HIV-1 protease sequence against NFV resistance. All measurements were conducted on an AMD EPYC 7662 64-Core Processor using standardized computing resources (1 CPU core, 10GB RAM) allocated through SLURM workload manager. The total prediction time was measured from sequence input through feature transformation to final prediction output. 

For different model types, the process includes:
- **Deep learning models (MLP, CNN, BRNN)**: Sequence encoding and prediction
- **KNN and RF models**: Delaunay triangulation and 210-dimensional feature vector generation
- **zScales LR model**: Physicochemical descriptor calculation
- **Rosetta LR model**: Structure modeling, energy minimization, and REF15 energy term extraction

Each model was run 100 times to ensure stable measurements. This standardized testing environment enables direct comparison of computational requirements between different approaches under realistic deployment conditions.

### Repository Structure

The repository contains timing scripts for each model type:

```
.
├── brnn_test_time.R       # Timing script for BRNN models
├── cnn_test_time.R        # Timing script for CNN models
├── knn_test_time.py       # Timing script for KNN models
├── mlp_test_time.R        # Timing script for MLP models
├── rf_test_time.py        # Timing script for Random Forest models
├── rosetta_test_time.py   # Timing script for Rosetta-based LR models
└── zscales_time_test.py   # Timing script for zScales-based LR models
```

### Data and Model Files

All model files and input data required to run the benchmarks are available at Zenodo:

[Zenodo Link to be added]

Please download these files and extract them to the appropriate directories before running the benchmarks.

### Installation and Setup

1. Clone this repository
2. Download the model files and data from the Zenodo link
3. Extract the downloaded files to the repository directory
4. Ensure you have the required dependencies installed:
   - R with required packages for R scripts (keras, tensorflow, etc.)
   - Python with required packages for Python scripts (numpy, scikit-learn, PyRosetta, etc.)

### Usage

To reproduce the performance benchmarks:

1. Download and place the model files in the repository directory
2. Run the timing script for the model of interest:
   ```
   # For R scripts
   Rscript brnn_test_time.R
   
   # For Python scripts
   python knn_test_time.py
   ```

Each script will load the corresponding pre-trained models and measure the time required to make predictions for the test sequence.
