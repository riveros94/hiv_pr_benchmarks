# HIV Protease Inhibitor Resistance Analysis Pipeline
(Steiner Dataset Processing)

## Dataset Description

The dataset is based on Steiner et al. (2020) and includes HIV protease sequences with resistance data for eight protease inhibitors:
- Darunavir (DRV)
- Fosamprenavir (FPV)
- Atazanavir (ATV)
- Indinavir (IDV)
- Lopinavir (LPV)
- Nelfinavir (NFV)
- Saquinavir (SQV)
- Tipranavir (TPV)

## Directory Structure

```
.
├── Input Data
│   ├── atv.fasta              # Atazanavir sequences
│   ├── drv.fasta              # Darunavir sequences
│   ├── fpv.fasta              # Fosamprenavir sequences
│   ├── idv.fasta              # Indinavir sequences
│   ├── lpv.fasta              # Lopinavir sequences
│   ├── nfv.fasta              # Nelfinavir sequences
│   ├── sqv.fasta              # Saquinavir sequences
│   └── tpv.fasta              # Tipranavir sequences
├── Structure
│   └── 3oxc_edited.pdb        # Template structure
├── Scripts
│   ├── fasta_to_csv.py        # FASTA preprocessing
│   ├── delaunay_triangulation.py    # Structural analysis
│   ├── rosetta_modeling.py          # Rosetta modeling
│   └── pos_rosetta_processing.py    # Feature processing
└── Models
    ├── nn_models/             # Neural network models
    ├── knn_rf/               # KNN and Random Forest models
    ├── rosetta_lr/           # Rosetta Linear Regression models
    └── zscales_lr/           # Z-scales Linear Regression models
```

## Processing Pipeline

### 1. Data Preprocessing
```bash
python fasta_to_csv.py
```
- Filters sequences containing special characters (*, ~, #, .)
- Converts FASTA files to CSV format
- Splits data into training and test sets for neural network models
- Output is used for models in the nn_models directory

### 2. Structural Analysis
```bash
python delaunay_triangulation.py
```
This script performs structural analysis using:
- Template structure: 3oxc_edited.pdb
- Delaunay triangulation for distance calculations
- Generates structural features for KNN and RF models

### 3. Rosetta Feature Generation
```bash
python rosetta_modeling.py
python pos_rosetta_processing.py
```
The Rosetta pipeline:
1. Generates structural models using Rosetta
2. Formats data for machine learning model training
   - Specifically prepares features for Rosetta Linear Regression models
   - Creates standardized input format for model training

