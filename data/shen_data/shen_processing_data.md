# HIV Protease Inhibitor Resistance Data Processing Protocol

## Dataset Description
The dataset comprises HIV protease sequences and their resistance profiles against eight protease inhibitors (PIs):
- Darunavir (DRV)
- Fosamprenavir (FPV)
- Atazanavir (ATV)
- Indinavir (IDV)
- Lopinavir (LPV)
- Nelfinavir (NFV)
- Saquinavir (SQV)
- Tipranavir (TPV)

### Source
- Stanford University HIV Drug Resistance Database (Rhee et al. 2003)
- Download date: February 7, 2023
- Data type: Subtype B sequences with PhenoSense assay measurements

### Resistance Classification
Sequences are classified as resistant (1) or susceptible (0) based on established clinical cutoffs of fold change in IC50:
- TPV: 2.0
- NFV/SQV/IDV/ATV: 3.0
- FPV: 4.0
- LPV: 9.0
- DRV: 10.0

## Data Processing Pipeline

### 1. Sequence Expansion
./run_sequence_expansion.sh

This script processes ambiguous sequences according to the Shen dataset methodology (Shen et al. 2016):
- Expands sequences containing ambiguous amino acids into multiple sequences
- Maintains identical resistance values for expanded sequences
- Excludes sequences with insertions, deletions, or stop codons
- Implementation: Python v3.10.8


### 2. Data Splitting for Neural Networks
python fasta_train_test_split.py

This script prepares the data for training different neural network architectures (Steiner, 2020):

Multilayer Perceptron (MLP)
Bidirectional Recurrent Neural Network (BRNN)
Convolutional Neural Network (CNN)

The script:

- Splits the sequence data into training and test sets
- Transform into fasta files per drug

### 3. Structural Analysis
python delaunay_triangulation.py

Performs structural analysis using:
- Template structure: 3oxc_edited.pdb
- Delaunay triangulation for distance calculations
- Generates structural features for machine learning

### 3. Rosetta Feature Generation
python rosetta_modeling.py
python pos_rosetta_processing.py

These scripts:
1. Generate structural models using Rosetta
2. Process the models to extract features
3. Format data for machine learning model training (specifically for Rosetta Linear Regression models)


## Directory Structure

```
.
├── PI_DataSet.Full.txt          # Raw dataset from Stanford DBd
├── run_sequence_expansion.sh    # Sequence expansion pipeline
├── fasta_train_test_split.py    # Data splitting for neural networks
├── 3oxc_edited.pdb              # Template structure for delaunay_triangulation and rosetta_modeling
├── delaunay_triangulation.py    # Data formatting for KNN and RF
├── rosetta_modeling.py          # Feature generation
└── pos_rosetta_processing.py    # Data formatting for LR

```
