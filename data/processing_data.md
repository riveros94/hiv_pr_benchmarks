# HIV-1 Protease Inhibitor Resistance Dataset Processing

This repository contains processing pipelines for three different approaches to constructing HIV-1 protease inhibitor resistance datasets for machine learning benchmarking.

## Repository Structure

```
.
├── inhouse_data/         # In-house clustering-based approach
├── shen_data/            # Shen et al. (2016) sequence expansion approach 
└── steiner_data/         # Steiner et al. (2020) filtered approach
```

## Dataset Approaches

### In-house Dataset

Our approach implements strict filtering and clustering-based test set construction:

- Filters sequences to include only those with exactly 99 amino acids
- Removes sequences with insertions, deletions, stop codons, or multiple mutations
- Eliminates redundant sequences while preserving median resistance values
- Uses t-SNE and K-means clustering (k=5) to ensure diverse sequence representation in test sets

Key scripts:
- `data_processing.R`: Initial filtering
- `clustering/scores_to_csv.py`: Prepares data for clustering
- `clustering/tsne_seq.py`: Implements t-SNE and K-means for the test set construction

### Shen Dataset 

This approach follows Shen et al. (2016) methodology:

- Expands sequences with ambiguous amino acids into multiple individual sequences
- Maintains identical resistance values for all expanded sequences
- Excludes sequences with insertions, deletions, or stop codons
- Uses random 80:20 train/test split

Key scripts:
- `run_sequence_expansion.sh`: Sequence expansion pipeline
- `fasta_train_test_split.py`: Data splitting for neural networks
- `delaunay_triangulation.py`: Structural feature extraction

### Steiner Dataset

Based on Steiner et al. (2020) approach:

- Filters out sequences with ambiguities at major drug resistance mutation positions
- Converts each dataset into format suitable for neural network training
- Uses 80:20 random split for training/test sets


## Common Data Processing

All three approaches process data from the Stanford HIV Drug Resistance Database for eight protease inhibitors:
- Darunavir (DRV)
- Fosamprenavir (FPV)
- Atazanavir (ATV)
- Indinavir (IDV)
- Lopinavir (LPV)
- Nelfinavir (NFV)
- Saquinavir (SQV)
- Tipranavir (TPV)

Sequences are classified as resistant (1) or susceptible (0) based on established clinical cutoffs of fold change in IC50:
- TPV: 2.0
- NFV/SQV/IDV/ATV: 3.0
- FPV: 4.0
- LPV: 9.0
- DRV: 10.0

## Feature Generation

Each approach includes scripts for generating different feature representations:

1. **Delaunay Triangulation Features**: For KNN and RF models
2. **Rosetta Energy Terms**: For Rosetta LR models
3. **Integer-encoded Sequences**: For neural network models
4. **zScales Descriptors**: For zScales LR models

## Usage

Each directory contains detailed instructions on how to process the data according to the specific approach. Follow the README file in each directory for step-by-step guidance.
