# HIV-1 Protease Inhibitor Resistance Dataset Processing
## In-house processing
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
## Processing Pipeline
### 1. Initial Data Filtering
```R
Rscript data_processing.R
```
Filters and processes raw sequence data:
- Selects sequences with exactly 99 amino acids
- Removes sequences with:
  - Insertions/deletions
  - Stop codons
  - Multiple mutations at same sites
- Eliminates redundant sequences while preserving median resistance values
### 2. Sequence Clustering & Test Set Construction
```
cd clustering
python scores_to_csv.py
python tsne_seq.py
```
Implements clustering-based sequence analysis:
- Generates pairwise alignments (BLOSUM80 matrix)
- Performs t-SNE dimensionality reduction
- Conducts K-means clustering (k=5, perplexity=50)
- Creates test set with balanced cluster representation
### 3. Feature Generation
#### a. Delaunay Triangulaion
```python
python delaunay_triangulation.py
```
Generates geometric features:
- Uses 3oxc crystal structure template
- Implements Delaunay triangulation
- Creates distance-based feature vectors
#### b. Rosetta Energy Features
```python
python rosetta_modeling.py
python pos_rosetta_processing.py
```
Calculates structural energetics:
- Template: 3oxc structure (cleaned)
- Three rounds FastRelax minimization
- Extracts per-residue energy terms
- Processes features for machine learning
### 4. Model-Specific Data Formatting
```python
python split_nn_models.py
```
Prepares data for different model architectures:
- Neural Networks (MLP, BRNN, CNN)
## Directory Structure
```
.
├── 3oxc_edited.pdb           # Template structure
├── clustering/               # Clustering analysis
├── knn_rf/                  # KNN/RF features
├── nn_models/               # Neural network data
├── rosetta_lr/             # Rosetta features
└── *.py/*.R                # Processing scripts
```
## References
- Rhee et al. (2003) - Stanford HIV Drug Resistance Database
- Zhang et al. (2005) - PhenoSense assay methodology
- Henikoff & Henikoff (1992) - BLOSUM matrices
- van der Maaten & Hinton (2008) - t-SNE
## Notes
- Maintains data quality through strict filtering
- Ensures diverse sequence representation in test sets
- Generates complementary feature types for analysis
