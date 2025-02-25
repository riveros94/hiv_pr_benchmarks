# HIV-1 Protease Inhibitor Resistance Prediction

This repository contains code for feature analysis of Rosetta and zScales Logistic Regression model


## Repository Structure

- `correlations.py`: Script for calculating and visualizing correlation between feature importance metrics across different dataset preparations
- `modify_bfactor_mi.R`: R script for modifying B-factor values in PDB files based on mutual information values for structural visualization
- `rosetta_mutual_information_calculation.py`: Script for calculating mutual information between Rosetta energy terms and resistance phenotype
- `zscales_mutual_information_calculation.py`: Script for calculating mutual information between zScales descriptors and resistance phenotype

## Usage

### Mutual Information Calculation

To calculate mutual information for zScales descriptors:
```bash
python zscales_mutual_information_calculation.py
```

To calculate mutual information for Rosetta energy terms:
```bash
python rosetta_mutual_information_calculation.py
```

### Correlation Analysis

To analyze correlation between feature importance metrics across datasets:
```bash
python correlations.py
```

### B-factor Modification for Visualization

To modify B-factors in PDB files for visualization:
```bash
Rscript modify_bfactor_mi.R
```

## Dependencies

- Python 3.10+
- R 3.6+
- Libraries:
  - scikit-learn 1.5.2
  - NumPy 1.24.4
  - Matplotlib 3.9
  - pyRosetta 4.2
  - pandas
  - BioPython

