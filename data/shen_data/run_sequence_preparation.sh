#!/bin/bash

# 1. Run the initial R processing script
echo "Running process_hiv_ambiguous_sequences.R..."
Rscript process_hiv_ambiguous_sequences.R

# 2. Run the Python script to generate ambiguous sequences
echo "Running gerate_all_ambiguos_sequences.py..."
python gerate_all_ambiguos_sequences.py

# 3. Run the final R processing script
echo "Running final_data_processing.R..."
Rscript final_data_processing.R

echo "Pipeline completed!"

