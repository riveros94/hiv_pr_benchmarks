#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified to read FASTA files instead of CSV
"""
import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial import Delaunay
from itertools import combinations_with_replacement

def read_fasta(file_path):
    """
    Reads a FASTA file and converts it to a pandas DataFrame.
    Treats the entire header as the sequence ID and doesn't 
    attempt to parse the header format.
    
    Parameters:
    - file_path (str): Path to the FASTA file.
    
    Returns:
    - DataFrame: Contains SeqID and Sequence columns.
    """
    sequences = {'SeqID': [], 'Sequence': []}
    caracteres_invalidos = {'X', '*', '.', '~'}
    
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id and not any(char in current_sequence for char in caracteres_invalidos):
                    sequences['SeqID'].append(current_id)
                    sequences['Sequence'].append(current_sequence)
                
                # Get the full header as ID (removing the '>' character)
                current_id = line[1:]
                current_sequence = ''
            else:
                current_sequence += line
        
        # Add the last sequence
        if current_id and not any(char in current_sequence for char in caracteres_invalidos):
            sequences['SeqID'].append(current_id)
            sequences['Sequence'].append(current_sequence)
    
    # Convert to DataFrame
    return pd.DataFrame(sequences)

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two three-dimensional points.

    Parameters:
    - point1 (array): Coordinates of the first point.
    - point2 (array): Coordinates of the second point.

    Returns:
    - float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_amino_acid_type(index, amino_acids):
    """
    Returns the amino acid type corresponding to an index in the amino acids list.

    Parameters:
    - index (int): The index of the amino acid in the list.
    - amino_acids (list): The list of amino acid types.

    Returns:
    - str: The amino acid type corresponding to the index.
    """
    return amino_acids[index]

def read_pdb_coordinates(file_path):
    """
    Reads the coordinates of alpha carbon atoms (Cα) and amino acid types from a PDB file.

    Parameters:
    - file_path (str): The path to the PDB file.

    Returns:
    - numpy.array: The coordinates of alpha carbon atoms (Cα).
    - list: The amino acid types present in the file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    coordinates = []
    amino_acids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords = atom.get_coord()
                    coordinates.append(coords)
                    amino_acids.append(residue.get_resname())

    return np.array(coordinates), amino_acids

def update_distance(df, amino_acid1, amino_acid2, distance):
    """
    Updates the distance between two amino acids in the dataframe.

    Parameters:
    - df (DataFrame): The dataframe containing distances between amino acids.
    - amino_acid1 (str): The type of the first amino acid.
    - amino_acid2 (str): The type of the second amino acid.
    - distance (float): The distance between the two amino acids.
    """
    for col in df.columns:
        if amino_acid1 + amino_acid2 == col or amino_acid2 + amino_acid1 == col:
            df[col] += distance

def construct_df(simplices, coordinates, aminoacids):
    """
    Constructs a dataframe containing distances between amino acids.

    Parameters:
    - simplices (array): The triangles in the triangulation.
    - coordinates (array): The coordinates of alpha carbon atoms (Cα).
    - aminoacids (list): The amino acid types present.

    Returns:
    - DataFrame: The dataframe containing distances between amino acids.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    combinations = list(combinations_with_replacement(amino_acids,2))
    df = pd.DataFrame(0, columns=[f'{a[0]}{a[1]}' for a in combinations], index=[0])

    for i, simplex in enumerate(simplices):
        index1, index2, index3, index4 = simplex

        amino_acid1 = get_amino_acid_type(index1, aminoacids)
        amino_acid2 = get_amino_acid_type(index2, aminoacids)
        amino_acid3 = get_amino_acid_type(index3, aminoacids)
        amino_acid4 = get_amino_acid_type(index4, aminoacids)

        distance1_2 = calculate_distance(coordinates[index1], coordinates[index2])
        update_distance(df, amino_acid1, amino_acid2, distance1_2)

        distance1_3 = calculate_distance(coordinates[index1], coordinates[index3])
        update_distance(df, amino_acid1, amino_acid3, distance1_3)

        distance1_4 = calculate_distance(coordinates[index1], coordinates[index4])
        update_distance(df, amino_acid1, amino_acid4, distance1_4)

        distance2_3 = calculate_distance(coordinates[index2], coordinates[index3])
        update_distance(df, amino_acid2, amino_acid3, distance2_3)

        distance2_4 = calculate_distance(coordinates[index2], coordinates[index4])
        update_distance(df, amino_acid2, amino_acid4, distance2_4)

        distance3_4 = calculate_distance(coordinates[index3], coordinates[index4])
        update_distance(df, amino_acid3, amino_acid4, distance3_4)

    return df

def process_structures(pdb_path, sequence_df):
    """
    Process PDB file and sequence dataframe.

    Args:
        pdb_path (str): Path to PDB file
        sequence_df (pd.DataFrame): DataFrame with sequences

    Returns:
        pd.DataFrame: Combined data
    """
    one_letter_code = {
        'PRO': 'P', 'GLN': 'Q', 'ILE': 'I', 'THR': 'T', 'LEU': 'L',
        'TRP': 'W', 'LYS': 'K', 'ARG': 'R', 'VAL': 'V', 'ALA': 'A',
        'ASP': 'D', 'GLY': 'G', 'GLU': 'E', 'MET': 'M', 'SER': 'S',
        'HIS': 'H', 'ASN': 'N', 'TYR': 'Y', 'PHE': 'F', 'CYS': 'C'
    }

    coordinates, amino_acids = read_pdb_coordinates(pdb_path)
    amino_acids = [one_letter_code.get(code, 'X') for code in amino_acids]  # Handle unknown codes

    triangulation = Delaunay(coordinates)
    simplices = triangulation.simplices
    sample = construct_df(simplices, coordinates, amino_acids)
    
    sequences = list(sequence_df['Sequence'])
    for seq in sequences:
        sequence_list = list(seq)
        sequence_list *= 2  # Duplicates the sequence
        sample = pd.concat([sample, construct_df(simplices, coordinates, sequence_list)], axis=0)

    sample = sample.iloc[1:]  # Remove the first row (template)
    sample.reset_index(drop=True, inplace=True)
    
    return sample

def make_prediction(model_pkl, data):
    """
    Make predictions using a pickled model.
    
    Parameters:
    - model_pkl (str): Path to the pickled model.
    - data (DataFrame or ndarray): Input data for prediction.
    
    Returns:
    - array: Predictions from the model.
    """
    with open(model_pkl, 'rb') as file:
        model = pickle.load(file)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    data = np.ascontiguousarray(data)

    prediction = model.predict(data)
    return prediction

def predict_hiv_resistance(model, fasta_path, drug, pdb_path, n_runs=100, seed=42, output_file="predictions.txt"):
    """
    Predict HIV resistance using a model, FASTA file, and PDB structure.
    
    Parameters:
    - model (str): Path to the pickled model.
    - fasta_path (str): Path to the FASTA file.
    - drug (str): Drug name.
    - pdb_path (str): Path to the PDB file.
    - n_runs (int): Number of runs for timing (default 100).
    - seed (int): Random seed.
    - output_file (str): Output file path.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    times = []
    for i in range(n_runs):
        start_time = time.time()
        dataframe = read_fasta(fasta_path)
        triang = process_structures(pdb_path, dataframe)
        prediction = make_prediction(model, triang)
        end_time = time.time()
        times.append(end_time - start_time)
    mean_time = np.mean(times)
    std_time = np.std(times)
    result = f"Prediction for drug {drug}: {prediction}\n"
    result += f"Number of runs: {n_runs}\n"
    result += f"Mean execution time: {mean_time:.4f} seconds\n"
    result += f"Standard deviation: {std_time:.4f} seconds\n"
    result += f"Seed used: {seed}\n"
    result += f"Individual run times: {', '.join([f'{t:.4f}' for t in times])}\n"
    result += "\nSequence Information:\n"
    for i, (seq_id, sequence) in enumerate(zip(dataframe['SeqID'], dataframe['Sequence'])):
        result += f"Sequence {i+1}: {seq_id}\n"
        result += f"Length: {len(sequence)}\n"
    
    print(result)  # Console output
    with open(output_file, 'a') as f:
        f.write(result + '\n')

# In-house test
fasta_path = 'seq_test.fasta'
pdb_path = '3oxc_edited.pdb'
predict_hiv_resistance(model='4NFV_fold_knn_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='inhouse_knn_prediction.txt')

# Steiner test
predict_hiv_resistance(model='steiner_3NFV_fold_knn_model.pkl',
                          fasta_path=fasta_path, 
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='steiner_knn_prediction.txt')

# Shen test
predict_hiv_resistance(model='3NFV_fold_knn_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          pdb_path=pdb_path, 
                          output_file='shen_knn_prediction.txt')