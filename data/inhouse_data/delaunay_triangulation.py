import os
import time
import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial import Delaunay
from itertools import combinations_with_replacement

def calculate_distance(point1, point2):
    """Calculate euclidean distance between two 3D points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_amino_acid_type(index, amino_acids):
    """Get amino acid type from index"""
    return amino_acids[index]

def read_pdb_coordinates(file_path):
    """Read alpha carbon (Cα) coordinates and amino acid types from PDB file"""
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
    """Update distance between amino acids in dataframe"""
    for col in df.columns:
        if amino_acid1 + amino_acid2 == col or amino_acid2 + amino_acid1 == col:
            df[col] += distance

def construct_df(simplices, coordinates, aminoacids):
    """Build dataframe with distances between amino acids"""
    aminoacidos = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    combinacoes = list(combinations_with_replacement(aminoacidos,2))
    df = pd.DataFrame(0, columns=[f'{a[0]}{a[1]}' for a in combinacoes], index=[0])

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
        'HIS': 'H', 'ASN': 'N', 'TYR': 'Y', 'PHE': 'F'
    }
    
    coordinates, amino_acids = read_pdb_coordinates(pdb_path)
    amino_acids = [one_letter_code[code] for code in amino_acids]
    
    triangulation = Delaunay(coordinates)
    simplices = triangulation.simplices
    sample = construct_df(simplices, coordinates, amino_acids)
    sequences = list(sequence_df['Sequence'])   
    for seq in sequences:
        sequences = [list(sequence) for sequence in seq]
        sequences *= 2  # Duplicates the sequences
        unique_sequence = [amino for amino_acid_list in sequences for amino in amino_acid_list]
        sample = pd.concat([sample, construct_df(simplices, coordinates, unique_sequence)], axis=0)

    sample = sample.iloc[1:]
    sample.reset_index(drop=True, inplace=True)
    sequence_df.reset_index(drop=True, inplace=True)
    combined_df = pd.concat([sequence_df.drop(columns=['Sequence']), sample], axis=1)
    return combined_df

os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/inhouse_data')
pdb_file_path = "../3oxc_edited.pdb"
sequence_df = pd.read_csv('pi_sequence_cluster_classification.csv')

triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('knn_rf/in_house_triang.csv', index= False)
