
import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial import Delaunay
from itertools import combinations_with_replacement


# Function to calculate the distance between two three-dimensional points
def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two three-dimensional points.

    Parameters:
    - point1 (array): The coordinates of the first point.
    - point2 (array): The coordinates of the second point.

    Returns:
    - float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to get the amino acid type corresponding to an index
def get_amino_acid_type(index, amino_acids):
    """
    Returns the amino acid type corresponding to an index in the amino acid list.

    Parameters:
    - index (int): The index of the amino acid in the list.
    - amino_acids (list): The list of amino acid types.

    Returns:
    - str: The amino acid type corresponding to the index.
    """
    return amino_acids[index]

# Function to read the coordinates and amino acid types from a PDB file
def read_pdb_coordinates(file_path):
    """
    Reads the coordinates of alpha carbon (Cα) atoms and the amino acid types from a PDB file.

    Parameters:
    - file_path (str): The path to the PDB file.

    Returns:
    - numpy.array: The coordinates of alpha carbon (Cα) atoms.
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

# Function to update the distance between amino acids in the dataframe
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

# Function to construct the dataframe with distances between amino acids
def construct_df(simplices, coordinates, aminoacids):
    """
    Constructs a dataframe containing the distances between amino acids.

    Parameters:
    - simplices (array): The triangles in the triangulation.
    - coordinates (array): The coordinates of alpha carbon (Cα) atoms.
    - aminoacids (list): The types of amino acids present.

    Returns:
    - DataFrame: The dataframe containing the distances between amino acids.
    """
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    combinations_list = list(combinations_with_replacement(amino_acids, 2))
    df = pd.DataFrame(0, columns=[f'{a[0]}{a[1]}' for a in combinations_list], index=[0])

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


# ATV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/atv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/atv_steiner_triang.csv', index= False)

# DRV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/drv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/drv_steiner_triang.csv', index= False)

# FPV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/fpv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/fpv_steiner_triang.csv', index= False)

# IDV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/idv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/idv_steiner_triang.csv', index= False)

# LPV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/lpv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/lpv_steiner_triang.csv', index= False)

# NFV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/nfv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/nfv_steiner_triang.csv', index= False)

# SQV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/sqv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/sqv_steiner_triang.csv', index= False)

# TPV
pdb_file_path = "hiv_benchmarks/data/steiner_data/3oxc_edited.pdb"
sequence_df = pd.read_csv('hiv_benchmarks/data/steiner_data/tpv_filtrated.csv')
triangulation = process_structures(pdb_file_path, sequence_df)
triangulation.to_csv('hiv_benchmarks/data/steiner_data/knn_rf/tpv_steiner_triang.csv', index= False)
