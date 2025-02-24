import os
import numpy as np
import pandas as pd

def split_df(df):
    # Converts the DataFrame into a NumPy array
    values_np = df.to_numpy()
    
    # Computes the number of columns
    num_columns = values_np.shape[1]
    
    # Divides the number of columns by 2 to find the split point
    split_point = num_columns // 2
    
    # Splits the array into two matrices
    matrix1 = values_np[:, :split_point]
    matrix2 = values_np[:, split_point:]
    
    return matrix1, matrix2

def mean_matrices(matrix1, matrix2):
    # Checks if the matrices have the same dimensions
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices have different dimensions and cannot be added.")
    
    # Computes the mean of the elements of the matrices
    result = (matrix1 + matrix2) / 2
    
    return result

def process_dataframe(df):
    values = df.drop(['Protein', 'Sequence'], axis=1)
    values = values.drop(['dG_Fold'], axis=1)
    m1, m2 = split_df(values)
    mean = mean_matrices(m1, m2)
    mean = pd.DataFrame(mean, columns=values.columns[:mean.shape[1]])
    p1 = df[['Protein', 'Sequence']]
    p3 = df[['dG_Fold']]
    sum_df = pd.concat([p1, mean, p3], axis=1)
    return sum_df


os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/inhouse_data/rosetta_lr')
df = pd.read_csv('output.csv')
data = process_dataframe(df)
data.to_csv('pi_monomer_rosetta.csv', index=False)


data1 = data.iloc[1:].reset_index(drop=True)

data2 = pd.read_csv('../pi_sequence_cluster_classification.csv')
data2 = data2.iloc[1:].reset_index(drop=True)

drugs = data2.loc[:, ['ATV', 'FPV', 'SQV', 'NFV', 'IDV', 'LPV', 'DRV', 'TPV']]
SeqID = data2.loc[:, ['SeqID']]            
rosetta_descriptors = data1.drop(['Protein', 'Sequence'], axis=1)

dataset = pd.concat([SeqID, drugs, rosetta_descriptors], axis=1)

dataset.to_csv('inhouse_rosetta_modeled_classification.csv', index=False) # Data for LR models