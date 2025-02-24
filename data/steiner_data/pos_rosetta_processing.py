import pandas as pd
import os

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

def create_rosetta_dataset(rosetta_file, df, drug, save_paths):
    """
    Process Rosetta output data and combine with ATV data to create classification dataset.
    
    Parameters:
    -----------
    rosetta_file : str
        Path to the initial output CSV file
    df          : str
        Path to the filtrated CSV file
    drug        : str
        processed drug
    save_paths : dict
        Dictionary containing paths for saving intermediate and final results
        Keys: 'pi_monomer', 'classification'
    
    Returns:
    --------
    pandas.DataFrame
        Processed dataset combining Rosetta descriptors with classification data
    """
    # Read and clean initial output
    rosetta = pd.read_csv(rosetta_file)
    
    # Process the dataframe (assuming process_dataframe function exists)
    data = process_dataframe(rosetta)
    
    # Save intermediate processed data
    data.to_csv(save_paths['pi_monomer'], index=False)
    
    # Extract Rosetta descriptors (skip first line which is pdb template)
    data1 = data.tail(-1)
    
    
    # Get classification data
    data2 = pd.read_csv(df)
    data2 = data2.head(1)
    
    # Extract features
    drugs = data2.loc[:, ['Class']]
    SeqID = data2.loc[:, ['SeqID']]
    rosetta_descriptors = data1.drop(['Protein', 'Sequence'], axis=1)
    
    # Reset indices to ensure alignment
    SeqID = SeqID.reset_index(drop=True)
    drugs = drugs.reset_index(drop=True)
    rosetta_descriptors = rosetta_descriptors.reset_index(drop=True)
    
    # Combine all features
    dataset = pd.concat([SeqID, drugs, rosetta_descriptors], axis=1)
    
    # Save final classification dataset
    dataset.to_csv(save_paths['classification'], index=False)
    
    return dataset


os.chdir('hiv_benchmarks/data/steiner_data/rosetta_lr/atv')

# ATV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/atv/atv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/atv/atv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../atv_filtrated.csv', drug = 'atv', save_paths=save_paths)

# DRV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/drv/drv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/drv/drv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../drv_filtrated.csv', drug = 'drv', save_paths=save_paths)

# FPV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/fpv/fpv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/fpv/fpv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../fpv_filtrated.csv', drug = 'fpv', save_paths=save_paths)

# IDV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/idv/idv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/idv/idv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../idv_filtrated.csv', drug = 'idv', save_paths=save_paths)

# LPV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/lpv/lpv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/lpv/lpv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../lpv_filtrated.csv', drug = 'lpv', save_paths=save_paths)

# NFV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/nfv/nfv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/nfv/nfv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../nfv_filtrated.csv', drug = 'nfv', save_paths=save_paths)

# SQV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/sqv/sqv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/sqv/sqv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../sqv_filtrated.csv', drug = 'sqv', save_paths=save_paths)

# TPV
save_paths = {
    'pi_monomer': 'hiv_benchmarks/data/steiner_data/rosetta_lr/tpv/tpv_monomer_rosetta.csv',
    'classification': 'hiv_benchmarks/data/steiner_data/rosetta_lr/tpv/tpv_pi_rosetta.csv'
}
result = create_rosetta_dataset(rosetta_file='output.csv', df='../../tpv_filtrated.csv', drug = 'tpv', save_paths=save_paths)
