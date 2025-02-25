import os
import time
import pickle
import random
import numpy as np
import pandas as pd


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


def assign_descriptor (sequence, descriptor = 'vhse'):
    vhse_tbl = [['A', 0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48], 
                     ['R', -1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83], 
                     ['N', -0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80], 
                     ['D', -1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56], 
                     ['C', 0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19], 
                     ['Q', -0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41], 
                     ['E', -1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.02], 
                     ['G', -0.20, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
                     ['H', -0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65], 
                     ['I', 1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13], 
                     ['L', 1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62], 
                     ['K', -1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13], 
                     ['M', 1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68], 
                     ['F', 1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.20], 
                     ['P', 0.22, -0.17, -0.50, 0.05, -0.01, -1.34, -0.19, 3.56],
                     ['S', -0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
                     ['T', -0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
                     ['W', 1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
                     ['Y', 0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52], 
                     ['V', 0.76, -0.92, -0.17, -1.91, 0.22, -1.40, -0.24, -0.03]]
    
    zscales_tbl = [['A', 0.24, -2.32, 0.60, -0.14, 1.30],
                   ['R', 3.52, 2.50, -3.50, 1.99, -0.17],
                   ['N', 3.05, 1.62, 1.04, -1.15, 1.61],
                   ['D', 3.98, 0.93, 1.93, -2.46, 0.75], 
                   ['C', 0.84, -1.67, 3.71, 0.18, -2.65],
                   ['Q', 1.75, 0.50, -1.44, -1.34, 0.66],
                   ['E', 3.11, 0.26, -0.11, -3.04, -0.25],
                   ['G', 2.05, -4.06, 0.36, -0.82, -0.38],
                   ['H', 2.47, 1.95, 0.26, 3.90, 0.09],
                   ['I', -3.89, -1.73, -1.71, -0.84, 0.26],
                   ['L', -4.28, -1.30, -1.49, -0.72, 0.84],
                   ['K', 2.29, 0.89, -2.49, 1.49, 0.31], 
                   ['M', -2.85, -0.22, 0.47, 1.94, -0.98], 
                   ['F', -4.22, 1.94, 1.06, 0.54, -0.62],
                   ['P', -1.66, 0.27, 1.84, 0.70, 2.00],
                   ['S', 2.39, -1.07, 1.15, -1.39, 0.67],
                   ['T', 0.75, -2.18, -1.12, -1.46, -0.40],
                   ['W', -4.36, 3.94, 0.59, 3.44, -1.59],
                   ['Y', -2.54, 2.44, 0.43, 0.04, -1.47],
                   ['V', -2.59, -2.64, -1.54, -0.85, -0.02]]
    
    seq = [x for x in sequence]
    result = []
    desc_tbl = zscales_tbl if descriptor == 'zscales' else vhse_tbl
   
    for i in seq:
        for j in range(len(desc_tbl)):
            if i == desc_tbl[j][0]:
                for q in desc_tbl[j][1:]:
                    result.append(q)
    return(result)

def get_descriptors(data_list, descriptor = 'vhse'): ## Receives a .csv with only protein sequences
    calculated_descriptors = []
    for i in range(len(data_list)):
        seq = data_list[i]
        list_descriptors = assign_descriptor(seq, descriptor)
        calculated_descriptors.append(list_descriptors)
    calculated_descriptors = pd.DataFrame(calculated_descriptors)
    return(calculated_descriptors)

def process_data(df, zscales_data):
    p1 = df.iloc[:, 1:17]
    sum_df = pd.concat([p1.reset_index(drop=True), 
                        zscales_data.reset_index(drop=True)], axis=1)
    return sum_df
    
def make_prediction(model_pkl, data):
    # Load the model
    with open(model_pkl, 'rb') as file:
        model = pickle.load(file)
    
    # Get the feature names expected by the model
    feature_names = model.feature_names_in_
    
    # Create a DataFrame with all required features, filled with zeros
    required_features = pd.DataFrame(0, index=data.index, columns=feature_names)
    
    # Update the features we have in our data
    for col in data.columns:
        if col in feature_names:
            required_features[col] = data[col]
    
    # Select the first row for prediction
    X_input = required_features.iloc[0:1]
    
    # Make the prediction
    prediction = model.predict(X_input)
    
    return prediction
    

def predict_hiv_resistance(model, fasta_path, drug, n_runs=100, seed=42, output_file="predictions.txt"):
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
        pi_list = list(dataframe['Sequence'])
        pi_zscales = get_descriptors(pi_list, descriptor='zscales')
        pi_zscales.columns = [str(i) for i in range(1, len(pi_zscales.columns) + 1)]
        data = process_data(pi_zscales, dataframe)
        prediction = make_prediction(model, data)
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


os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/time_test/test_python_2/zscales')
# In-house test
fasta_path = 'seq_test.fasta'
predict_hiv_resistance(model='140_best_logistic_regression_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          output_file='inhouse_zscales_prediction.txt')

# Steiner test
predict_hiv_resistance(model='168_best_logistic_regression_model.pkl',
                          fasta_path=fasta_path, 
                          drug='NFV', 
                          output_file='steiner_zscales_prediction.txt')

# Shen test
predict_hiv_resistance(model='410_best_logistic_regression_model.pkl', 
                          fasta_path=fasta_path,
                          drug='NFV', 
                          output_file='shen_zscales_prediction.txt')



