import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def process_fasta(input_fasta, descriptor):
    atv = fasta_to_list(input_fasta)
    headers = fasta_headers_to_dataframe(input_fasta)
    atv_descriptors = get_descriptors(atv, descriptor=descriptor)
    df = pd.concat([headers, atv_descriptors], axis=1, ignore_index=False)
    df.dropna(inplace=True)  # Delete rows with NA
    return df

def fasta_to_list(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    sequences = []
    current_sequence = ""
    
    for line in lines:
        if line.startswith('>'):
            if current_sequence:
                sequences = sequences + [current_sequence]
                current_sequence = ""
            continue
        current_sequence += line.strip()
    
    # Add the last sequence
    if current_sequence:
        sequences = sequences + [current_sequence]
    
    return sequences

def fasta_headers_to_dataframe(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    data = {'SeqID': [], 'Class': []}
    current_id = ""
    current_label = ""
    
    for line in lines:
        if line.startswith('>'):
            # Extract ID and Label from the header
            header_parts = line[1:].strip().split('_')
            current_id = header_parts[0]
            current_label = header_parts[1] if len(header_parts) > 1 else ""
            
            # Add the extracted information to the data dictionary using concatenation
            data['SeqID'] = data['SeqID'] + [current_id]
            data['Class'] = data['Class'] + [current_label]
    
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)
    
    return df

def assign_descriptor(sequence, descriptor='vhse'):
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
                result = result + desc_tbl[j][1:]
                break
    
    return result

def get_descriptors(data_list, descriptor='vhse'): ## Receives a .csv with only protein sequences
    calculated_descriptors = []
    for i in range(len(data_list)):
        seq = data_list[i]
        list_descriptors = assign_descriptor(seq, descriptor)
        calculated_descriptors = calculated_descriptors + [list_descriptors]
    calculated_descriptors = pd.DataFrame(calculated_descriptors)
    # Rename columns from 0,1,2,... to V1,V2,V3,...
    new_columns = [f'V{i+1}' for i in range(calculated_descriptors.shape[1])]
    calculated_descriptors.columns = new_columns

    return calculated_descriptors

def train_logistic_regression_with_feature_selection(df, drug):
    
    drug_name = drug + '_cv.csv'
    # Split data into training (80%) and testing (20%) sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df.to_csv(f'{drug}_train.csv', index=False)
    test_df.to_csv(f'{drug}_test.csv', index=False)
    
    X1 = train_df.drop(train_df.columns[0:2], axis=1)
    Y1 = train_df.iloc[:, [1]].values
    Y1 = Y1.astype(int)
    
    X2 = test_df.drop(test_df.columns[0:2], axis=1)
    Y2 = test_df.iloc[:, [1]].values
    Y2 = Y2.astype(int)

    # Feature selection using mutual information
    mi_scores = mutual_info_classif(X1, Y1, random_state=42)
    mi_scores = mi_scores.reshape(495, 1)
    colnames_t1 = X1.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores'])
    sorted_mi_values = mi_values.sort_values(by='MI_Scores', ascending=False)
    desired_order = sorted_mi_values.index
    
    X1 = X1.loc[:, desired_order]
    X2 = X2.loc[:, desired_order]
    
    # Compute class weights
    zero = np.sum(Y1 == 0)
    one = np.sum(Y1 == 1)
    weight_0 = 1.0
    weight_1 = zero / one
    Y1_weight = train_df.iloc[:, 1]
    sample_weight = Y1_weight.apply(lambda x: weight_1 if x == 1 else weight_0)
    sample_weight = sample_weight.values
    
    grid = {"C": np.array([0.01, 0.1, 1, 10, 100])}
    
    # Initialize variables
    selected_predictors = pd.DataFrame()
    selected_test = pd.DataFrame()
    results = pd.DataFrame()
    predictors = list(X1.columns)
    
    # Loop through predictors - MODELO COM PESO
    for predictor in predictors:
        # Add the current predictor to the selected list
        selected_predictors.loc[:, predictor] = X1[predictor]
        selected_test.loc[:, predictor] = X2[predictor]
        # Fit a logistic regression model with cross-validation
        model = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', class_weight={0: weight_0, 1: weight_1})
        model_cv = GridSearchCV(model, grid, cv=5)
        model_cv.fit(selected_predictors, Y1, sample_weight=sample_weight)
        best_model = model_cv.best_estimator_
        Y1_predicted = best_model.predict(selected_predictors).tolist()
        Y2_predicted = best_model.predict(selected_test).tolist()
        Training_Accuracy = accuracy_score(Y1, Y1_predicted)
        Testing_Accuracy = accuracy_score(Y2, Y2_predicted)
        Training_precision = precision_score(Y1, Y1_predicted)
        Testing_precision = precision_score(Y2, Y2_predicted)
        train_recall = recall_score(Y1, Y1_predicted)
        test_recall = recall_score(Y2, Y2_predicted)
        train_f1 = f1_score(Y1, Y1_predicted)
        test_f1 = f1_score(Y2, Y2_predicted)
        test_auc = roc_auc_score(Y2, best_model.predict_proba(selected_test)[:,1])
        c = model_cv.best_params_
        
        result_row = {'C': c['C'], "Number_features": len(selected_predictors),
                'Predictors': ', '.join([str(p) for p in selected_predictors]),
                'Accuracy_Train': Training_Accuracy,
                'Accuracy_Test': Testing_Accuracy,
                'Precision_Train': Training_precision, 'Precision_Test': Testing_precision,
                'Recall_Train': train_recall, 'Recall_Test': test_recall,
                'F1_Train': train_f1, 'F1_test': test_f1, 
                'Test_AUC': test_auc}
        results = pd.concat([results, pd.DataFrame(result_row, index=[0])], ignore_index=True)    

        model_number = str(len(results.index)) + '_best_logistic_regression_model.pkl'
        # Save the best model to a file using pickle
        with open(model_number, 'wb') as file:
            pickle.dump(best_model, file)
    
    results.to_csv(drug_name)
    
    return results

os.chdir('hiv_pr_benchmarks/data/steiner_data/')

# NFV
df = process_fasta(input_fasta = 'nfv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'nfv')


# ATV
df = process_fasta(input_fasta = 'atv.fasta', descriptor='zscales')
cv_atv = train_logistic_regression_with_feature_selection(df, 'atv')

# FPV
df = process_fasta(input_fasta = 'fpv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'fpv')

# # IDV
df = process_fasta(input_fasta = 'idv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'idv')

# DRV
df = process_fasta(input_fasta = 'drv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'drv')

# TPV
df = process_fasta(input_fasta = 'tpv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'tpv')

# LPV
df = process_fasta(input_fasta = 'lpv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'lpv')

# SQV
df = process_fasta(input_fasta = 'sqv.fasta', descriptor='zscales')
cv_data = train_logistic_regression_with_feature_selection(df, 'sqv')
