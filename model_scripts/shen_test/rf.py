
import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


def load_best_model_and_evaluate(results, test_data, drug):
  
    # Find the index of the best performing model
    best_model_idx = results['Test_AUC'].idxmax()
    
    # Calculate the model number (adding 1 since indices start at 0)
    model_number = str(best_model_idx + 1) + drug + '_fold_rf_model.pkl'
    
    # Verify if the model file exists
    if not os.path.exists(model_number):
        raise FileNotFoundError(f"Model file {model_number} not found")
    
    # Prepare test data
    Y2 = test_data[drug].values
    X2 = test_data.drop(test_data.columns[0:16], axis=1)
    
    # Load the best model
    try:
        with open(model_number, 'rb') as file:
            loaded_model = pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    # Make predictions
    y_test_predicted = loaded_model.predict(X2)
    y_test_proba = loaded_model.predict_proba(X2)[:, 1]
    
    # Calculate evaluation metrics
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(Y2, y_test_predicted).ravel()
    
    # Calculate all metrics
    metrics = {
        'Accuracy_Test': accuracy_score(Y2, y_test_predicted),
        'Precision_Test': precision_score(Y2, y_test_predicted),
        'Specificity_Test': tn_test / (tn_test + fp_test),
        'Recall_Test': recall_score(Y2, y_test_predicted),
        'F1_test': f1_score(Y2, y_test_predicted),
        'AUC': roc_auc_score(Y2, y_test_proba)
    }
    
    # Create results DataFrame
    results_df = pd.DataFrame([metrics])
    
    # Add model information
    results_df['Best_Model_File'] = model_number
    results_df['Original_Index'] = best_model_idx

    return results_df

def train_rf(drug, df):
    start_time = time.time()  # Inicia a contagem do tempo  

    data_drug = df[df[drug].notna()]
    
    drug_name = drug + '_cv.csv'
    # Split data into training (80%) and testing (20%) sets
    train_df, test_df = train_test_split(data_drug, test_size=0.2, random_state=42)
   
    train_df.to_csv(f'{drug}_train.csv', index=False)
    test_df.to_csv(f'{drug}_test.csv', index=False)
    
    model = RandomForestClassifier(n_estimators=10)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    X1 = train_df.drop(data_drug.columns[0:16], axis=1)
    Y1 = train_df[drug].values
    
    results = pd.DataFrame()

    for train_index, test_index in skf.split(X1, Y1):
        x_train_fold, x_test_fold = X1.iloc[train_index], X1.iloc[test_index]
        y_train_fold, y_test_fold = Y1[train_index], Y1[test_index]
        model.fit(x_train_fold, y_train_fold)
        
        y_train_predicted = model.predict(x_train_fold).tolist()
        y_test_predicted = model.predict(x_test_fold).tolist()
        
        # Calculate metrics
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_fold, y_train_predicted).ravel()
        Train_specificity = tn_train / (tn_train + fp_train)
        Train_accuracy = accuracy_score(y_train_fold, y_train_predicted)
        Train_precision = precision_score(y_train_fold, y_train_predicted)
        Train_recall = recall_score(y_train_fold, y_train_predicted)
        Train_f1 = f1_score(y_train_fold, y_train_predicted)
        
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test_fold, y_test_predicted).ravel()
        Test_specificity = tn_test / (tn_test + fp_test)
        Test_accuracy = accuracy_score(y_test_fold, y_test_predicted)
        Test_precision = precision_score(y_test_fold, y_test_predicted)
        Test_recall = recall_score(y_test_fold, y_test_predicted)
        Test_f1 = f1_score(y_test_fold, y_test_predicted)
        test_auc = roc_auc_score(y_test_fold, model.predict_proba(x_test_fold)[:,1])

        # Append results to results_list
        result_row = {'Accuracy_Train': Train_accuracy,'Accuracy_Test': Test_accuracy,
                'Precision_Train': Train_precision, 'Precision_Test': Test_precision,
                'Recall_Train': Train_recall, 'Recall_Test': Test_recall,
                'F1_Train': Train_f1, 'F1_test': Test_f1, 
                'Test_AUC': test_auc}
        results = pd.concat([results, pd.DataFrame(result_row, index=[0])], ignore_index=True)

       
        model_number = str(len(results.index)) + drug + '_fold_rf_model.pkl'

        # Save the best model to a file using pickle
        with open(model_number, 'wb') as file:
            pickle.dump(model, file)

    # Concatenate all DataFrames in results_list

    results.to_csv(f'fold_{str(len(results.index))}{drug_name}.csv')

    test_results = load_best_model_and_evaluate(results, test_df, drug)
    test_results.to_csv('test_performance_values.csv')

    end_time = time.time()  # Termina a contagem do tempo
    print(f'Tempo decorrido: {end_time - start_time:.2f} segundos')

    return results, test_results

# Data geral
data = pd.read_csv('hiv_pr_benchmarks/data/shen_data/knn_rf/shen_triang_data.csv', index_col=False)
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# FPV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/fpv')
cv_r, test_r = train_rf('FPV', data)

# NFV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/nfv')
cv_r, test_r = train_rf('NFV', data)

# IDV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/idv')
cv_r, test_r = train_rf('IDV', data)

# ATV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/atv')
cv_r, test_r = train_rf('ATV', data)

# LPV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/lpv')
cv_r, test_r = train_rf('LPV', data)

# SQV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/sqv')
cv_r, test_r = train_rf('SQV', data)

# TPV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/tpv')
cv_r, test_r = train_rf('TPV', data)

# DRV
os.chdir('hiv_pr_benchmarks/data/shen_data/knn_rf/drv')
cv_r, test_r = train_rf('DRV', data)

