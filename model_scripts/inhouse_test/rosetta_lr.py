
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


def data_normalization(dataset, norm_type):
    if norm_type == "normalize":

        normalized_dataset = dataset

        scaler = MinMaxScaler()
        scaler.fit(normalized_dataset)

        normalized_dataset = scaler.transform(normalized_dataset)
        return normalized_dataset, scaler

    if norm_type == "standard":

        standardized_dataset = dataset

        scaler = StandardScaler()
        scaler.fit(standardized_dataset)

        standardized_dataset = scaler.transform(standardized_dataset)
        return standardized_dataset, scaler

def split_dataset(df, cluster_column, test_ratio=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    total_samples = len(df)
    test_samples = int(total_samples * test_ratio)

    clusters = df[cluster_column].unique()
    test_samples_per_cluster = test_samples // len(clusters)
    
    test_df = pd.DataFrame(columns=df.columns)
    
    for cluster in clusters:
        cluster_samples = df[df[cluster_column] == cluster]
        test_samples_cluster = cluster_samples.sample(test_samples_per_cluster, random_state=seed)
        test_df = pd.concat([test_df, test_samples_cluster], axis=0)
    
    train_df = df.drop(test_df.index)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_df, test_df

def remove_high_correlate(x_train, x_test, y_train, cutoff):

    mi_scores = mutual_info_classif(x_train, y_train)

    correlation_matrix = x_train.corr().abs()

    # Find pairs of highly correlated features
    high_corr_pairs = np.where((correlation_matrix > cutoff) | (correlation_matrix < -cutoff))

    columns_to_drop = []
    # Drop features based on mutual information
    for feature1, feature2 in zip(*high_corr_pairs):
        if feature1 != feature2:
            mi1 = mi_scores[feature1]
            mi2 = mi_scores[feature2]
            if mi1 < mi2:
                # Drop feature1
                columns_to_drop.append(feature1)
            else:
                # Drop feature2
                columns_to_drop.append(feature2)

    unique_columns_to_drop = set(columns_to_drop)
    unique_columns_to_drop_list = list(set(unique_columns_to_drop))

    x_train.drop(x_train.columns[unique_columns_to_drop_list], axis=1, inplace=True)
    x_test.drop(x_test.columns[unique_columns_to_drop_list], axis=1, inplace=True)


    return x_train, x_test

def LogRegClass(df, drug, out, feature_selection = False, c_mi=0.8):
    logging.basicConfig(filename='f{drug}_process.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    start_time = time.time()  # Inicia a count do tempo
    drug_name = drug + '_cv.csv'
    logging.info(f"Starting training for drug: {drug}")
    data_drug = df[df[drug].notna()]
    del df
    
    # Split data into training (80%) and testing (20%) sets
    train_df, test_df = split_dataset(data_drug, 'cluster', seed=13)   
    train_df.to_csv(f'{drug}train.csv', index=False)
    test_df.to_csv(f'{drug}test.csv', index=False)

    x_train = train_df.drop(["SeqID", "FPV", "ATV","IDV", "LPV", "NFV", "SQV", "TPV", "DRV", "cluster"], axis = 1)
    y_train = train_df[drug].values

    x_test = test_df.drop(["SeqID", "FPV", "ATV","IDV", "LPV", "NFV", "SQV", "TPV", "DRV", "cluster"], axis = 1)
    y_test = test_df[drug].values
    logging.info(f"Data split completed: {len(train_df)} samples in training set, {len(test_df)} samples in testing set")

    #### Removing high correlated
    x_train, x_test = remove_high_correlate(x_train = x_train,
                                            x_test = x_test,
                                            y_train = y_train,
                                            cutoff = c_mi)
    columns = x_train.columns


    #### Normalizing
    x_train_normalized, scaller = data_normalization(dataset = x_train, norm_type = "normalize")
    x_test_normalized = scaller.transform(x_test)

    x_train_normalized = pd.DataFrame(data=x_train_normalized, columns=columns)
    x_test_normalized = pd.DataFrame(data=x_test_normalized, columns=columns)

    # Feature selection using mutual information
    logging.debug("Starting to do Mutual Information.")
    mi_start_time = time.time()


    mi_scores = mutual_info_classif(x_train_normalized, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train_normalized.columns), 1)
    
    logging.info(f"Feature selection completed in {time.time() - mi_start_time:.2f} seconds.")
    colnames_t1 = x_train_normalized.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores'])
    sorted_mi_values = mi_values.sort_values(by='MI_Scores', ascending=False)
    desired_order = sorted_mi_values.index

    x_train_normalized = x_train_normalized.loc[:, desired_order]
    x_test_normalized = x_test_normalized.loc[:, desired_order]
    
    # Compute class weights
    zero = np.sum(y_train == 0)
    one = np.sum(y_train == 1)
    weight_0 = 1.0
    weight_1 = zero / one
    print(weight_1)
    Y1_weight = train_df.iloc[:, 2]
    sample_weight = Y1_weight.apply(lambda x: weight_1 if x == 1 else weight_0)
    sample_weight = sample_weight.values
    grid = {"C": np.array([0.01, 0.1, 1, 10, 100])}

    results = pd.DataFrame()
    predictors = list(x_train_normalized.columns)
    num_features_list = [1, 5, 10, 15, 20, 30, 50, 70, 100, 200, 500, 700, 1000, len(predictors)]
    count = 0

    for num_features in num_features_list:
        selected_predictors = predictors[:num_features]

        X1_selected = x_train_normalized[selected_predictors]
        X2_selected = x_test_normalized[selected_predictors]
        
        model = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', class_weight={0: weight_0, 1: weight_1})
        model_cv = GridSearchCV(model, grid, cv=5)
        model_fit_start_time = time.time()
        model_cv.fit(X1_selected, y_train, sample_weight=sample_weight)
        logging.info(f"Model {count} fitted in {time.time() - model_fit_start_time:.2f} seconds.")
        best_model = model_cv.best_estimator_

        Y1_predicted = best_model.predict(X1_selected)
        Y2_predicted = best_model.predict(X2_selected)
        
        Training_Accuracy = accuracy_score(y_train, Y1_predicted)
        Testing_Accuracy = accuracy_score(y_test, Y2_predicted)
        Training_precision = precision_score(y_train, Y1_predicted)
        Testing_precision = precision_score(y_test, Y2_predicted)
        train_recall = recall_score(y_train, Y1_predicted)
        test_recall = recall_score(y_test, Y2_predicted)
        train_f1 = f1_score(y_train, Y1_predicted)
        test_f1 = f1_score(y_test, Y2_predicted)
        test_auc = roc_auc_score(y_test, best_model.predict_proba(X2_selected)[:,1])
        c = model_cv.best_params_
        
        count += 1
        trem = time.time()
        print(f'Elapsed time for model {count}: {trem - start_time:.2f} seconds')
        result_row = {'C': c['C'], "Number_features": len(selected_predictors),
                      'Predictors': ', '.join(selected_predictors),
                      'Accuracy_Train': Training_Accuracy,
                      'Accuracy_Test': Testing_Accuracy,
                      'Precision_Train': Training_precision, 'Precision_Test': Testing_precision,
                      'Recall_Train': train_recall, 'Recall_Test': test_recall,
                      'F1_Train': train_f1, 'F1_test': test_f1, 
                      'Test_AUC': test_auc}
    
        results = pd.concat([results, pd.DataFrame(result_row, index=[0])], ignore_index=True)    
        model_number = drug + str(len(results.index)) + '_best_logistic_regression_model.pkl'
        # Save the best model to a file using pickle
        with open(model_number, 'wb') as file:
            pickle.dump(best_model, file)

    results.to_csv(drug_name)
    end_time = time.time()  # Termina a count do tempo
    print(f'Elapsed time: {end_time - start_time:.2f} seconds')
    
    return results

    
os.chdir('hiv_pr_benchmarks/data/inhouse_data/rosetta_lr')

data = pd.read_csv('inhouse_rosetta_modeled_classification.csv')

# SQV
drug = 'SQV'
results_df = LogRegClass(data, drug, out='sqv', c_mi=0.8)

# DRV
drug = 'DRV'
results_df = LogRegClass(data, drug, out='drv', c_mi=0.8)

# ATV
drug = 'ATV'
results_df = LogRegClass(data, drug, out='atv', c_mi=0.8)

# FPV
drug = 'FPV'
results_df = LogRegClass(data, drug, out='fpv', c_mi=0.8)

# IDV
drug = 'IDV'
results_df = LogRegClass(data, drug, out='idv', c_mi=0.8)

# LPV
drug = 'LPV'
results_df = LogRegClass(data, drug, out='lpv', c_mi=0.8)

# NFV
drug = 'NFV'
results_df = LogRegClass(data, drug, out='nfv', c_mi=0.8)

# TPV
drug = 'TPV'
results_df = LogRegClass(data, drug, out='tpv', c_mi=0.8)
