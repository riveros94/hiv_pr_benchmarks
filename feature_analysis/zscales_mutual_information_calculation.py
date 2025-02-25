import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Define a function to generate the new column names
def generate_new_column_names(df_zscales):
    num_columns = len(df_zscales.columns)
    pos = list(range(1,495))
    new_columns = []
    for i in range(num_columns):
        group_number = i // 5
        within_group_number = i % 5 + 1
        new_columns.append(f'{pos[group_number]}_{within_group_number}')
    df_zscales.columns = new_columns
    return df_zscales

def mutual_info_scores(train_path, drug):

    train = pd.read_csv(train_path)
    x_train = train.drop(["seqid", drug, "cluster"], axis = 1)
    x_train = generate_new_column_names(x_train)

    y_train = train[drug].values

    mi_scores = mutual_info_classif(x_train, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train.columns), 1)
    colnames_t1 = x_train.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores']) # dado cru

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) # Para figura pymol

    transposed_mi_values = mi_scores_normalized.T

    return  transposed_mi_values, mi_scores_normalized

def mutual_info_scores_steiner(train_path, drug): # For steiner data
    train = pd.read_csv(train_path)
    x_train = train.drop(["ID", "Label"], axis = 1)
    x_train = generate_new_column_names(x_train)

    y_train = train['Label'].values

    mi_scores = mutual_info_classif(x_train, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train.columns), 1)
    colnames_t1 = x_train.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores']) # dado cru

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) # Para figura pymol


    transposed_mi_values = mi_scores_normalized.T

    return  transposed_mi_values, mi_scores_normalized


def mutual_info_scores_shen(train_path, drug): # For shen data
    train = pd.read_csv(train_path)
    x_train = train.drop(["SeqID", drug], axis = 1)
    x_train = generate_new_column_names(x_train)

    y_train = train[drug].values

    mi_scores = mutual_info_classif(x_train, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train.columns), 1)
    colnames_t1 = x_train.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores']) # dado cru

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) # Para figura pymol


    transposed_mi_values = mi_scores_normalized.T

    return  transposed_mi_values, mi_scores_normalized

def select_highest_mi_score(df_normalized):

    highest_scores = {}

    for index, row in df_normalized.iterrows():
        try:
            amino_acid = int(index.split('_')[0])
        except ValueError:
            continue

        if amino_acid in highest_scores:
            if row['MI_Scores'] > highest_scores[amino_acid]:
                highest_scores[amino_acid] = row['MI_Scores']
        else:
            highest_scores[amino_acid] = row['MI_Scores']

    highest_scores_vector = [highest_scores[key] for key in sorted(highest_scores.keys())]

    # Format vector for R
    highest_scores_vector_r = ', '.join(map(str, highest_scores_vector))

    return highest_scores_vector, highest_scores_vector_r

# In-house data
drugs = {
    "ATV": "hiv_benchmarks/models/inhouse_results/zscales_lr/atv/train.csv",
    "DRV": "hiv_benchmarks/models/inhouse_results/zscales_lr/drv/train.csv",
    "FPV": "hiv_benchmarks/models/inhouse_results/zscales_lr/fpv/train.csv",
    "IDV": "hiv_benchmarks/models/inhouse_results/zscales_lr/idv/train.csv",
    "LPV": "hiv_benchmarks/models/inhouse_results/zscales_lr/lpv/train.csv",
    "NFV": "hiv_benchmarks/models/inhouse_results/zscales_lr/nfv/train.csv",
    "SQV": "hiv_benchmarks/models/inhouse_results/zscales_lr/sqv/train.csv",
    "TPV": "hiv_benchmarks/models/inhouse_results/zscales_lr/tpv/train.csv",
}

# Loop to process each drug
for drug, train_path in drugs.items():
    print(f"Processing {drug}...")

    # Compute mutual information values
    transposed_mi_values, mi_scores_normalized = mutual_info_scores(train_path, drug)

    # Save processed values
    mi_scores_normalized.to_csv(f"hiv_benchmarks/feature_analysis/inhouse_mi_norm_values_{drug.lower()}.csv", index=False)
    transposed_mi_values.to_csv(f"hiv_benchmarks/feature_analysis/inhouse_mi_values_{drug.lower()}.csv", index=False)

    # Select highest MI score and save
    lista, a = select_highest_mi_score(mi_scores_normalized)
    with open(f"hiv_benchmarks/feature_analysis/inhouse_mi_score_a_{drug.lower()}.txt", "w") as file:
        file.write(str(a))

    print(f"{drug} processed successfully!\n")

# Steiner data
drugs = {
    "ATV": "hiv_benchmarks/models/steiner_results/zscales_lr/atv/train.csv",
    "DRV": "hiv_benchmarks/models/steiner_results/zscales_lr/drv/train.csv",
    "FPV": "hiv_benchmarks/models/steiner_results/zscales_lr/fpv/train.csv",
    "IDV": "hiv_benchmarks/models/steiner_results/zscales_lr/idv/train.csv",
    "LPV": "hiv_benchmarks/models/steiner_results/zscales_lr/lpv/train.csv",
    "NFV": "hiv_benchmarks/models/steiner_results/zscales_lr/nfv/train.csv",
    "SQV": "hiv_benchmarks/models/steiner_results/zscales_lr/sqv/train.csv",
    "TPV": "hiv_benchmarks/models/steiner_results/zscales_lr/tpv/train.csv",
}

# Loop to process each drug
for drug, train_path in drugs.items():
    print(f"Processing {drug}...")

    # Compute mutual information values
    transposed_mi_values, mi_scores_normalized = mutual_info_scores_steiner(train_path, drug)

    # Save processed values
    mi_scores_normalized.to_csv(f"hiv_benchmarks/feature_analysis/zscales/steiner_mi_norm_values_{drug.lower()}.csv", index=False)
    transposed_mi_values.to_csv(f"hiv_benchmarks/feature_analysis/zscales/steiner_mi_values_{drug.lower()}.csv", index=False)

    # Select highest MI score and save
    lista, a = select_highest_mi_score(mi_scores_normalized)
    with open(f"hiv_benchmarks/feature_analysis/zscales/steiner_mi_score_a_{drug.lower()}.txt", "w") as file:
        file.write(str(a))

    print(f"{drug} processed successfully!\n")

# Shen data
os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark')

drugs = {
    "ATV": "hiv_benchmarks/models/shen_results/zscales_lr/atv/train.csv",
    "DRV": "hiv_benchmarks/models/shen_results/zscales_lr/drv/train.csv",
    "FPV": "hiv_benchmarks/models/shen_results/zscales_lr/fpv/train.csv",
    "IDV": "hiv_benchmarks/models/shen_results/zscales_lr/idv/train.csv",
    "LPV": "hiv_benchmarks/models/shen_results/zscales_lr/lpv/train.csv",
    "NFV": "hiv_benchmarks/models/shen_results/zscales_lr/nfv/train.csv",
    "SQV": "hiv_benchmarks/models/shen_results/zscales_lr/sqv/train.csv",
    "TPV": "hiv_benchmarks/models/shen_results/zscales_lr/tpv/train.csv",
}

# Loop to process each drug
for drug, train_path in drugs.items():
    print(f"Processing {drug}...")

    # Compute mutual information values
    transposed_mi_values, mi_scores_normalized = mutual_info_scores_shen(train_path, drug)

    # Save processed values
    mi_scores_normalized.to_csv(f"hiv_benchmarks/feature_analysis/shen_mi_norm_values_{drug.lower()}.csv", index=False)
    transposed_mi_values.to_csv(f"hiv_benchmarks/feature_analysis/shen_mi_values_{drug.lower()}.csv", index=False)

    # Select highest MI score and save
    lista, a = select_highest_mi_score(mi_scores_normalized)
    with open(f"hiv_benchmarks/feature_analysis/shen_mi_score_a_{drug.lower()}.txt", "w") as file:
        file.write(str(a))

    print(f"{drug} processed successfully!\n")
