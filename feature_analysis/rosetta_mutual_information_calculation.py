
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def mutual_info_scores(train_path, drug):
    train = pd.read_csv(train_path)
    x_train = train.drop(["SeqID", "FPV", "ATV","IDV", "LPV", "NFV", "SQV", "TPV", "DRV", "cluster"], axis = 1)
    y_train = train[drug].values
    columns = x_train.columns
    #### Normalizing
    x_train_normalized, scaller = data_normalization(dataset = x_train, norm_type = "normalize")
    x_train_normalized = pd.DataFrame(data=x_train_normalized, columns=columns)

    mi_scores = mutual_info_classif(x_train_normalized, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train_normalized.columns), 1)
    colnames_t1 = x_train_normalized.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores']) # dado cru

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) #

    transposed_mi_values = mi_scores_normalized.T

    return  transposed_mi_values, mi_scores_normalized

def mutual_info_scores_steiner(train_path, drug):
    train = pd.read_csv(train_path)
    x_train = train.drop(["ID","Label"], axis = 1)
    y_train = train["Label"].values
    columns = x_train.columns
    #### Normalizing
    x_train_normalized, scaller = data_normalization(dataset = x_train, norm_type = "normalize")
    x_train_normalized = pd.DataFrame(data=x_train_normalized, columns=columns)

    mi_scores = mutual_info_classif(x_train_normalized, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train_normalized.columns), 1)
    colnames_t1 = x_train_normalized.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores']) # dado cru

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) #

    transposed_mi_values = mi_scores_normalized.T

    return  transposed_mi_values, mi_scores_normalized

def mutual_info_scores_shen(train_path, drug): # For shen data
    train = pd.read_csv(train_path)
    x_train = train.drop(["SeqID", "FPV", "ATV","IDV", "LPV", "NFV", "SQV", "TPV", "DRV"], axis = 1)

    y_train = train[drug].values

    mi_scores = mutual_info_classif(x_train, y_train, random_state=42)
    mi_scores = mi_scores.reshape(len(x_train.columns), 1)
    colnames_t1 = x_train.columns
    mi_values = pd.DataFrame(data=mi_scores, index=colnames_t1, columns=['MI_Scores'])

    scaler = MinMaxScaler()
    mi_scores_normalized = scaler.fit_transform(mi_values)
    mi_scores_normalized = pd.DataFrame(data=mi_scores_normalized, index=colnames_t1, columns=['MI_Scores']) #

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
    "ATV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/ATVtrain.csv",
    "DRV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/DRVtrain.csv",
    "FPV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/FPVtrain.csv",
    "IDV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/IDVtrain.csv",
    "LPV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/LPVtrain.csv",
    "NFV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/NFVtrain.csv",
    "SQV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/SQVtrain.csv",
    "TPV": "hiv_benchmarks/models/inhouse_results/rosetta_lr/TPVtrain.csv",
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


# Shen data
drugs = {
    "ATV": "hiv_benchmarks/models/shen_results/rosetta_lr/atv/ATVtrain.csv",
    "DRV": "hiv_benchmarks/models/shen_results/rosetta_lr/drv/DRVtrain.csv",
    "FPV": "hiv_benchmarks/models/shen_results/rosetta_lr/fpv/FPVtrain.csv",
    "IDV": "hiv_benchmarks/models/shen_results/rosetta_lr/idv/IDVtrain.csv",
    "LPV": "hiv_benchmarks/models/shen_results/rosetta_lr/lpv/LPVtrain.csv",
    "NFV": "hiv_benchmarks/models/shen_results/rosetta_lr/nfv/NFVtrain.csv",
    "SQV": "hiv_benchmarks/models/shen_results/rosetta_lr/sqv/SQVtrain.csv",
    "TPV": "hiv_benchmarks/models/shen_results/rosetta_lr/tpv/TPVtrain.csv",
}

# Loop to process each drug
for drug, train_path in drugs.items():
    print(f"Processing {drug}...")

    # Compute mutual information values
    transposed_mi_values, mi_scores_normalized = mutual_info_scores_shen(train_path, drug)

    # Save processed values
    mi_scores_normalized.to_csv(f"hiv_benchmarks/feature_analysis/rosetta/shen_mi_norm_values_{drug.lower()}.csv", index=False)
    transposed_mi_values.to_csv(f"hiv_benchmarks/feature_analysis/rosetta/shen_mi_values_{drug.lower()}.csv", index=False)

    # Select highest MI score and save
    lista, a = select_highest_mi_score(mi_scores_normalized)
    with open(f"hiv_benchmarks/feature_analysis/rosetta/shen_mi_score_a_{drug.lower()}.txt", "w") as file:
        file.write(str(a))

    print(f"{drug} processed successfully!\n")

# Steiner data
os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark')

drugs = {
    "ATV": "hiv_benchmarks/models/steiner_results/rosetta_lr/atv/atvtrain.csv",
    "DRV": "hiv_benchmarks/models/steiner_results/rosetta_lr/drv/drvtrain.csv",
    "FPV": "hiv_benchmarks/models/steiner_results/rosetta_lr/fpv/fpvtrain.csv",
    "IDV": "hiv_benchmarks/models/steiner_results/rosetta_lr/idv/idvtrain.csv",
    "LPV": "hiv_benchmarks/models/steiner_results/rosetta_lr/lpv/lpvtrain.csv",
    "NFV": "hiv_benchmarks/models/steiner_results/rosetta_lr/nfv/nfvtrain.csv",
    "SQV": "hiv_benchmarks/models/steiner_results/rosetta_lr/sqv/sqvtrain.csv",
    "TPV": "hiv_benchmarks/models/steiner_results/rosetta_lr/tpv/tpvtrain.csv",
}

# Loop to process each drug
for drug, train_path in drugs.items():
    print(f"Processing {drug}...")

    # Compute mutual information values
    transposed_mi_values, mi_scores_normalized = mutual_info_scores_steiner(train_path, drug)

    # Save processed values
    mi_scores_normalized.to_csv(f"hiv_benchmarks/feature_analysis/rosetta/steiner_mi_norm_values_{drug.lower()}.csv", index=False)
    transposed_mi_values.to_csv(f"hiv_benchmarks/feature_analysis/rosetta/steiner_mi_values_{drug.lower()}.csv", index=False)

    # Select highest MI score and save
    lista, a = select_highest_mi_score(mi_scores_normalized)
    with open(f"hiv_benchmarks/feature_analysis/rosetta/steiner_mi_score_a_{drug.lower()}.txt", "w") as file:
        file.write(str(a))

    print(f"{drug} processed successfully!\n")
