import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. Load and Preprocess Data ---
try:
    df = pd.read_excel('for_ml_model.xlsx')
except FileNotFoundError:
    print("Error: 'for_ml_model.xlsx' not found.")
    exit()

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_encoded = [mapping.get(char, -1) for char in sequence if char in mapping]
    onehot_encoded = np.zeros((len(int_encoded), 4), dtype=int)
    for i, integer in enumerate(int_encoded):
        if integer != -1:
            onehot_encoded[i, integer] = 1
    return onehot_encoded.flatten()

# --- 2. Task 1: Mutation Prediction ---
if not os.path.exists('logistic_mutation_model.joblib'):
    print("Preparing data for Mutation Prediction...")
    sequences_mut = df['mutated'].dropna().tolist()
    sequences_non = df['non muated'].dropna().tolist()
    all_sequences_mutation = sequences_mut + sequences_non
    labels_mutation = [1] * len(sequences_mut) + [0] * len(sequences_non)

    print("Encoding sequences for mutation task...")
    max_len_mut = max(len(s) for s in all_sequences_mutation)
    X_mutation = np.array([one_hot_encode(s.ljust(max_len_mut, 'A')) for s in all_sequences_mutation])
    y_mutation = np.array(labels_mutation)

    X_train_mut, X_test_mut, y_train_mut, y_test_mut = train_test_split(
        X_mutation, y_mutation, test_size=0.2, random_state=42, stratify=y_mutation)

    print("Training Model for Mutation Prediction...")
    model_mutation = LogisticRegression(random_state=42, max_iter=1000)
    model_mutation.fit(X_train_mut, y_train_mut)
    y_pred_mut = model_mutation.predict(X_test_mut)
    accuracy_mutation = accuracy_score(y_test_mut, y_pred_mut)

    joblib.dump(model_mutation, 'logistic_mutation_model.joblib')
    print("✅ Mutation model saved as 'logistic_mutation_model.joblib'")
else:
    print("🔁 Skipping training: 'logistic_mutation_model.joblib' already exists.")
    model_mutation = joblib.load('logistic_mutation_model.joblib')
    sequences_mut = df['mutated'].dropna().tolist()
    sequences_non = df['non muated'].dropna().tolist()
    all_sequences_mutation = sequences_mut + sequences_non
    labels_mutation = [1] * len(sequences_mut) + [0] * len(sequences_non)
    max_len_mut = max(len(s) for s in all_sequences_mutation)
    X_mutation = np.array([one_hot_encode(s.ljust(max_len_mut, 'A')) for s in all_sequences_mutation])
    y_mutation = np.array(labels_mutation)
    _, X_test_mut, _, y_test_mut = train_test_split(
        X_mutation, y_mutation, test_size=0.2, random_state=42, stratify=y_mutation)
    y_pred_mut = model_mutation.predict(X_test_mut)
    accuracy_mutation = accuracy_score(y_test_mut, y_pred_mut)

# --- 3. Task 2: Gene Type Prediction ---
if not os.path.exists('logistic_type_model.joblib'):
    print("\nPreparing data for Gene Type Prediction...")
    sequences_mut_type = df['mutated'].dropna()
    sequences_non_mut_type = df['non muated'].dropna()
    sequences_type = sequences_mut_type.tolist() + sequences_non_mut_type.tolist()

    gene_types_mut = df.loc[sequences_mut_type.index, 'gene_type']
    gene_types_non_mut = df.loc[sequences_non_mut_type.index, 'gene_type']
    gene_types = gene_types_mut.tolist() + gene_types_non_mut.tolist()

    le_type = LabelEncoder()
    y_type = le_type.fit_transform(gene_types)

    print("Encoding sequences for gene type task...")
    max_len_type = max(len(s) for s in sequences_type)
    X_type = np.array([one_hot_encode(s.ljust(max_len_type, 'A')) for s in sequences_type])

    X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
        X_type, y_type, test_size=0.2, random_state=42, stratify=y_type)

    print("Training Model for Gene Type Prediction...")
    model_type = LogisticRegression(random_state=42, max_iter=1000)
    model_type.fit(X_train_type, y_train_type)
    y_pred_type = model_type.predict(X_test_type)
    accuracy_type = accuracy_score(y_test_type, y_pred_type)

    joblib.dump(model_type, 'logistic_type_model.joblib')
    joblib.dump(le_type, 'gene_type_label_encoder.joblib')
    print("✅ Gene Type model saved as 'logistic_type_model.joblib'")
    print("✅ Label encoder saved as 'gene_type_label_encoder.joblib'")
else:
    print("🔁 Skipping training: 'logistic_type_model.joblib' already exists.")
    model_type = joblib.load('logistic_type_model.joblib')
    le_type = joblib.load('gene_type_label_encoder.joblib')
    sequences_mut_type = df['mutated'].dropna()
    sequences_non_mut_type = df['non muated'].dropna()
    sequences_type = sequences_mut_type.tolist() + sequences_non_mut_type.tolist()
    gene_types_mut = df.loc[sequences_mut_type.index, 'gene_type']
    gene_types_non_mut = df.loc[sequences_non_mut_type.index, 'gene_type']
    gene_types = gene_types_mut.tolist() + gene_types_non_mut.tolist()
    y_type = le_type.transform(gene_types)
    max_len_type = max(len(s) for s in sequences_type)
    X_type = np.array([one_hot_encode(s.ljust(max_len_type, 'A')) for s in sequences_type])
    _, X_test_type, _, y_test_type = train_test_split(
        X_type, y_type, test_size=0.2, random_state=42, stratify=y_type)
    y_pred_type = model_type.predict(X_test_type)
    accuracy_type = accuracy_score(y_test_type, y_pred_type)

# --- 4. Results ---
print("\n--- Model Results (Optimized) ---")
print(f"🧬 Prediction Accuracy for Mutation Status: {accuracy_mutation:.4f}")
print(f"🧬 Prediction Accuracy for Gene Type (CFTR/DSCAM): {accuracy_type:.4f}")
print("------------------------------------------\n")
