import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC # --- MODIFIED: Import SVC instead of LinearSVC ---
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
        onehot_encoded[i, integer] = 1
    return onehot_encoded.flatten()

# --- 2. Task 1: Mutation Prediction ---
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

print("Training SVM for Mutation Prediction...")
# --- MODIFIED: Use SVC with probability=True ---
svm_mutation = SVC(kernel='linear', probability=True, random_state=42, max_iter=1000)
svm_mutation.fit(X_train_mut, y_train_mut)
y_pred_mut = svm_mutation.predict(X_test_mut)
accuracy_mutation = accuracy_score(y_test_mut, y_pred_mut)

# --- SAVE THE MUTATION MODEL ---
joblib.dump(svm_mutation, 'svm_mutation_model.joblib')
print("✅ Mutation SVM model saved as 'svm_mutation_model.joblib'")

# --- 3. Task 2: Gene Type Prediction ---
print("\nPreparing data for Gene Type Prediction...")
# Correctly handle cases where columns might have different numbers of non-NaN values
sequences_mut_type = df['mutated'].dropna()
sequences_non_mut_type = df['non muated'].dropna()
sequences_type = sequences_mut_type.tolist() + sequences_non_mut_type.tolist()

# Ensure labels match the number of sequences
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

print("Training SVM for Gene Type Prediction...")
# --- MODIFIED: Use SVC with probability=True ---
svm_type = SVC(kernel='linear', probability=True, random_state=42, max_iter=1000)
svm_type.fit(X_train_type, y_train_type)
y_pred_type = svm_type.predict(X_test_type)
accuracy_type = accuracy_score(y_test_type, y_pred_type)

# --- SAVE THE GENE TYPE MODEL ---
joblib.dump(svm_type, 'svm_type_model.joblib')
joblib.dump(le_type, 'gene_type_label_encoder.joblib')
print("✅ Gene Type SVM model saved as 'svm_type_model.joblib'")
print("✅ Label encoder saved as 'gene_type_label_encoder.joblib'")

# --- 4. Results ---
print("\n--- SVM Results (With Probability Support) ---")
print(f"🧬 Prediction Accuracy for Mutation Status: {accuracy_mutation:.4f}")
print(f"🧬 Prediction Accuracy for Gene Type (CFTR/DSCAM): {accuracy_type:.4f}")
print("------------------------------------------\n")