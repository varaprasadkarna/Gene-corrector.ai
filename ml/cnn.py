import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

tf.random.set_seed(42)

# --- 1. Load Data ---
try:
    df = pd.read_excel('for_ml_model.xlsx')
except FileNotFoundError:
    print("Error: 'for_ml_model.xlsx - Sheet1.csv' not found.")
    exit()

def one_hot_encode_dl(sequence, max_len):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    padded_seq = sequence.ljust(max_len, 'A')
    return np.array([mapping.get(char, [0,0,0,0]) for char in padded_seq])

# --- 2. Task 1: Mutation Prediction ---
print("Training CNN for Mutation Prediction...")
sequences_mut = df['mutated'].dropna().tolist()
sequences_non = df['non muated'].dropna().tolist()
all_sequences_mutation = sequences_mut + sequences_non
labels_mutation = np.array([1] * len(sequences_mut) + [0] * len(sequences_non))
max_len_mut = max(len(s) for s in all_sequences_mutation)
X_mutation = np.array([one_hot_encode_dl(s, max_len_mut) for s in all_sequences_mutation])
X_train_mut, X_test_mut, y_train_mut, y_test_mut = train_test_split(
    X_mutation, labels_mutation, test_size=0.2, random_state=42, stratify=labels_mutation)

cnn_mutation = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len_mut, 4)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_mutation.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_mutation.fit(X_train_mut, y_train_mut, epochs=10, batch_size=32, verbose=0)
_, accuracy_mutation = cnn_mutation.evaluate(X_test_mut, y_test_mut, verbose=0)

# --- SAVE THE MUTATION MODEL ---
cnn_mutation.save('cnn_mutation_model.keras')
print("✅ Mutation CNN model saved as 'cnn_mutation_model.keras'")


# --- 3. Task 2: Gene Type Prediction ---
print("\nTraining CNN for Gene Type Prediction...")
sequences_type = df['mutated'].dropna().tolist() + df['non muated'].dropna().tolist()
gene_types = df['gene_type'].dropna().tolist() * 2
le_type = LabelEncoder()
y_type = le_type.fit_transform(gene_types)
max_len_type = max(len(s) for s in sequences_type)
X_type = np.array([one_hot_encode_dl(s, max_len_type) for s in sequences_type])
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X_type, y_type, test_size=0.2, random_state=42, stratify=y_type)

cnn_type = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_len_type, 4)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # For binary classification
])
cnn_type.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_type.fit(X_train_type, y_train_type, epochs=10, batch_size=32, verbose=0)
_, accuracy_type = cnn_type.evaluate(X_test_type, y_test_type, verbose=0)

# --- SAVE THE GENE TYPE MODEL ---
cnn_type.save('cnn_type_model.keras')
joblib.dump(le_type, 'gene_type_label_encoder.joblib') # Save the encoder too
print("✅ Gene Type CNN model saved as 'cnn_type_model.keras'")
print("✅ Label encoder saved as 'gene_type_label_encoder.joblib'")


# --- 4. Results ---
print("\n--- CNN Results ---")
print(f"🧬 Prediction Accuracy for Mutation Status: {accuracy_mutation:.4f}")
print(f"🧬 Prediction Accuracy for Gene Type (CFTR/DSCAM): {accuracy_type:.4f}")
print("-------------------\n")