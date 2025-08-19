import os
import joblib
import numpy as np
import tensorflow as tf
import pickle
import sys
import requests 

# --- Configuration ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global variables to hold models ---
models = {}

# --- Model Loading Function ---
def load_all_models():
    """Loads all models and tokenizers into the global 'models' dictionary."""
    if models:
        return True
    print("Loading all models, this may take a moment...")
    try:
        models['type_classifier'] = joblib.load(os.path.join(SCRIPT_DIR, 'logistic_type_model.joblib'))
        models['mutation_classifier'] = joblib.load(os.path.join(SCRIPT_DIR, 'logistic_mutation_model.joblib'))
        models['le_type'] = joblib.load(os.path.join(SCRIPT_DIR, 'gene_type_label_encoder.joblib'))
        models['genai_cftr_model'] = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'genai_model2.keras'))
        models['genai_dscam_encoder'] = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'genai_dscam_encoder_model.keras'))
        models['genai_dscam_decoder'] = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, 'genai_dscam_decoder_model.keras'))

        with open(os.path.join(SCRIPT_DIR, 'char2idx.pkl'), 'rb') as f: models['cftr_char2idx'] = pickle.load(f)
        with open(os.path.join(SCRIPT_DIR, 'idx2char.pkl'), 'rb') as f: models['cftr_idx2char'] = pickle.load(f)
        with open(os.path.join(SCRIPT_DIR, 'dscam_char2idx.pkl'), 'rb') as f: models['dscam_char2idx'] = pickle.load(f)
        with open(os.path.join(SCRIPT_DIR, 'dscam_idx2char.pkl'), 'rb') as f: models['dscam_idx2char'] = pickle.load(f)
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"FATAL ERROR: Could not load models. {e}")
        return False

# --- PDB Data Fetching Function ---
def get_protein_pdb_data(gene_type):
    """
    Takes a gene type, finds its PDB ID, and fetches the raw PDB file
    content from the RCSB PDB database.
    """
    pdb_id_map = {
        'cftr': '5UAK',
        'dscam': '1SRR'
    }
    pdb_id = pdb_id_map.get(gene_type.lower())

    if not pdb_id:
        raise ValueError("No PDB ID found for the specified gene type.")

    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=15)
        response.raise_for_status() 
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch PDB data for {pdb_id}: {e}")


# --- Analysis Functions ---
def validate_input_sequence(sequence):
    VALID_CHARS = set('ACGT')
    if not set(sequence).issubset(VALID_CHARS):
        raise ValueError("Error: Invalid characters found. Please use only A, C, G, T.")
    if len(sequence) < 10:
        raise ValueError("Error: Sequence is too short for analysis.")
    return True

def one_hot_encode(sequence, max_len):
    padded_sequence = sequence.ljust(max_len, 'A')
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_encoded = [mapping.get(char, 0) for char in padded_sequence]
    onehot_encoded = np.zeros((len(int_encoded), 4), dtype=int)
    for i, integer in enumerate(int_encoded):
        onehot_encoded[i, integer] = 1
    return onehot_encoded.flatten()

def preprocess_sequence_for_model(sequence, model):
    max_len = model.n_features_in_ // 4
    return one_hot_encode(sequence, max_len).reshape(1, -1)

def classify_gene_type(sequence):
    processed_sequence = preprocess_sequence_for_model(sequence, models['type_classifier'])
    probabilities = models['type_classifier'].predict_proba(processed_sequence)[0]
    predicted_class_index = np.argmax(probabilities)
    confidence = probabilities[predicted_class_index]
    predicted_class_label = models['le_type'].inverse_transform([predicted_class_index])[0]
    return predicted_class_label, confidence

def classify_mutation(sequence):
    processed_sequence = preprocess_sequence_for_model(sequence, models['mutation_classifier'])
    prediction = models['mutation_classifier'].predict(processed_sequence)
    return 'mutated' if prediction[0] == 1 else 'non-mutated'

def correct_gene_sequence(sequence, gene_type):
    gene_type_lower = gene_type.lower()
    if gene_type_lower == 'cftr':
        max_len_input = 100
        max_len_target = 102
        input_tokenized = [models['cftr_char2idx'].get(char, 0) for char in sequence]
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences([input_tokenized], maxlen=max_len_input, padding='post')
        decoder_input = np.zeros((1, max_len_target))
        decoder_input[0, 0] = models['cftr_char2idx']['<start>']
        for i in range(1, max_len_target):
            output_tokens = models['genai_cftr_model'].predict([encoder_input, decoder_input], verbose=0)
            sampled_token_index = np.argmax(output_tokens[0, i-1, :])
            if models['cftr_idx2char'].get(sampled_token_index) == '<end>': break
            decoder_input[0, i] = sampled_token_index
        return "".join([models['cftr_idx2char'].get(int(i), '') for i in decoder_input[0] if i != 0 and models['cftr_idx2char'].get(int(i), '') not in ['<start>', '<end>']])
    elif gene_type_lower == 'dscam':
        max_len_input = 350
        max_len_target = 250
        input_tokenized = [models['dscam_char2idx'].get(char, 0) for char in sequence]
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences([input_tokenized], maxlen=max_len_input, padding='post')
        states_value = models['genai_dscam_encoder'].predict(encoder_input, verbose=0)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = models['dscam_char2idx']['<start>']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = models['genai_dscam_decoder'].predict([target_seq] + states_value, verbose=0)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = models['dscam_idx2char'].get(sampled_token_index, '')
            if (sampled_char == '<end>' or len(decoded_sentence) > max_len_target):
                stop_condition = True
                continue
            decoded_sentence += sampled_char
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        return decoded_sentence
    else:
        return "Unknown gene type, cannot correct."

def run_full_pipeline(sequence):
    """A single function to run the entire analysis pipeline."""
    validate_input_sequence(sequence)

    gene_type, confidence = classify_gene_type(sequence)
    
    CONFIDENCE_THRESHOLD = 0.85
    if confidence < CONFIDENCE_THRESHOLD:
        raise ValueError(f"Low confidence score ({confidence:.2%}). The sequence is unlikely to be a CFTR or DSCAM gene.")

    mutation_status = classify_mutation(sequence)
    
    corrected_display = "No correction needed."
    if mutation_status == 'mutated':
        corrected_sequence = correct_gene_sequence(sequence, gene_type)
        if gene_type.lower() == 'dscam':
            corrected_sequence *= 4
        corrected_display = corrected_sequence
    
    return {
        'geneType': gene_type,
        'confidence': f"{confidence:.2%}",
        'mutationStatus': mutation_status,
        'correctedSequence': corrected_display,
    }

def start_cli():
    """ The function to run the command-line interface. """
    print("\n--- Gene Analysis Pipeline (CLI) ---")
    print("Enter 'exit' or 'quit' to stop.")

    while True:
        print("\n" + "="*40)
        input_sequence = input("Please enter the gene sequence: ").strip().upper()

        if input_sequence in ['EXIT', 'QUIT']:
            print("Exiting pipeline. Goodbye!")
            break
        
        if not input_sequence:
            print("No sequence entered. Please try again.")
            continue
        
        try:
            results = run_full_pipeline(input_sequence)
            print(f"Gene Type: {results['geneType']} (Confidence: {results['confidence']})")
            print(f"Mutation Status: {results['mutationStatus']}")
            print(f"Corrected Sequence: {results['correctedSequence']}")
        except ValueError as e:
            print(f"Validation Error: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

# --- Main Execution Block ---
if __name__ == "__main__":
    if load_all_models():
        start_cli()
