import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import pickle

# Load data directly from the Excel file
df = pd.read_excel("for_gen_ai_model.xlsx")

# --- ADDED SECTION ---
# Ensure sequence columns are strings to prevent errors
df['input_sequence'] = df['input_sequence'].astype(str)
df['target_sequence'] = df['target_sequence'].astype(str)
# --- END OF ADDED SECTION ---

# Add start and end tokens to target sequences
df['target_sequence_mod'] = df['target_sequence'].apply(lambda x: '<start> ' + x + ' <end>')

# Create a vocabulary
all_chars = set(''.join(df['input_sequence']) + ''.join(df['target_sequence']))
dna_vocab = sorted(list(all_chars))
dna_vocab.extend(['<pad>', '<start>', '<end>'])
dna_vocab = sorted(list(set(dna_vocab))) # Remove duplicates
vocab_size = len(dna_vocab)

# Create char2idx and idx2char mappings
char2idx = {char: idx for idx, char in enumerate(dna_vocab)}
idx2char = {idx: char for idx, char in enumerate(dna_vocab)}

# Save the mappings
with open('char2idx.pkl', 'wb') as f:
    pickle.dump(char2idx, f)
with open('idx2char.pkl', 'wb') as f:
    pickle.dump(idx2char, f)

# Tokenize input sequences
input_sequences = [[char2idx[char] for char in seq] for seq in df['input_sequence']]

# Tokenize target sequences
target_sequences = []
for seq in df['target_sequence_mod']:
    token_list = []
    for part in seq.split(' '):
        if part in ['<start>', '<end>']:
            if part in char2idx:
                token_list.append(char2idx[part])
        else:
            for char in part:
                if char in char2idx:
                    token_list.append(char2idx[char])
    target_sequences.append(token_list)

# Pad sequences
max_len_input = 100
max_len_target = 102
encoder_input_data = pad_sequences(input_sequences, maxlen=max_len_input, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# --- THIS IS THE CORRECTED SECTION ---
# Prepare decoder target data (one-hot encoded and shifted)
decoder_target_data = np.zeros((len(target_sequences), max_len_target, vocab_size), dtype='float32')

# Iterate over the PADDED sequences ('decoder_input_data')
for i, seq in enumerate(decoder_input_data):
    for t, char_idx in enumerate(seq):
        if t > 0:
            # This will now not go out of bounds
            decoder_target_data[i, t - 1, char_idx] = 1.0
# --- END OF CORRECTION ---

# Define the model
embedding_dim = 128
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=100,
          validation_split=0.2)

# Save the model
model.save("genai_model2.keras")

print("\nModel training complete and saved as genai_model.keras")
