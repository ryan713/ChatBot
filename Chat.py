import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Input

# define documents
inp_docs = open('mynewinput.txt', "r").read().split('\n')
out_docs = open('mynewoutput.txt', "r").read().split('\n')

n_train = len(inp_docs)

# prepare tokenizers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inp_docs + out_docs)

vocab_size = len(tokenizer.word_index) + 1

max_length = 20

model = load_model('s2s.h5')

encoder_inputs = model.input[0]
_, eh, ec = (model.get_layer('encoder_lstm')).output
encoder_states = [eh, ec]

# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_embedding_layer = model.get_layer('embedding_layer')(decoder_inputs)
decoder_lstm = model.get_layer('decoder_lstm')
decoder_dense = model.get_layer('decoder_dense')

### INFERENCE MODEL ###

latent_dim = 256

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape = (latent_dim,), name = 'decoder_state_input_h')
decoder_state_input_c = Input(shape = (latent_dim,), name = 'decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embedding_layer,
                                                    initial_state = decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Final decoder model
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs2] + decoder_states2)

# Build reverse vocab to decode output to readable sentences
reverse_vocab = dict()
forward_vocab = dict()
for word, i in tokenizer.word_index.items():
    reverse_vocab[i] = word
    forward_vocab[word] = i

reverse_vocab[0] = 'null'
forward_vocab['null'] = 0

# comparator function for sorting all candidates by probabilities
def comparator(tupleElement):
    
    # Length normalization
    return tupleElement[0][1] / (len(tupleElement[0][0].split(' ')))

while True:
    input_sentence = 'start ' + input('You: ') + ' end'
    tokenized_sentence = tokenizer.texts_to_sequences([input_sentence])
    encoder_input = pad_sequences(tokenized_sentence, maxlen = max_length, padding = 'post')
    
    # Decode this sentence.
    input_seq = encoder_input

    # Beam width for beam search
    beam_width = 5
    
    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
    decoded_sentences = []
    
    # state values for all possible sentences
    states_value = []
    
    # Encode the input as state vectors.
    encoder_states_value = encoder_model.predict(input_seq, batch_size = 1)
    
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = forward_vocab['start']
    
    # predict probabiltes for next possible token
    output_tokens, h, c = decoder_model.predict([target_seq] + encoder_states_value, batch_size = 1)
        
    # Update states
    initial_states_values = [h, c]
        
    # Sample a token
    sampled_token_probs = output_tokens[0, -1, :]
    max_probs_tokens = sampled_token_probs.argsort()[::-1][:beam_width]

    # Top #beam_width candidates.
    for i in range(beam_width):
        states_value.append(initial_states_values)
        decoded_sentences.append((reverse_vocab[max_probs_tokens[i]],
                                  np.log(sampled_token_probs[max_probs_tokens[i]])))
    
    while True:

        # empty tuple of possible intermediate candidates
        candidates = []
                                 
        # counter for stopping the loop
        sentence_end_count = 0
        
        for i in range(beam_width):
            target_seq = np.zeros((1,1))
            
            # get token of last word of sentence for i'th candidate
            target_seq[0, 0] = forward_vocab[decoded_sentences[i][0].split(' ')[-1]]

            if target_seq[0, 0] == forward_vocab['end']:
                candidates.append((decoded_sentences[i], i))
                sentence_end_count = sentence_end_count + 1
                continue
            
            # predict probabiltes for next possible token
            output_tokens, hi, ci = decoder_model.predict([target_seq] + states_value[i], batch_size = 1)
        
            # Update states
            states_value[i] = [hi, ci]
            
            # Sample a token
            sampled_token_probs = output_tokens[0, -1, :]
            max_probs_tokens = sampled_token_probs.argsort()[::-1][:beam_width]
            
            for j in range(beam_width):
                if decoded_sentences[i][0].split(' ')[-1] == 'end':
                    continue
                candidate = (decoded_sentences[i][0] + ' ' + reverse_vocab[max_probs_tokens[j]],
                             decoded_sentences[i][1] + np.log(sampled_token_probs[max_probs_tokens[j]]))
                candidates.append((candidate, i))

        if sentence_end_count == beam_width:
            break
            
        top_candidates = sorted(candidates, key = comparator)[::-1][:beam_width]
        temp_states = []

        for i, candidate in enumerate(top_candidates):
            temp_states.append(states_value[candidate[1]])
            decoded_sentences[i] = candidate[0]
                
        states_value = temp_states

    for i in range(beam_width):
        response = decoded_sentences[i][0][:-4].replace(' comma', ',')
        print('HotorBot(' + str(i+1) + '): ' + str(response) + '.')

'''
response = ''
last_word = ''
    
for i in range(100):
    # Cosine Similarity
    emb_distances = np.dot(embedding_matrix, output[i])
    word = reverse_vocab[np.argmax(emb_distances)]
    if last_word == word:
        break
    else:
        last_word = word
    response = response + word + ' '
'''
