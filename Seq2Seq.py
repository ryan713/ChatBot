import numpy as np
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Embedding, Input, Dense, LSTM
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

# define documents
inp_docs = open('mynewinput.txt', "r").read().split('\n')
out_docs = open('mynewoutput.txt', "r").read().split('\n')

# prepare tokenizers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inp_docs + out_docs)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: ' + str(vocab_size))

n_train = len(inp_docs)
print('n_train: ' + str(n_train))

# Build reverse vocab to decode output to readable sentences
reverse_vocab = dict()
forward_vocab = dict()
for word, i in tokenizer.word_index.items():
    # print(str(i) + ': ' + word)
    reverse_vocab[i] = word
    forward_vocab[word] = i

reverse_vocab[0] = 'null'
forward_vocab['null'] = 0

# integer encode the documents.
encoded_inp_docs = tokenizer.texts_to_sequences(inp_docs)
encoded_out_docs = tokenizer.texts_to_sequences(out_docs)

# pad documents to a max length of 30 words.
max_length = 20
encoder_input_data = pad_sequences(encoded_inp_docs, maxlen = max_length, padding = 'post')
decoder_input_data = pad_sequences(encoded_out_docs, maxlen = max_length, padding = 'post')

# encoder_input_data = to_categorical(np.array(encoder_input_data), vocab_size)
# decoder_input_data = to_categorical(np.array(decoder_input_data), vocab_size)

# One-hot encode decoder targets and shift decoder targets one time step ahead.
decoder_target_data = np.zeros((n_train, max_length, vocab_size))
for i in range(n_train):
    for t in range(max_length - 1):
        if decoder_input_data[i, t+1] > 0:
            decoder_target_data[i, t, decoder_input_data[i, t+1]] = 1

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

'''
# Make word embedding for decoder_target_data.
new_decoder_target_data = np.zeros((n_train, max_length, 50))

for i in range(decoder_target_data.shape[0]):
    for j in range(decoder_target_data.shape[1]):
        new_decoder_target_data[i][j] = embedding_matrix[decoder_target_data[i][j]]

decoder_target_data = new_decoder_target_data '''

print("embedding_matrix_shape: " + str(embedding_matrix.shape))
print("encoder_input_data_shape: " + str(encoder_input_data.shape))
print("decoder_input_data_shape: " + str(decoder_input_data.shape))
print("decoder_target_data_shape: " + str(decoder_target_data.shape))

# Configure the shapes
num_encoder_tokens = max_length
num_decoder_tokens = vocab_size
latent_dim = 256
my_batch_size = 10

### DEFINE THE MODEL ###

# Encoder and decoder share the same embedding layer
shared_embedding_layer = Embedding(vocab_size,
                                   embedding_matrix.shape[1],
                                   weights = [embedding_matrix],
                                   trainable = False,
                                   name = 'embedding_layer')

# Define an input sequence and process it.
encoder_inputs = Input(batch_shape = (None, None), name = 'encoder_input')
encoder_embedding_layer = shared_embedding_layer(encoder_inputs)
encoder_lstm = LSTM(latent_dim,
                    return_sequences = False,
                    return_state = True,
                    name = 'encoder_lstm')
_, state_h, state_c = encoder_lstm(encoder_embedding_layer)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(batch_shape = (None, None), name = 'decoder_input')
decoder_embedding_layer = shared_embedding_layer(decoder_inputs)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim,
                    return_state = True,
                    return_sequences = True,
                    name = 'decoder_lstm')

decoder_lstm_output, h, c = decoder_lstm(decoder_embedding_layer, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax', name = 'decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_output)

# Define the model that will turn
# encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

# summarize
print(model.summary())

# plot the model
# plot_model(model, to_file = 'model.png', show_shapes = True)

# fit the model
for i in range(10):
    print('Epochs: ' + str((i * 10) + 1) + '-' + str((i+1) * 10))
    loss, acc = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
    print('loss - ' + str(loss) + '\t' + 'acc - ' + str(acc))
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          verbose = 0,
          batch_size = my_batch_size,
          epochs = 10)

loss, acc = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
print('loss - ' + str(loss) + '\t' + 'acc - ' + str(acc))

# Save the models
model.save('s2s.h5')
print('Model Saved Successfully.')

### INFERENCE MODELS ###

# Encoder setup
encoder_model = Model(encoder_inputs, encoder_states)

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

'''
with open('encoder_model.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('encoder_model_weights.h5')

with open('decoder_model.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('decoder_model_weights.h5')
'''

for i in range(25):
    test_on = i
    einput = encoder_input_data[test_on]
    einput = np.expand_dims(einput, axis = 0)
    
    # Decode this sentence.
    input_seq = einput
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, batch_size = 1)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = forward_vocab['start']
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''
    
    while True:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, batch_size = 1)
        # Sample a token
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_vocab[sampled_token_index]
        
        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'end' or sampled_token_index == 0 or len(decoded_sentence.split('\n')) >= max_length):
            break
        
        decoded_sentence = decoded_sentence + sampled_char + ' '
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros(target_seq.shape)
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    print()
    print('----------------------- ' + str(test_on) + ' -----------------------' )
    print('Input: ' + inp_docs[test_on])
    print('Exp output: ' + out_docs[test_on])
    print('Predicted:' + decoded_sentence)



