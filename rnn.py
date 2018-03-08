from keras.models import Sequential
from P3CHelpers import *
import sys

from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

'''
RECURRENT NEURAL NETWORK POETRY GENERATION IMPLEMENTATION
Train a character-based LSTM -> single layer of 100-200 LSTM units
fully-connected output layer with softmax nonlinearity

minimize categorial cross-entropy
train for a sufficient # of epochs so that loss converges
do not need to keep track of overfitting / keep a validation set

training data: sequences of 40 chars from sonnet corpus
take all possible subsequences of 40 consecutive chars from dataset
pick only sequences starting every nth char

generate poems -> draw softmax samples from trained models
play with temperature parameter, which controls variance of sampled text
'''

filename = "data/shakespeare.txt"
# Load in a list of words from the specified file; remove non-alphanumeric characters
# and make all chars lowercase.
# TODO edit load_word_list for preprocessing of data
sample_text = load_word_list(filename)
print(sample_text)
# # Create dictionary mapping unique words to their one-hot-encoded index
# word_to_index = generate_onehot_dict(sample_text)
# # Create training data using default window size
# trainX, trainY = generate_traindata(sample_text, word_to_index)

# # vocab_size = number of unique words in our text file. Will be useful when adding layers
# # to your neural network
# vocab_size = len(word_to_index)
# model = Sequential()
# model.add(Dense(num_latent_factors, input_shape=(vocab_size,)))
# model.add(Dense(vocab_size, activation = 'softmax'))

# model.compile(optimizer='rmsprop',
#           loss='categorical_crossentropy',
#           metrics=['accuracy'])

# model.fit(trainX, trainY, epochs=10, batch_size=32)
# # Extract weights for hidden layer, set <weights> variable below
# weights = (model.layers[0].get_weights())[0]

# print("Hidden Layer weights dimension: ", weights.shape)

# weightsoutput = (model.layers[1].get_weights())[0]