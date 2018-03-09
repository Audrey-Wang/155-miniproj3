from keras.models import Sequential
from P3CHelpers import *
import sys

from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Embedding
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

num_letters = 26


def rnn():
	filename = "data/shakespeare.txt"
# Load in a list of words from the specified file; remove non-alphanumeric characters
# and make all chars lowercase.
# TODO edit load_word_list for preprocessing of data
	chars = get_chars(load_word_list(filename))
	num = len(chars)
	n = num // 40 // 10

	# Create dictionary mapping letters to their one-hot-encoded index
	let_to_index = generate_onehot_dict(chars)
	# Create training data
	# while (i != n):
	# 	trainX, trainY = generate_traindata(chars[i:i+40], let_to_index)

	trainX, trainY = generate_traindata(let_to_index, chars[0:41])

	trainX = np.reshape(trainX, (40, 1, 26))
	#trainX = np.reshape(trainX, (1, 1040,))
	model = Sequential()
	#100 to 200 units
	#model.add(Embedding(num_letters, output_dim = 256))
	model.add(LSTM(26, input_shape = (1, 26), return_sequences = True))
	#model.add(LSTM(26, input_shape = (1,26*40), return_sequences = False))

	print(model.output_shape)
	#model.add(LSTM(150, return_sequences = True, batch_size = 32, stateful = True, input_shape = (num_letters, )))
	model.add(Dense(26, activation = 'softmax'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	model.fit(trainX, trainY, epochs=10, batch_size=32)

	score = model.evaluate(trainX, trainY, verbose=0)
	print(score)

	x, y = generate_traindata(chars[0:41], let_to_index)
	prediction = model.predict_on_batch(y)
	print(prediction)
	# prediction[(prediction.tolist()).index(max(prediction))] = 1.
	# for i in range(0, len(prediction)):
	# 	if i != (prediction.tolist()).index(max(prediction)):
	# 		prediction[i] = 0.
	# start = one_hot_to_let(let_to_index, prediction)
	# print(start)
rnn()

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