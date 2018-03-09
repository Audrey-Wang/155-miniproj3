import numpy as np

def load_word_list(path):
    """
    Loads a list of the words from the file at path <path>, removing all
    non-alpha-numeric characters from the file.
    """
    with open(path) as handle:
        # Load a list of whitespace-delimited words from the specified file
        raw_text = handle.read().strip().split()
        # Strip non-alphanumeric characters from each word
        alphanumeric_words = map(lambda word: ''.join(char for char in word if char.isalnum()), raw_text)
        # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
        alphanumeric_words = filter(lambda word: len(word) > 0, alphanumeric_words)
        # Convert each word to lowercase and return the result
        return list(map(lambda word: word.lower(), alphanumeric_words))

def get_chars(list):
    string = ""
    for word in list:
        if word.isdigit():
            continue
        string += word
    return string

def generate_onehot_dict(let_list):
    """
    Takes a list of the words in a text file, returning a dictionary mapping
    words to their index in a one-hot-encoded representation of the words.
    """
    let_to_index = {}
    i = 0
    for letter in let_list:
        if letter not in let_to_index:
            let_to_index[letter] = i
            i += 1
        if len(let_to_index) == 26:
            break
    return let_to_index

def get_let_rep(let_to_index, letter):
    unique_lets = let_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_lets))
    feature_representation[let_to_index[letter]] = 1
    return feature_representation    

def one_hot_to_let(let_to_index, one_hot):
    unique_lets = let_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    for i in range(0, len(one_hot)):
        if (one_hot[i] == 1.):
            return list(unique_lets)[i]
    return -1

def get_let_list_rep(let_to_index, let_list):
    ret = []
    for letter in let_list:
        let = get_let_rep(let_to_index, letter)
        for i in let:
            ret.append(i)
    return ret

def generate_traindata(let_to_index, let_list):
    trainX = get_let_list_rep(let_to_index, let_list[0:40])
    trainY = get_let_rep(let_to_index, let_list[40])

    return (np.array(trainX), np.array(trainY))

def most_similar_pairs(weight_matrix, word_to_index):
    """
    For each word a in our vocabulary, computes the most similar word b to a, along with the
    cosine similarity of a and b.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

    Returns: 
        A list of WordPair instances sorted in decreasing order of similarity,
        one representing each word <vocab_word> and its most similar word.
    """
    word_to_feature_repr = get_word_to_feature_repr(weight_matrix, word_to_index)
    result = []
    for word in word_to_feature_repr:
        result.append(most_similar_word(word_to_feature_repr, word))
    return sort_by_similarity(result)

def most_similar_word(word_to_feature_repr, input_word):
    """
    Given a dictionary mapping words to their feature representations (word_to_feature_repr),
    returns the a WordPair instance corresponding to the word
    whose feature vector is most similar to the feature representation of the
    passed-in word (input_word).
    """
    best_word = None
    best_similarity = 0
    input_vec = word_to_feature_repr[input_word]
    for word, feature_vec in word_to_feature_repr.items():
        similarity = get_similarity(input_vec, feature_vec)
        if similarity > best_similarity and np.linalg.norm(feature_vec - input_vec) != 0:
            best_similarity = similarity
            best_word = word
    return WordPair(input_word, best_word, best_similarity)

def get_word_to_feature_repr(weight_matrix, word_to_index):
    """
    Returns a dictionary mapping each word in our vocabulary to its one-hot-encoded
    feature representation.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.
    """
    assert(weight_matrix is not None)
    word_to_feature_repr = {}
    for word, one_hot_idx in word_to_index.items():
        word_to_feature_repr[word] = weight_matrix[one_hot_idx]
    return word_to_feature_repr
