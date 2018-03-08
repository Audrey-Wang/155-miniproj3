import os
import numpy as np

from nltk.corpus import cmudict

from HMM import unsupervised_HMM
from helper import (
    parse_observations,
    generate_line,
)

# PREPROCESSING
text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
# TODO: extract words
# - keep hyphenated words hyphenated
# - some words could be tokenized as bigrams
# - separate punctuation from words, and store them separately
obs, obs_map = parse_observations(text)
syllables = cmudict.dict()
for punct in [".", ",", ":", ";", "!", "?"]:
    syllables.update({punct:[[]]})

# UNSUPERVISED LEARNING
hmm8 = unsupervised_HMM(obs, 1, 10)

# POETRY GENERATION, PART 1: HMMs
# TODO: write poem generation using hmm.generate_emission()
# - (suggested on piazza) in generate_emission() function, before generating 
#   the next word, go through all possibilities and check for (1) whether 
#   there are still enough syllables left for it and (2) whether it starts
#   with the right stress. if one of those is violated, manually set its 
#   probability to 0. renormalize and then pick a next-word.
# - keep in mind last words may have special syllable counts and also 
#   need to rhyme
print('Sample Sentence:\n====================')
generate_line(hmm8, syllables, obs_map)

