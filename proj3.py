import os
import numpy as np

from nltk.corpus import cmudict

from HMM import unsupervised_HMM
from helper import *

# PREPROCESSING
# text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
text = open(os.path.join(os.getcwd(), 'data/allpoems.txt')).read()
# visualization of whole data set
wordcloud = text_to_wordcloud(text, title='Shakespeare')
# TODO: extract words
# - keep hyphenated words hyphenated
# - some words could be tokenized as bigrams
# - separate punctuation from words, and store them separately
obs, obs_map = parse_observations(text)
syllables = cmudict.dict()
for punct in [".", ",", ":", ";", "!", "?"]:
    syllables.update({punct:[[]]})

# UNSUPERVISED LEARNING
#Was 20
hmm8 = unsupervised_HMM(obs, 10, 100)

# visualizations of sparsity of A, O as well as
# visualizations of states as wordclouds
visualize_sparsities(hmm8, O_max_cols=50)
wordclouds = states_to_wordclouds(hmm8, obs_map)

#This part only works in Jupyter Notebook
anim = animate_emission(hmm8, obs_map, M=8)
HTML(anim.to_html5_video())

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
generate_quatrain(hmm8, syllables, obs_map)
print()
generate_quatrain(hmm8, syllables, obs_map)
print() 
generate_quatrain(hmm8, syllables, obs_map)
print()
generate_couplet(hmm8, syllables, obs_map)

