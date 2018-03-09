########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from HMM import unsupervised_HMM

import os
import numpy as np


####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

####################
# HMM FUNCTIONS
####################

#pre-processing
def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]

    dig_ = 0
    obs_counter = 0
    obs = []
    obs_map = {}
    for line in lines:
        obs_elem = []
        sigh = 1
        for word in line:
            if word.isdigit():
                dig_ = 1
                continue

            #store puncuation as "words" in obs_map
            puncuation = [':', ';', '.', ',', '!', '?']
            end = 0
            for punc in puncuation:
                if word[-1] == punc:
                    end, endp = 1, punc
                    break
            #keep hypens and apostrophes
            word = re.sub(r'[^\w\-\']', '', word).lower()

            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])

            if (end == 1 and endp not in obs_map):
                obs_map[endp] = obs_counter
                obs_counter += 1
                obs_elem.append(obs_map[endp])

        if (dig_ != 1):
            # Add the encoded sequence.
            obs.append(obs_elem)
        else:
            dig_ = 0

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_line(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return ' '.join(sentence).capitalize() + '...'

def generate_haiku(hmm, syllables, obs_map):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)
    num_syllables = [5, 7, 5]
    start_state = -1

    for i in range(len(num_syllables)):
        emission, start_state = hmm.generate_line(
            syllables, obs_map_r, num_syllables[i], start_state=start_state)
        sentence = [obs_map_r[j] for j in emission]
        print(' '.join(sentence).capitalize())

def main():
    text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
    obs, obs_map = parse_observations(text)
    #hmm8 = unsupervised_HMM(obs, 5, 20)
    # print('Sample Sentence:\n====================')
    # print(sample_sentence(hmm8, obs_map, n_words=25))

main()