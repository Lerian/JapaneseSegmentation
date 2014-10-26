#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:Name:
    hmm_segmenter.py

:Authors:
    Florian Boudin (florian.boudin@univ-nantes.fr)

:Date:
    22 july 2013 (creation)

:Description:
    A HMM Japanese segmenter.
"""

import re
import sys
import xml.sax
import codecs
import bisect
import time


################################################################################
# Main function handling function calls
################################################################################
def main(argv):

    if len(argv) != 3:
        print "Usage : hmm_segmenter.py training_file test_file output_file"
        sys.exit()

    train_set1 = content_handler(argv[0])
    train_set2 = content_handler(argv[0])
    test_set = content_handler(argv[1])

    """model2 = train2(train_set.sentences)
    model3 = train3(train_set.sentences)
    model = model3.extend(model2)"""
    
    model2 = train2(train_set1.sentences)
    model = train3(train_set2.sentences)
    model.append(model2[0])
    model.append(model2[1])
    model.append(model2[2])

    handle = codecs.open(argv[2], 'w', 'utf-8')
    handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
    handle.write('<dataset>\n')
    for i in range(len(test_set.raw_sentences)):
        segmented_sentence = word_segmentation(model, test_set.raw_sentences[i])
        handle.write('\t<sentence sid="'+str(i)+'">\n')
        handle.write('\t\t<raw>'+segmented_sentence[:]+'</raw>\n')
        handle.write('\t</sentence>\n')
    handle.write('</dataset>')
    handle.close()

################################################################################


################################################################################
# Useful functions
################################################################################
def add_one(dic, value):
    if not dic.has_key(value):
        dic[value] = 0.0
    dic[value] += 1.0

def hiragana(char):
    if ord(char) >= 0x3040 and ord(char) <= 0x309F:
        return True
    else:
        return False

def katakana(char):
    if ord(char) >= 0x30A0 and ord(char) <= 0x30FF:
        return True
    else:
        return False

def romaji(char):
    if ord(char) >= 0xFF01 and ord(char) <= 0xFFEF:
        return True
    else:
        return False

def get_alphabet(char):
    if katakana(char):
        return 'katakana'
    elif hiragana(char):
        return 'hiragana'
    elif romaji(char):
        return 'romaji'
    else:
        return 'kanji'

################################################################################


################################################################################
# XML Sax Parser for KNBC xml file
################################################################################
class content_handler(xml.sax.ContentHandler):
    #-T-----------------------------------------------------------------------T-
    def __init__(self, path):
        # Tree for pushing/poping tags, attrs
        self.tree = []
        
        # Buffer for xml element
        self.buffer = ''

        # Container for the sentence list
        self.sentences = []
        self.current_sentence = []
        self.raw_sentences = []

        # Construct and launch the parser
        parser = xml.sax.make_parser()
        parser.setContentHandler(self)
        parser.parse(path)
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def startElement(self, name, attrs):
        self.tree.append((name, attrs))
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def characters(self, data):
        self.buffer += data.strip()
    #-B-----------------------------------------------------------------------B-
        
    #-T-----------------------------------------------------------------------T-
    def endElement(self, name):
        tag, attrs = self.tree.pop()

        if tag == 'token':
            self.current_sentence.append(self.buffer)

        elif tag == 'sentence':
            self.sentences.append(self.current_sentence)
            self.current_sentence = []

        elif tag == 'raw':
            self.raw_sentences.append(self.buffer)

        # Flush the buffer
        self.buffer = ''
    #-B-----------------------------------------------------------------------B-
################################################################################

################################################################################
# Training function with 2-grams
################################################################################
def train2(sentences):

    observation_prob = {}
    state_trans_prob = {}
    alphabet_trans_prob = {'hiragana':{ \
                                'katakana':{'c':0,'b':0}, \
                                'romaji':{'c':0,'b':0}, \
                                'kanji':{'c':0,'b':0}, \
                                'hiragana':{'c':0,'b':0}}, \
                            'katakana':{ \
                                'katakana':{'c':0,'b':0}, \
                                'romaji':{'c':0,'b':0}, \
                                'kanji':{'c':0,'b':0}, \
                                'hiragana':{'c':0,'b':0}}, \
                            'romaji':{ \
                                'katakana':{'c':0,'b':0}, \
                                'romaji':{'c':0,'b':0}, \
                                'kanji':{'c':0,'b':0}, \
                                'hiragana':{'c':0,'b':0}}, \
                            'kanji':{ \
                                'katakana':{'c':0,'b':0}, \
                                'romaji':{'c':0,'b':0}, \
                                'kanji':{'c':0,'b':0}, \
                                'hiragana':{'c':0,'b':0}}}

    # Iterates through the sentences
    for sentence in sentences:

        for i in range(len(sentence)):
            sentence[i] = 'c'.join(sentence[i])
        annotated_sequence = 'b'.join(sentence)

        previous_state = ''
        current_state = ''

        for i in range(0, len(annotated_sequence)-1, 2):
            observation = annotated_sequence[i:i+3]
            bigram = observation[0]+observation[2]
            current_state = observation[1]

            alphabet_trans_prob[get_alphabet(observation[0])][get_alphabet(observation[2])][current_state] += 1

            if not observation_prob.has_key(bigram):
                observation_prob[bigram] = {'c': 0.0, 'b': 0.0}
            observation_prob[bigram][current_state] += 1.0

            if previous_state != '':
                add_one(state_trans_prob, (previous_state, current_state))
            previous_state = current_state

    # Normalize observation probabilities
    for bigram in observation_prob:
        norm_factor = observation_prob[bigram]['c'] \
                      + observation_prob[bigram]['b']
        observation_prob[bigram]['c'] /= norm_factor
        observation_prob[bigram]['b'] /= norm_factor

    # Normalize transition probabilities
    norm_factor = 0.0
    for t in state_trans_prob:
        norm_factor += state_trans_prob[t]
    for t in state_trans_prob:
        state_trans_prob[t] /= norm_factor
    
    # Normalize alphabet transition probabilities
    norm_factor = 0.0
    for i in alphabet_trans_prob.keys():
        for j in alphabet_trans_prob[i].keys():
            for k in alphabet_trans_prob[i][j].keys():
                norm_factor += alphabet_trans_prob[i][j][k]
    for i in alphabet_trans_prob.keys():
        for j in alphabet_trans_prob[i].keys():
            for k in alphabet_trans_prob[i][j].keys():
                alphabet_trans_prob[i][j][k] /= norm_factor
    
    return [observation_prob, state_trans_prob, alphabet_trans_prob]
################################################################################

################################################################################
# Training function with 3-grams
################################################################################
def train3(sentences):

    observation_prob = {}
    state_trans_prob = {}

    # Iterates through the sentences
    for sentence in sentences:

        for i in range(len(sentence)):
            sentence[i] = 'c'.join(sentence[i])
        annotated_sequence = "Wc" + ('b'.join(sentence))

        previous_state = ''
        current_state = ''

        for i in range(0, len(annotated_sequence)-3, 2):
            observation = annotated_sequence[i:i+5]
            trigram = observation[0]+observation[2]+observation[4]
            current_state = observation[3]

            if not observation_prob.has_key(trigram):
                observation_prob[trigram] = {'c': 0.0, 'b': 0.0}
            #print trigram
            observation_prob[trigram][current_state] += 1.0

            if previous_state != '':
                add_one(state_trans_prob, (previous_state, current_state))
            previous_state = current_state

    # Normalize observation probabilities
    for trigram in observation_prob:
        norm_factor = observation_prob[trigram]['c'] \
                      + observation_prob[trigram]['b']
        observation_prob[trigram]['c'] /= norm_factor
        observation_prob[trigram]['b'] /= norm_factor

    # Normalize transition probabilities
    norm_factor = 0.0
    for t in state_trans_prob:
        norm_factor += state_trans_prob[t]
    for t in state_trans_prob:
        state_trans_prob[t] /= norm_factor
    
    return [observation_prob, state_trans_prob]
################################################################################

def printTest(sentence, arg):
    
    if len(sentence) == 9:
        #print arg
        return

################################################################################
# Function for word segmentation
################################################################################
def word_segmentation(model, sentence):

    observation_prob3 = model[0]
    state_trans_prob3 = model[1]
    
    observation_prob2 = model[2]
    state_trans_prob2 = model[3]
    alphabet_trans_prob = model[4]

    unseen_prob_c = 0.01
    unseen_prob_b = 0.02

    observations = []
    lattice = []
    
    isTrigram = []

    """print "--- sentence ---"
    print sentence
    print "--- modif sentence ---" """
    sentence = "W"+sentence
    for i in range(len(sentence)-2):
        trigram = sentence[i:i+3]
        observations.append(trigram)

        if observation_prob3.has_key(trigram):
            lattice.append(observation_prob3[trigram])
            lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
            lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])
            isTrigram.append(True)
        else:
            if observation_prob2.has_key(trigram[1:]):
                lattice.append(observation_prob2[trigram[1:]])
                lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
                lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])
                isTrigram.append(False)
            else:
                lattice.append(alphabet_trans_prob[get_alphabet(trigram[1])][get_alphabet(trigram[2])])
                lattice[-1]['c'] = max(unseen_prob_c, lattice[-1]['c'])
                lattice[-1]['b'] = max(unseen_prob_b, lattice[-1]['b'])
                isTrigram.append(False)

    # Use viterbi to decode the lattice

    # Initialisation
    V = []
    bisect.insort(V, (lattice[0]['c'], ['c']))
    bisect.insort(V, (lattice[0]['b'], ['b']))

    # Looping the observations
    for i in range(1, len(observations)):

        # Select the n-best paths
        best_V = []
        best_V.append(V[-1])
        highest_prob = best_V[0][0]
        for j in range(len(V)-2,-1,-1):
            if V[j][0] >= highest_prob:
                best_V.append(V[j])
            else:
                break

        new_V = []

        for prob, stack in best_V:
            """
                Gérer les 2 matrices state_trans_prob!!!!!!
            """
            # Compute probability for continuation
            if isTrigram[i]:
                prob_c = prob * lattice[i]['c'] * state_trans_prob3[(stack[-1], 'c')]
                bisect.insort(new_V, (prob_c, stack+['c']))
                prob_b = prob * lattice[i]['b'] * state_trans_prob3[(stack[-1], 'b')]
                bisect.insort(new_V, (prob_b, stack+['b']))
            else:
                prob_c = prob * lattice[i]['c'] * state_trans_prob2[(stack[-1], 'c')]
                bisect.insort(new_V, (prob_c, stack+['c']))
                prob_b = prob * lattice[i]['b'] * state_trans_prob2[(stack[-1], 'b')]
                bisect.insort(new_V, (prob_b, stack+['b']))

        V = list(new_V)

    best_V = V[-1][1]

    # Sentence segmentation 
    segmented_sentence = []
    for i in range(len(sentence)-2):
        if best_V[i] == 'c':
            segmented_sentence.append(sentence[i+1])
        else:
            segmented_sentence.append(sentence[i+1]+" ")
    segmented_sentence.append(sentence[-1])
    return ''.join(segmented_sentence)


################################################################################

################################################################################
# To launch the script 
################################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
################################################################################
