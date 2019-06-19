import random
import re
import pandas as pd
import numpy as np
import string
from string import digits
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)
''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines = open('movie_lines2.txt', "r")
    lines = lines.readlines()
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open('movie_conversations.txt').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs, id2line,path):
    q_conv = open(path+'input.txt', 'w')
    a_conv = open(path+'output.txt', 'w')
    remove_punct= str.maketrans('','',string.punctuation)
    remove_digits = str.maketrans('','',digits)
    for conv in convs:
        a = ""
        i=1
        for line_id in conv:
            if i>1:
                itemp = str(a).lower()
                otemp = str(id2line[line_id]).lower()
                itemp = re.sub("'", '', itemp)
                itemp = re.sub(",", ' <COMMA>', itemp)
                itemp = itemp.translate(remove_punct)
                itemp = itemp.translate(remove_digits)
                otemp = re.sub("'", '', otemp)
                otemp = re.sub(",", ' <COMMA>', otemp)
                otemp = otemp.translate(remove_punct)
                otemp = otemp.translate(remove_digits)
                itemp = lemmatize_sentence(itemp)
                otemp = lemmatize_sentence(otemp)
                itemp = '<START> ' + itemp + ' <END>' + '\n'
                otemp = '<START> ' + otemp + ' <END>' + '\n'
                q_conv.writelines(itemp)
                a_conv.writelines(otemp)
            i = i+1
            a = id2line[line_id]
    q_conv.close()
    a_conv.close()

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


'''
    We need 4 files
    1. train.enc : Encoder input for training
    2. train.dec : Decoder input for training
    3. test.enc  : Encoder input for testing
    4. test.dec  : Decoder input for testing
'''
def prepare_seq2seq_files(questions, answers, path='',TESTSET_SIZE = 30000):
    
    # open files
    train_enc = open(path + 'train.enc','w')
    train_dec = open(path + 'train.dec','w')
    test_enc  = open(path + 'test.enc', 'w')
    test_dec  = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))],TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )


    # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
            

####
# main()
####

id2line = get_id2line()

convs = get_conversations()

extract_conversations(convs,id2line,"/Users/mukul/Desktop/")

questions, answers = gather_dataset(convs,id2line)

#print '>> gathered questions and answers.\n'
#prepare_seq2seq_files(questions,answers)
