# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:08:52 2020

@author: panay
"""
'''

df = pd.read_fwf('./pervjustice.txt')
df.to_csv('log.csv')
'''

'''# First time running only
#download data 
import nltk
nltk.download()
'''

import nltk
import csv

#query a sentence to remove stopwords
def clean(query,stopwords):
    # tokenize sentence into words
    querywords = nltk.tokenize.word_tokenize(query)
    # make words lowercase, remove stopwords
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    return(result)



# load data
filename = './pervjustice.txt'
file = open(filename, 'rt')

#open a scv file to write,newline='' used to not leave blank lines in between written rows
with open('./datasets/multiple chats.csv', 'w',newline='') as out_file:
    writer = csv.writer(out_file)
    #Titles
    writer.writerow(('Username','Time','Message'))
    
    for aline in file:
   
        sentences=clean(aline,[':'])
        #replace anything to split using a single separetor '##' 
        sentences=sentences.replace('(','##').replace(')','##').replace('[','##').replace(']','##').split('##')
        #write each line in the csv file
        writer.writerow(sentences)

