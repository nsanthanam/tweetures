#Import the necessary libraries
import pandas as pd
import nltk
from collections import Counter
import numpy as np
import re
import sys
import os

tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def preprocess_data(filename):
    """
    This function takes in a filename and opens the file in read mode. Using the format provided in the sample CSV
    file, it removes extraneous characters and generates a list of lists where each list is a single row of the 
    input file.
    
    This list of lists is then converted into a Pandas DataFrame with the tweet id as the index.
    
    Input: fn (string)
    Output: tweets (Pandas DataFrame)
    """
    lines = []
    try:
        with open(filename, 'r') as fd:
            for line in fd:
                split_line = line.split(';') #columns are separated by semi-colons
                
                #remove extraneous characters from the line
                split_line2 = [item.replace('"', "") for item in split_line] 
                split_line2 = [item.replace('\r', "") for item in split_line2]
                split_line2 = [item.replace('\n', "") for item in split_line2]
                split_line2 = [re.sub(pattern=',+$', repl='', string=item) for item in split_line2] #remove all commas at end of line
                
                #tweets containing a comma are split into multiple strings in the list - here we concatenate them into one string
                if len(split_line2) > 3:
                    for item in range(3, len(split_line2)):
                        split_line2[2] = split_line2[2] + split_line2[item]
                    split_line2 = split_line2[0:3]
                lines.append(split_line2) #append each line to list

        #Convert list into dataframe
        tweets = pd.DataFrame(lines, columns=['id', 'polarity', 'tweet'])
        tweets.drop(labels=0, axis=0, inplace=True) #first list row is the header, dropping it
        tweets['id'] = tweets['id'].astype(int)
        tweets.set_index(keys='id', inplace=True) #the tweet id is being made the index
        tweets['polarity'] = tweets['polarity'].astype(int)

        return tweets
    except IOError, e:
        print 'File not found'

def generate_features(DF):
    """
    Generates features based on the tweets DataFrame. Calculates how many mentions, hashtags, and words in a tweet. 
    Also counts the different parts of speech each tweet contains.
    
    Input: tweets DataFrame
    Output: features DataFrame
    """
    #basic features such as the number of '@'-mentions, hashtags, and number of words in the tweet
    DF['mentions'] = DF['tweet'].apply(lambda s: len(re.findall(pattern= '@[a-z]+', string=s)))
    DF['hashtags'] = DF['tweet'].apply(lambda s: len(re.findall(pattern= '#[a-z]+', string=s)))
    tokenized = tweets['tweet'].apply(nltk.word_tokenize) #tokenise tweets into list of words
    DF['wordcount'] = tokenized.apply(len) #number of words in tweet
    
    #categorise tokens into relevant part of speech and store in DataFrame
    pos_list = tokenized.apply(lambda t: map(lambda x: x[1], nltk.pos_tag(t)))
    pos_dict = pos_list.apply(lambda x: Counter(x)) #Count how many of each type there are
    
    #convert the list of dicts into a dataframe
    pos_df = pd.DataFrame(columns=[], index=pos_dict.index.values, data=0)
    for i in pos_dict.index.values:
        for j in pos_dict[i].keys():
            pos_df.loc[i, j] = pos_dict[i][j]
    pos_df.replace(to_replace=np.nan, value=0, inplace=True) #replace any NaNs with 0 (meaning that part of speech wasn't in the tweet)
    pos_df = pos_df.astype(int) #convert everything to integer

    #merge the DataFrame with basic features and the parts of speech counts
    features = pd.merge(DF, pos_df, left_index=True, right_index=True)
    features.drop(labels=['tweet'], axis=1, inplace=True) #drop the original tweet text
    
    return features
def write_output(DF, filename):
    try:
        input_file_path = os.path.dirname(os.path.abspath(filename))
        output_filename = input_file_path + '/' + 'tweet_features.csv'
        DF.to_csv(output_filename)
    except IOError, e:
        print 'Unable to write'

fn = sys.argv[1] #read in the filename from the command line
tweets = preprocess_data(filename=fn)
features = generate_features(tweets)
write_output(features, fn)

