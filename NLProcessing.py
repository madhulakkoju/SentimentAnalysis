import re 
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import DataSetExtraction 

# Extracting stop words
stopWords = stopwords.words("english")
stopWords.extend(["I'm","i'm","y'all" ])
# Removing Stop words from the dataset's text column
DataSetExtraction.trainData.clean_text  = DataSetExtraction.trainData.clean_text.apply(lambda val: ' '.join([word for word in val.split() if word not in (stopWords)]))

# @[A-Za-z0-9]+ for user mentions 
# [^A-Za-z0-9']+ remove unnecessary special characters in the word splitted.
# \w+:\/\/\S+  for hyperlinks

def clean(x):
    x=' '.join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z0-9']+)|(\w+:\/\/\S+)"," ",x).split())
    return x.lower()

# Cleaning Text 
DataSetExtraction.trainData.clean_text = DataSetExtraction.trainData.clean_text.apply(clean)
# Lemmatizing the Text.
# Modifying the words in verb form to their base form for processing
lemmatizer = WordNetLemmatizer()
# LLemmatizer object
DataSetExtraction.trainData.clean_text = DataSetExtraction.trainData.clean_text.apply(lambda val: ' '.join([lemmatizer.lemmatize(word,'v') for word in val.split()])) 

# The String data is not understandable by Machine Learning models. 
# the string data is to be modified into numbers which can be used by the models.
# so, Vectorizing the text. 

# Vetorizer Object
tfidf = TfidfVectorizer()

vectorizedData = tfidf.fit_transform(DataSetExtraction.trainData.clean_text)

import pickle 

with open('tfidfFile.txt','wb') as fh:
    pickle.dump(tfidf,fh)

def getProcessedData():
    return vectorizedData
def getTrainLabels():
    return DataSetExtraction.labels





