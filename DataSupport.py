
import pandas as pd 
import numpy as np
import re # Regular expressions
from nltk.corpus import stopwords
#from DataPreprocessing import *
import nltk
from nltk.corpus import stopwords
import re 
from nltk.stem.wordnet import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
stopWords = stopwords.words("english")
stopWords.extend(["I'm","i'm","y'all" ])

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

stopWords = stopwords.words("english")
stopWords.extend(["I'm","i'm","y'all" ])

# Twitter Data Extraction from the training data file.
trainDataTwitter = pd.read_csv("Twitter_Data.csv",encoding='latin - 1')
#print(trainDataTwitter.head(5))

#print(" Categorical Values and Counts : ")
#print(trainDataTwitter.category.value_counts()) 

# Reddit Data Extraction from Training data File
trainDataReddit = pd.read_csv("Reddit_Data.csv",encoding = 'latin - 1')
#print(trainDataReddit.head(5))

#print(trainDataReddit.category.value_counts())

# combining DataSets 
trainData = pd.concat([trainDataTwitter,trainDataReddit])
print(type(trainData))
print("Whole Dataset made ")
#print(trainData.category.value_counts())

#print(trainData.shape)
# (200229, 2)
#print(trainData.isna().sum())
# null values found are 
# cleanText = 104 
# category = 7

# Data Pre Processing
trainData = trainData.dropna()
trainData = trainData.reset_index(drop = True)
trainData = trainData.astype({'category':int})
#print(trainData.isna().sum())
# since the null rouws are very low compared to the train data , drop them.
print("Whole Dataset made without null values ")
#print(trainData.shape)
#(200118, 2)

#print(trainData.describe())

#print(trainData.category)

trainData.clean_text  = trainData.clean_text.apply(lambda val: ' '.join([word for word in val.split() if word not in (stopWords)]))
print("Whole Dataset Stop Words Removed ")
# Data Cleaning operation 
# removing referances, Links, special chars , User mentions 
# with the help of regular expressions 

# @[A-Za-z0-9]+ for user mentions 
# [^A-Za-z0-9']+ remove unnecessary special characters in the word splitted.
# \w+:\/\/\S+  for hyperlinks


def clean(x):
    x=' '.join(re.sub("(@[A-Za-z0-9]+)|([^A-Za-z0-9']+)|(\w+:\/\/\S+)"," ",x).split())
    return x.lower()


trainData.clean_text = trainData.clean_text.apply(clean)
print(" Data Cleaning Done ")
# Lemmatization 
#print(trainData.clean_text)
trainData.clean_text = trainData.clean_text.apply(lambda val: ' '.join([lemmatizer.lemmatize(word,'v') for word in val.split()])) 
print(" Lemmatization ")
# Testing Data  Creation 
x = trainData.iloc[:,:-1]
y = trainData.iloc[:,-1]
text_Train, text_Test, yTrain,yTest = train_test_split(x,y, test_size = 0.3)
print(" Test Train Split made ")
#print(text_Test)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

vectorizedTrain = tfidf.fit_transform(text_Train.clean_text)
vectorizedTest = tfidf.transform(text_Test.clean_text)

'''
tester = pd.Series(["happy"])
print(tester)
testVector = tfidf.transform(tester)
print("test Vector ----------------------------------")
print(testVector)
print(testVector[0])
'''
print(" TFIDF Vectorization done ")
# SMOTE algorithm use for improve dataset 
# to achieve equality in dataset labels 

'''
# importing the SMOTE tool
from imblearn.combine import SMOTETomek

# creating an object
smk = SMOTETomek(random_state=42)

# generating the input and output training data after applying the sythentic generation 
X_train,y_train=smk.fit_sample(new_inp,tweets.label)
'''

'''
# Confirmation of Train Data and Test Data Splis

print(text_Test.shape, yTest.shape)
print(text_Train.shape, yTrain.shape)

print(yTest.head(5), yTrain.head(5))

'''
#print(vectorizedTrain[0],vectorizedTest.shape)
#print(vectorizedTrain)
#print(yTrain.shape, yTest.shape)

# Logistic Regression Model Creation

logisticRegressor = LogisticRegression()

logisticRegressor.fit(vectorizedTrain,yTrain)

trainPredictedLogistic = logisticRegressor.predict(vectorizedTrain)
print("Logistic Regressor on Train")
print(classification_report(yTrain,trainPredictedLogistic,digits = 4))

testPredictedLogistic = logisticRegressor.predict(vectorizedTest)
print("Logistic Regressor on Test")
print(classification_report(yTest,testPredictedLogistic,digits = 4))

# Naive Bayes Classsifier Creation
'''
#Multinomial NB

multiNB = MultinomialNB()
multiNB.fit(vectorizedTrain,yTrain)
trainPredictedMulti = multiNB.predict(vectorizedTrain)
print("Mutinomial Naive Bayes on Train")
print(classification_report(yTrain,trainPredictedMulti,digits = 4))

testPredictedMulti = multiNB.predict(vectorizedTest)
print("Mutinomial Naive Bayes on Test")
print(classification_report(yTest,testPredictedMulti,digits = 4))

# Decision Tree Classifier

decisionTree = DecisionTreeClassifier(criterion='entropy')
decisionTree.fit(vectorizedTrain,yTrain)
print(" Decision Tree on Train")
trainPredictedDTree = decisionTree.predict(vectorizedTrain)
print(classification_report(yTrain,trainPredictedDTree,digits=4))

testPredictedDTree = decisionTree.predict(vectorizedTest)
print("Decision Tree on Test")
print(classification_report(yTest,testPredictedDTree,digits=4))

# SVM classifier
# SVM Linear
linearsvm = svm.SVC(kernel = 'linear')
linearsvm.fit(vectorizedTrain,yTrain)
print("LInear svm on Train")
trainPredictedlsvm = linearsvm.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedlsvm ,digits=4))
print("Linear svm on Test")
testPredictedlsvm = linearsvm.predict(vectorizedTest)
print(classification_report( yTest,testPredictedlsvm ,digits=4))

# SVM Polynomial
polysvm = svm.SVC(kernel = 'poly')
polysvm.fit(vectorizedTrain,yTrain)
print("Ploynomial SVM on Train")
trainPredictedpolysvm = polysvm.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedpolysvm ,digits=4))
print("Polynomial SVM on Test")
testPredictedpolysvm = polysvm.predict(vectorizedTest)
print(classification_report( yTest,testPredictedpolysvm ,digits=4))

# SVM Sigmoid

sigmoidsvm = svm.SVC(kernel = 'sigmoid')
sigmoidsvm.fit(vectorizedTrain,yTrain)
print("Sigmoid SVM on Train")
trainPredictedsigmoidsvm = sigmoidsvm.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedsigmoidsvm ,digits=4))
print("Sigmoid SVM on Test")
testPredictedsigmoidsvm = sigmoidsvm.predict(vectorizedTest)
print(classification_report( yTest,testPredictedsigmoidsvm ,digits=4))

# SVM RVB

rvbsvm = svm.SVC(kernel='rvb')
rvbsvm.fit(vectorizedTrain,yTrain)
print("RVB SVM on Train")
trainPredictedrvbsvm = rvbsvm.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedrvbsvm ,digits=4))
print("RVB SVM on Test")
testPredictedrvbsvm = linearsvm.predict(vectorizedTest)
print(classification_report( yTest,testPredictedrvbsvm ,digits=4))

# KNN 

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(vectorizedTrain,yTrain)
print("KNN on Train")
trainPredictedknn = knn.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedknn ,digits=4))
print("knn on Test")
testPredictedknn = knn.predict(vectorizedTest)
print(classification_report( yTest,testPredictedknn ,digits=4))

# Vote on all these classifiers

estimators=[('logisticR',logisticRegressor) ,('multinomialNB',multiNB),('decisionTree',decisionTree),
            ('Knn',knn),('lsvm',linearsvm),('ploysvm',polysvm),('rvbsvm',rvbsvm),
            ('sigmoidsvm',sigmoidsvm) ]

voteClassifier = VotingClassifier(estimators,voting='hard')

voteClassifier.fit(vectorizedTrain,yTrain)
print("Vote Classifier on Train")
trainPredictedvote = voteClassifier.predict(vectorizedTrain)
print(classification_report( yTrain,trainPredictedvote ,digits=4))
print("Vote Classifier on Test ")
testPredictedvote = voteClassifier.predict(vectorizedTest)
print(classification_report(yTest,testPredictedvote,digits = 4))

'''
print("Done Now Optimization")



