# The Data set is extracted from csv files and are used further in the project 
import pandas as pd 
import numpy as np

# Extracting Twitter tweets Dataset
trainDataTwitter = pd.read_csv("Twitter_Data.csv",encoding='latin - 1')
# Extracing Reddit Commens Dataset
trainDataReddit = pd.read_csv("Reddit_Data.csv",encoding = 'latin - 1')

# combining DataSets 
trainData = pd.concat([trainDataTwitter,trainDataReddit])

#print(trainData.shape)
# (200229, 2)

#print(trainData.isna().sum())
# null values found are 
# cleanText = 104 
# category = 7

# The dataset is very huge. aound 200,000 texts are available 
# the nulls found are 111 
# since the trade is not very low, we can remove the null vales

# Preprocessing - removing null values
trainData = trainData.dropna()
# resetting the indices such that the dropped row's indices are re used 
trainData = trainData.reset_index(drop = True)
# the data type used by the category column is 'float64' 
# But the range of our categorical values is [-1,0,1]
# float is very huge for this. 
# So changing to integer. 
trainData = trainData.astype({'category':int})

#print(trainData.shape)
#(200118, 2)

labels = trainData.iloc[:,-1]
trainData = trainData.iloc[:,:-1]



