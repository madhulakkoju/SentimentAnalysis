import pickle 
import NLProcessing
import pandas as pd
from sklearn.linear_model import LogisticRegression


logisticRegressor = LogisticRegression()

logisticRegressor.fit(NLProcessing.getProcessedData(),NLProcessing.getTrainLabels())

with open('datafile.txt','wb') as fh:
    pickle.dump(logisticRegressor,fh)

'''
pickle_off = open ("datafile.txt", "rb")
logisticRegressor = pickle.load(pickle_off)


def predict(inptext):
    doc = NLProcessing.preprocessText(pd.Series([inptext]))
    #print("pre processed one ",doc)
    #print(doc.shape)
    op = logisticRegressor.predict(doc)
    #print(op)
    op = op[0]
    if(op==-1):
        #print("Negative")
        return "Negative"
    elif(op == 0):
        #print("Neutral")
        return "Neutral"
    else:
        #print("Positive")
        return "positive"

'''