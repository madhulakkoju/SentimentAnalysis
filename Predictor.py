import pickle
pickle_off = open ("datafile.txt", "rb")
logisticRegressor = pickle.load(pickle_off)



def predict(doc):
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