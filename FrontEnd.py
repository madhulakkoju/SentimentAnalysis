import tkinter as tk
from tkinter import *
print("tkinter")
import tkinter.font as font  
import Predictor
import pandas as pd
from nltk.corpus import stopwords
print("nltk")
stopWords = stopwords.words("english")
stopWords.extend(["I'm","i'm","y'all" ])
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
print("lems")
pickle_off = open ("tfidfFile.txt", "rb")
tfidf = pickle.load(pickle_off)

# Input Text is sent to this method for preprocessing
# inputText is a series dataframe. 
def preprocessText( inputText ):
    # remove stop words
    inputText[0] = ' '.join([word for word in inputText[0].split() if word not in stopWords ]) 
    # Lemmatize the text
    inputText = inputText.apply(lambda text:" ".join([lemmatizer.lemmatize(word,'v') for word in text.split()]))
    # Vectorizing the text data
    inputText = tfidf.transform(inputText)
    return inputText
    
print("start tkinter")
master = Tk(screenName="HomePage")
#master.configure(bg='white')
master.geometry("500x250")
master.title(" Home ")
myFont = font.Font(size = 12)

Label(master, text='Enter Text',font = myFont) .grid(row = 0)

inputTxt = Text(master,height = 5,width=30,bg='light yellow')
inputTxt.grid(row=2,column=1,columnspan=5)

#outputTxt = Text(master,height = 3,width=25,bg='light yellow')
Label(master,text="Neutral",font = myFont).grid(row=5,column=3)

def getText():
    inpText = inputTxt.get("1.0","end")
    #print(inpText)
    if(inpText=="" or inpText==" " or inpText=="\n"):
        return 
    # prdict here
    Label(master,text=Predictor.predict(preprocessText(pd.Series([inpText]))),font = myFont).grid(row=5,column=3)
    pass

check = Button(master,text="check",command = lambda:getText() )
check.grid(row=10,column =0 )
def reset():
    inputTxt.delete("1.0",'end')
    Label(master,text="Neutral",font = myFont).grid(row=5,column=3)
Button(master,text=" reset ",command = lambda:reset() ).grid(row=10,column = 6)
    
mainloop()

