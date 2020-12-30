
def preprocessText( inputText ):
    # remove stop words
    inputText[0] = ' '.join([word for word in inputText[0].split() if word not in stopWords ]) 
    # Lemmatize the text
    inputText = inputText.apply(lambda text:" ".join([lemmatizer.lemmatize(word,'v') for word in text.split()]))
    # Vectorizing the text data
    inputText = tfidf.transform(inputText)
    return inputText
    pass