from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

PATH_FINALFILE = os.getenv('PATH_FINALFILE')

corpus = []
# Funcion bolsa de palabras
def bagOfWords():
    vectorizer = CountVectorizer()
    # Leer JSON del corpus subject, message, label
    partialCorpus = pd.read_json(f'{PATH_FINALFILE}traincorpus.json', dtype=object)
    # print(partialCorpus.index)
    for index in partialCorpus.index:
        vector = vectorizer.fit_transform(partialCorpus.loc[index,["message"]])
        print(vectorizer.get_feature_names_out())
        print(vector.toarray())
        break
        # print(index)

bagOfWords()