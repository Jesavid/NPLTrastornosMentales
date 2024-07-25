from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from dotenv import load_dotenv

# Ajustar las opciones de visualización en terminal
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho de la visualización
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de las columnas

# Cargar variables de entorno
load_dotenv()
PATH_FINALFILE = os.getenv('PATH_FINALFILE')

# Funcion bolsa de palabras
def bagOfWords():
    vectorizer = CountVectorizer()
    # Leer JSON del corpus subject, message, label
    trainCorpus = pd.read_json(f'{PATH_FINALFILE}trainCorpus.json', dtype=object)
    trialCorpus = pd.read_json(f'{PATH_FINALFILE}trialCorpus.json', dtype=object)


    # Vectoriacion global
    xTrain = vectorizer.fit_transform(trainCorpus['message'])
    xTrial = vectorizer.transform(trialCorpus['message'])

    # Convertir en dataframe y guardar CSV de la vectorizacion
    xTrainDF = pd.DataFrame(xTrain.toarray(), columns=vectorizer.get_feature_names_out())
    xTrialDF = pd.DataFrame(xTrial.toarray(), columns=vectorizer.get_feature_names_out())

    # Guardar BoW como CSV
    xTrainDF.to_csv(f'{PATH_FINALFILE}xTrainDF.csv')
    xTrialDF.to_csv(f'{PATH_FINALFILE}xTrialDF.csv')

    return xTrainDF, xTrialDF, trainCorpus, trialCorpus

bagOfWords()