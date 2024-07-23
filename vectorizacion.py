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
    partialCorpus = pd.read_json(f'{PATH_FINALFILE}traincorpus.json', dtype=object)

    # Vectoriacion global
    vector = vectorizer.fit_transform(partialCorpus['message'])

    # Convertir en dataframe y guardar CSV de la vectorizacion
    df = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names_out())
    df.to_csv(f'{PATH_FINALFILE}vector.csv')

    return df

bagOfWords()