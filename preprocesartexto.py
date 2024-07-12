import os
from operator import index

import pandas
import pandas as pd
import spacy
from dotenv import load_dotenv

#Cargar variables de entorno
load_dotenv()

PATH_SUBJECTSTRAIN = os.getenv('PATH_SUBJECTSTRAIN')

#nlp = spacy.load('es_core_news_sm')


#a = pd.read_json(f"{PATH_SUBJECTSTRAIN}/subject334.json")
#print(a.message)
#print(os.listdir(PATH_SUBJECTSTRAIN))




def readJSONFiles():
    # Crear diccionarios para los nombres de archivos y la información que contienen
    names_files = []
    files_reads = []

    for name_file in os.listdir(PATH_SUBJECTSTRAIN):
        # print(type(name_file))
        file = pd.read_json(f"{PATH_SUBJECTSTRAIN}/{name_file}")
        names_files.append(name_file)
        files_reads.append(file)
        # print(file)
        # partialinput = pd.DataFrame({index=[0],file})
        # i = 0

        # input = pd.DataFrame(index=[0],partialinput.head)
        # i = i + 1

    input = pd.DataFrame({'name_file': names_files, 'file': files_reads})

    premensaje = input[input['name_file'] == 'subject334.json']
    mensaje = premensaje.iloc[0]['file']['message']

    print(mensaje)

