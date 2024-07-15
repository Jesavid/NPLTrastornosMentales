import os
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import numpy
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))

# Ajustar las opciones de visualización
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho de la visualización
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de las columnas

# Cargar variables de entorno
load_dotenv()
PATH_SUBJECTSTRAIN = os.getenv('PATH_SUBJECTSTRAIN')
PATH_TRAIN = os.getenv('PATH_TRAIN')
PATH_CSVLINKEDFILES = os.getenv('PATH_CSVLINKEDFILES')

# Declarar de dict para guardar los subjects preprocesados
dataArray = {
    'Subject': [],
    'id_message': [],
    'message': [],
    'preproccedMessage': [],
    'date': []
}

# Función para leer los archivos JSON y crear un DataFrame con el nombre del archivo y
# los datos asociados
def readJSONFiles():
    # Crear diccionarios para los nombres de archivos y la información que contienen
    names_files = []
    files_reads = []

    # Obtener el nombre del archivo para cada archivo en la ruta TRAIN
    i = 0
    for name_file in os.listdir(PATH_SUBJECTSTRAIN):
        # leer JSON
        file = pd.read_json(f"{PATH_SUBJECTSTRAIN}/{name_file}")
        preprocesstext(file, name_file)
        # Guardar nombre de archivo
        #names_files.append(name_file)
        # Guardar datos asociados
        #files_reads.append(file)

    # Crear dataframe de los datos
    #input = pd.DataFrame({'name_file': names_files, 'file': files_reads})

    #print(input)
    #premensaje = input[input['name_file'] == 'subject334.json']
    #mensaje = premensaje.iloc[0]['file']['message']
    #print(mensaje)
    #return input

# Función para leer las etiquetas
def readTXT():
    trainLabel = pd.read_csv(PATH_TRAIN)
    #print(trainLabel[trainLabel['label']==1])
    #print(trainLabel)
    return trainLabel

# Unir las etiquetas con los datos
def linkJSONLabel():
    input = readJSONFiles()
    trainLabel = readTXT()
    data = []
    labels = []

    for index, fila in trainLabel.iterrows():
        # Obtener la data de input para cada subject en fila
        tempData = input[input['name_file'] == f'{fila['Subject']}.json']
        data.append(tempData)
        labels.append(fila['label'])

    # Unir los datos con la etiqueta correspondiente
    linkedFiles = pd.DataFrame({'data': data, 'label':labels})
    #print(linkedFile.iloc[0])
    linkedFiles.to_csv(f'{PATH_CSVLINKEDFILES}result.csv')
    return linkedFiles

#Preprocesar texto
def preprocesstext(file, name_file):
    i = 0
    #print(f"Preprocessado del {name_file}")

    # Agregar la columna subject
    file.insert(0, 'Subject', name_file)
    # Agregar la columna para el texto preprocesado
    file.insert(3, 'preproccedMessage', " ")

    for content in file['message']:
        # Tokenizar y convertir a minuscula
        tempMessage = word_tokenize(content.lower())
        # Eliminar stop words
        tempMessage = [tempMessage for tempMessage in tempMessage if tempMessage.lower() not in stop_words]
        # Eliminar signos de puntuacion
        tempMessage = [tempMessage for tempMessage in tempMessage if tempMessage not in string.punctuation]
        # Regresar tempMessage a una oracion
        tempMessage = ' '.join(tempMessage)
        # Agregar en la columna preproccedMessage el texto preprocesado
        file.loc[i, 'preproccedMessage'] = tempMessage
        i = i + 1

    # Agregar al dict dataArray los subjects
    dataArray['Subject'].append(name_file.replace('.json', ''))
    dataArray['id_message'].append(file['id_message'])
    dataArray['message'].append(file['message'])
    dataArray['preproccedMessage'].append(file['preproccedMessage'])
    dataArray['date'] = file['date']

        # TODO agregar signos que no están es string.punctuation
        # TODO unir mensaje limpio con name_file
        # TODO escribir en archivo, registrar

readJSONFiles()
print(dataArray['preproccedMessage'])
finalFile = pd.DataFrame(dataArray.items())
finalFile.to_json(f'{PATH_CSVLINKEDFILES}final.json')