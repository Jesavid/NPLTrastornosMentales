import os
import pandas
import pandas as pd
import spacy
from dotenv import load_dotenv

# Ajustar las opciones de visualización
pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho de la visualización
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de las columnas

# Cargar variables de entorno
load_dotenv()
PATH_SUBJECTSTRAIN = os.getenv('PATH_SUBJECTSTRAIN')
PATH_TRAIN = os.getenv('PATH_TRAIN')

#nlp = spacy.load('es_core_news_sm')

# Función para leer los archivos JSON y crear un DataFrame con el nombre del archivo y
# los datos asociados
def readJSONFiles():
    # Crear diccionarios para los nombres de archivos y la información que contienen
    names_files = []
    files_reads = []

    # Obtener el nombre del archivo para cada archivo en la ruta TRAIN
    for name_file in os.listdir(PATH_SUBJECTSTRAIN):
        # leer JSON
        file = pd.read_json(f"{PATH_SUBJECTSTRAIN}/{name_file}")
        # Guardar nombre de archivo
        names_files.append(name_file)
        # Guardar datos asociados
        files_reads.append(file)

    # Crear dataframe de los datos
    input = pd.DataFrame({'name_file': names_files, 'file': files_reads})

    #print(input)
    #premensaje = input[input['name_file'] == 'subject334.json']
    #mensaje = premensaje.iloc[0]['file']['message']
    #print(mensaje)

    return input

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
    linkedFile = pd.DataFrame

    for index, fila in trainLabel.iterrows():
       # print(subject)
        #print(index)
        #print(fila)
        specdata = input[input['name_file'] == f'{fila['Subject']}.json']

        #print(specdata)
        label = trainLabel.loc[trainLabel['Subject'] != 'Subject', 'label']
        linkedFile = pd.concat([specdata,label], ignore_index=True)


    print(linkedFile)

    # premensaje = input[input['name_file'] == 'subject334.json']
    #TODO porque no imprimie todos los subject
    #TODO crear un archivo para ver cómo está generando los datos
linkJSONLabel()