from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import vectorizacion
import pandas as pd
import os
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

# Cargar variables de entorno
load_dotenv()
PATH_FINALFILE = os.getenv('PATH_FINALFILE')

def logisticRegresion():
    # Obtener vectorizacion de los datos
    xTrainDF, xTrialDF = vectorizacion.bagOfWords()

    # Obtener labels
    trainLabel = pd.read_json(f'{PATH_FINALFILE}trainCorpus.json')
    trialLabel = pd.read_json(f'{PATH_FINALFILE}trialCorpus.json')

    # Conjunto de datos entrenamiento prueba. 30% de los datos para prueba
    # xTrain, xTrial, yTrain, yTrial = train_test_split (xTrainDF, trainLabel['label'], test_size=.30)

    # Crear modelo
    logisticReg = LogisticRegression().fit(xTrainDF, trainLabel['label'])

    # Mostrar resultados

    # Eliminar formato de notacion cientifica
    np.set_printoptions(suppress=True)

    # Probar modelo
    print(logisticReg.predict(xTrainDF))

    # Reporte de clasificacion
    reporte = classification_report(trialLabel['label'], logisticReg.predict(xTrialDF))
    print(reporte)

logisticRegresion()

