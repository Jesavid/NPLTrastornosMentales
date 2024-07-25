from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import vectorizacion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def logisticRegresion():
    # Obtener vectorizacion de los datos y corpus
    xTrainDF, xTrialDF, trainCorpus, trialCorpus = vectorizacion.bagOfWords()

    # Conjunto de datos entrenamiento prueba. 30% de los datos para prueba
    # xTrain, xTrial, yTrain, yTrial = train_test_split (xTrainDF, trainLabel['label'], test_size=.30)

    # Crear modelo
    logisticReg = LogisticRegression().fit(xTrainDF, trialCorpus['label'])

    # Mostrar resultados

    # Eliminar formato de notacion cientifica
    np.set_printoptions(suppress=True)

    # Probar modelo
    print(logisticReg.predict(xTrainDF))

    # Reporte de clasificacion
    reporte = classification_report(trialCorpus['label'], logisticReg.predict(xTrialDF))
    print(reporte)

logisticRegresion()

