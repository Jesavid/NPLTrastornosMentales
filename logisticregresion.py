from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
    xTrain, xTrial, yTrain, yTrial = train_test_split (xTrainDF, trainCorpus['label'], test_size=.30, random_state=3)

    # Crear modelo
    logisticReg = LogisticRegression(random_state=3).fit(xTrain, yTrain)

    # Mostrar resultados

    # Eliminar formato de notacion cientifica
    np.set_printoptions(suppress=True)

    # Probar modelo
    # print(logisticReg.predict(xTrainDF))
    print('Evaluación con los datos trial')
    # Reporte de clasificacion
    reporte = classification_report(trialCorpus['label'], logisticReg.predict(xTrialDF))
    print(reporte)

    # Crear matriz de confusion
    matrizConfusion = confusion_matrix(trialCorpus['label'], logisticReg.predict(xTrialDF))
    sns.heatmap(matrizConfusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # plt.savefig()
    print('Evaluación con los datos train')
    reporte = classification_report(yTrial, logisticReg.predict(xTrial))
    print(reporte)

    # Crear matriz de confusion
    matrizConfusion = confusion_matrix(yTrial, logisticReg.predict(xTrial))
    sns.heatmap(matrizConfusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Validacion Cruzada

    acurracy = cross_val_score(logisticReg, xTrain, yTrain, cv=10, scoring="accuracy")

    # Obtener promedio
    promedioAccuracy =acurracy.mean()

    print(f"Accuracies por fold: {acurracy}")
    print(f'Promedio accuracy: {promedioAccuracy}')

if __name__ == "__main__":
    logisticRegresion()

