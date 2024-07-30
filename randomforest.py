from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import vectorizacion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree


def randomForest():
    # Obtener vectorizacion de los datos
    xTrainDF, xTrialDF, trainCorpus, trialCorpus = vectorizacion.bagOfWords()

    # Conjunto de datos entrenamiento prueba. 30% de los datos para prueba
    xTrain, xTrial, yTrain, yTrial = train_test_split(xTrainDF, trainCorpus['label'], test_size=.30, random_state=3)

    # Establecer RandomForest
    randomF = RandomForestClassifier(n_estimators=100,
                                     criterion='gini',
                                     max_features='sqrt',
                                     bootstrap=True,
                                     max_samples=3/4,
                                     oob_score=True,
                                     random_state=3)

    randomF.fit(xTrain, yTrain)

    # Mostrar arboles
    # print(classification_report())
    # print(f'F1: {f1_score(df, label['label'])}')
    # print(randomF.predict(trail))

    # Mostrar resultados
    print(f'Score: {randomF.score(xTrainDF, trainCorpus['label'])}')
    print(f'Obb Score: {randomF.oob_score_*100}')
    print(f'Predict: {randomF.predict(xTrialDF)}')

    print('Evaluación usando trial')
    # Crear matriz de confusion
    matrizConfusion = confusion_matrix(trialCorpus['label'], randomF.predict(xTrialDF))
    sns.heatmap(matrizConfusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Reporte de clasificacion
    reporte = classification_report(trialCorpus['label'], randomF.predict(xTrialDF))
    print(reporte)

    precision = precision_score(trialCorpus['label'], randomF.predict(xTrialDF), average='macro')
    print(f"La precisión es: {precision}")

    print('Evaluación con los datos train')
    reporte = classification_report(yTrial, randomF.predict(xTrial))
    print(reporte)

    # Crear matriz de confusion
    matrizConfusion = confusion_matrix(yTrial, randomF.predict(xTrial))
    sns.heatmap(matrizConfusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Validacion Cruzada

    acurracy = cross_val_score(randomF, xTrain, yTrain, cv=10, scoring="accuracy")

    # Obtener promedio
    promedioAccuracy = acurracy.mean()

    print(f"Accuracies por fold: {acurracy}")
    print(f'Promedio accuracy: {promedioAccuracy}')

if __name__ == "__main__":
    randomForest()