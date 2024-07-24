from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import vectorizacion
import pandas as pd
import os
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn import tree

# Cargar variables de entorno
load_dotenv()
PATH_FINALFILE = os.getenv('PATH_FINALFILE')

def randomForest():
    # Obtener vectorizacion de los datos
    xTrainDF, xTrialDF = vectorizacion.bagOfWords()

    # Obtener labels
    trainLabel = pd.read_json(f'{PATH_FINALFILE}trainCorpus.json')
    trialLabel = pd.read_json(f'{PATH_FINALFILE}trialCorpus.json')

    # Establecer RandomForest
    randomF = RandomForestClassifier(n_estimators=100,
                                     criterion='gini',
                                     max_features='sqrt',
                                     bootstrap=True,
                                     max_samples=2/3,
                                     oob_score=True)

    randomF.fit(xTrainDF, trainLabel['label'])
    print(f'Score: {randomF.score(xTrainDF, trainLabel['label'])}')
    print(f'Obb Score: {randomF.oob_score_*100}')
    print(f'Predict: {randomF.predict(xTrialDF)}')

    # Crear matriz de confusion
    matrizConfusion = confusion_matrix(trialLabel['label'], randomF.predict(xTrialDF))
    sns.heatmap(matrizConfusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Reporte de clasificacion
    reporte = classification_report(trialLabel['label'], randomF.predict(xTrialDF))
    print(reporte)

    # print(classification_report())
    # print(f'F1: {f1_score(df, label['label'])}')
    # print(randomF.predict(trail))

    # for forest in randomF.estimators_:
    #     tree.plot_tree(forest, feature_names=df.columns)
    #     plt.show()

randomForest()