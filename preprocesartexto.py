import os
import pandas
import pandas as pd
import spacy
from dotenv import load_dotenv

#Cargar variables de entorno
load_dotenv()

PATH_SUBJECTSTRAIN = os.getenv('PATH_SUBJECTSTRAIN')

#nlp = spacy.load('es_core_news_sm')

a = pd.read_json(f"{PATH_SUBJECTSTRAIN}/subject2.json")

for name_file in os.listdir(PATH_SUBJECTSTRAIN):
    INPUT = pd.read_json(f"{PATH_SUBJECTSTRAIN}/subject2.json")