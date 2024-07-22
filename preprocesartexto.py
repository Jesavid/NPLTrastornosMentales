import os
import string
# import fontstyle
# import tkinter
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go

nltk.download('punkt')
nltk.download('stopwords')

stopWords = set(stopwords.words('spanish'))

# Ajustar las opciones de visualización en terminal
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
    'date': [],
    'label': []
}

# Declarar array para contar las palabras
wordCount = {}

# Declarar array para ver las palabras y su subject asociado
wordStrange = {}

# Declarar array de stop words no contenidas en nltk
stop_words = stopWords.union(
    [
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando", "de",
        "del", "desde",
        "donde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "éramos", "eran",
        "eras", "eres",
        "es", "esa", "esas", "ese", "esos", "esta", "estaba", "estabais", "estábamos", "estaban", "estabas", "estad",
        "estada",
        "estadas", "estado", "estados", "estamos", "estando", "estar", "estaremos", "estará", "estarán", "estarás",
        "estaré",
        "estaréis", "estaría", "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto",
        "estos",
        "estoy", "estuve", "estuviera", "estuvierais", "estuviéramos", "estuvieran", "estuvieras", "estuvieron",
        "estuviese",
        "estuvieseis", "estuviésemos", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis",
        "estuviéramos",
        "estuviésemos", "estuvo", "ex", "excepto", "fue", "fuera", "fuerais", "fuéramos", "fueran", "fueras", "fueron",
        "fuese",
        "fueseis", "fuésemos", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis", "ha", "había", "habíais",
        "habíamos",
        "habían", "habías", "habéis", "habida", "habidas", "habido", "habidos", "habiendo", "habrá", "habrán", "habrás",
        "habré",
        "habréis", "habremos", "habría", "habríais", "habríamos", "habrían", "habrías", "habéis", "había", "habíais",
        "habíamos",
        "habían", "habías", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo", "haces", "hacia", "haciendo",
        "hago",
        "han", "has", "hasta", "hay", "haya", "hayamos", "hayan", "hayas", "he", "hecho", "hemos", "hube", "hubiera",
        "hubierais",
        "hubiéramos", "hubieran", "hubieras", "hubieron", "hubiese", "hubieseis", "hubiésemos", "hubiesen", "hubieses",
        "hubimos",
        "hubiste", "hubisteis", "hubiéramos", "hubiésemos", "hubo", "la", "las", "le", "les", "lo", "los", "me", "mi",
        "mis",
        "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "nada", "ni", "ningún", "ninguna",
        "ningunas",
        "ninguno", "ningunos", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros",
        "nunca",
        "os", "otra", "otras", "otro", "otros", "para", "parecer", "pero", "poca", "pocas", "poco", "pocos", "podéis",
        "podemos",
        "poder", "podría", "podríamos", "podrían", "podrías", "poner", "por", "porque", "primero", "puede", "pueden",
        "puedo",
        "pues", "que", "qué", "querer", "quién", "quienes", "quiere", "quiénes", "quiso", "saber", "se", "sé", "ser",
        "si",
        "sí", "siendo", "sin", "sino", "so", "sobre", "sois", "solamente", "solo", "sólo", "somos", "son", "soy", "su",
        "sus", "suya", "suyas", "suyo", "suyos", "sí", "también", "tanto", "te", "tendré", "tendréis", "tendremos",
        "tendría",
        "tendríais", "tendríamos", "tendrían", "tendrías", "tened", "tenemos", "tener", "tenga", "tengamos", "tengan",
        "tengas", "tengo", "tenía", "teníais", "teníamos", "tenían", "tenías", "ti", "tiene", "tienen", "tienes",
        "todo",
        "todos", "tu", "tus", "tuve", "tuviera", "tuvierais", "tuviéramos", "tuvieran", "tuvieras", "tuvieron",
        "tuviese",
        "tuvieseis", "tuviésemos", "tuviesen", "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuviéramos",
        "tuviésemos",
        "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "un", "una", "uno", "unos", "usa", "usamos", "usan", "usar", "usas",
        "uso", "usted", "ustedes", "va", "vais", "valor", "vamos", "van", "varias", "varios", "vaya", "veces", "ver",
        "verdad", "verdadera", "verdadero", "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros",
        "y", "ya", "yo", "él", "éramos", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última",
        "últimas",
        "último", "últimos"
    ]
)

# Declarar array de signos de puntuacion
puntuacion = ['___', '✦', '...', '“', '«', '✗', '¿', '»', '⁣', '``', '°', '━']

# Declarar diccionario de emojis
emojis = {
    "🙋‍♀": "mujer con la mano levantada",
    "🏃🏾‍♀": "mujer corriendo: tono de piel oscuro medio",
    "🤷‍♀": "mujer encogida de hombros",
    "🫡": "hola",
    "🏋‍♂": "persona levantando pesas signo masculino",
    "🇴": "la letra o mayuscula cirilo",
    "💆‍♀": "mujer recibiendo masaje",
    "🙆‍♀": "mujer haciendo el gesto de de acuerdo️",
    "2⃣": "teclas 2",
    "🤦🏻‍♀": "mujer con la mano en la frente tono de piel claro",
    "🤦‍♀": "mujer con la mano en la frente",
    "🫠": "estoy derretido de felicidad",
    "🇨": "la letra c",
    "🙋🏻‍♀": "mujer con la mano levantada tono de piel claro",
    "🙋🏽‍♀": "mujer con la mano levantada tono de piel medio",
    "🤸🏻‍♀": "mujer haciendo voltereta lateral tono de piel claro",
    "🧚‍♀": "hada hembra",
    "👨‍⚕": "profesional sanitario hombre",
    "1⃣": "teclas 1",
    "🤦‍♂": "hombre con la mano en la frente",
    "🤦🏼‍♀": "mujer con la mano en la frente tono de piel claro medio",
    "🤷🏻‍♀": "mujer encogida de hombros tono de piel claro",
    "🙋🏼‍♀": "mujer con la mano levantada tono de piel claro medio",
    "3⃣": "teclas 3"
}

# Declarar diccionario para corregir las palabras con una fuente diferente
fontsWords = {
    # Primer set
    "ᴀ": 'A',
    "ᴅ": 'D',
    "ᴇ": 'E',
    "ғ": 'F',
    "ɪ": 'I',
    "ᴍ": 'M',
    "ɴ": 'N',
    "ᴏ": 'O',
    "ʀ": 'R',
    "s": 'S',
    "ᴛ": 'T',
    "ᴠ": 'V',
    "ʏ": 'Y',
    # Segundo set
    "𝐴": 'A',
    "𝑎": 'a',
    "𝑎̀": 'a',
    "𝑏": 'b',
    "𝐶": 'C',
    "𝑐": 'c',
    "𝐷": 'D',
    "𝑑": 'd',
    "𝐸": 'E',
    "𝑒": 'e',
    "𝑓": 'f',
    "𝑔": 'g',
    "𝐻": 'H',
    "ℎ": 'h',
    "𝑖": 'i',
    "𝑗": 'j',
    "𝑘": 'k',
    "𝐿": 'L',
    "𝑙": 'l',
    "𝑀": 'M',
    "𝑚": 'm',
    "𝑁": 'N',
    "𝑛": 'n',
    "𝑜": 'o',
    "𝑜́": 'o',
    "𝑃": 'P',
    "𝑝": 'p',
    "𝑄": 'Q',
    "𝑞": 'q',
    "𝑅": 'R',
    "𝑟": 'r',
    "𝑆": 'S',
    "𝑠": 's',
    "𝑇": 'T',
    "𝑡": 't',
    "𝑈": 'U',
    "𝑢": 'u',
    "𝑣": 'v',
    "𝑥": 'x',
    "𝑦": 'y',
    "𝑧": 'z',
    # Tercer set
    "𝐀": 'A',
    "𝐁": 'B',
    "𝐄": 'E',
    "𝐇": 'H',
    "𝐈": 'I',
    "𝐋": 'L',
    "𝐌": 'M',
    "𝐍": 'N',
    "𝐎": 'O',
    "𝐏": 'P',
    "𝐑": 'R',
    "𝐒": 'S',
    "𝐘": 'Y',

    "": 'A',
    "": 'a',
    "": 'B',
    "": 'b',
    "": 'C',
    "": 'c',
    "": 'D',
    "": 'd',
    "": 'E',
    "": 'e',
    "": 'F',
    "": 'f',
    "": 'G',
    "": 'g',
    "": 'H',
    "": 'h',
    "": 'I',
    "": 'i',
    "": 'J',
    "": 'j',
    "": 'K',
    "": 'k',
    "": 'L',
    "": 'l',
    "": 'M',
    "": 'm',
    "": 'N',
    "": 'n',
    "": 'Ñ',
    "": 'ñ',
    "": 'O',
    "": 'o',
    "": 'P',
    "": 'p',
    "": 'Q',
    "": 'q',
    "": 'R',
    "": 'r',
    "": 'S',
    "": 's',
    "": 'T',
    "": 't',
    "": 'U',
    "": 'u',
    "": 'V',
    "": 'v',
    "": 'W',
    "": 'w',
    "": 'X',
    "": 'x',
    "": 'Y',
    "": 'y',
    "": 'Z',
    "": 'z',
}

def readJSONFiles():
    # Obtener el nombre del archivo para cada archivo en la ruta TRAIN
    for name_file in os.listdir(PATH_SUBJECTSTRAIN):
        # leer JSON
        file = pd.read_json(f"{PATH_SUBJECTSTRAIN}/{name_file}")
        preprocesstext(file, name_file)


# Función para leer los archivos JSON y crear un DataFrame con el nombre del archivo y
# los datos asociados

# Función para leer las etiquetas
def readTXT():
    trainLabel = pd.read_csv(PATH_TRAIN)
    return trainLabel

# Preprocesar texto
def preprocesstext(file, name_file):
    trainLabel = readTXT()
    i = 0

    # Agregar la columna subject
    file.insert(0, 'Subject', name_file)
    # Agregar la columna para el texto preprocesado
    file.insert(3, 'preproccedMessage', " ")

    # Obtener el label para un subject especifico
    specLabel = trainLabel[trainLabel['Subject'] == name_file.replace('.json', '')]
    # Agregar la columna label e insertar el valor correspondiente
    file.insert(5, 'label', specLabel['label'].values[0])

    for content in file['message']:
        # Convertir las palabras en la fuente general
        tempMessage = ''.join(fontsWords[caracter] if caracter in fontsWords else caracter for caracter in content)

        # Cambiar emoji por texto
        tempMessage = ''.join(emojis[caracter] if caracter in emojis else caracter for caracter in tempMessage)

        # Eliminar signos de puntuacion extraños, cambiar letra acentuada
        tempMessage = unicodedata.normalize('NFKD', tempMessage).encode('ASCII', 'ignore').decode('utf-8')
        tempMessage = tempMessage.replace('.', ' ').replace('-', ' ')

        # Tokenizar y convertir a minuscula
        tempMessage = word_tokenize(tempMessage.lower())

        # Eliminar stop words
        tempMessage = [tempMessage for tempMessage in tempMessage if tempMessage.lower() not in stop_words]

        # Eliminar signos de puntuacion
        tempMessage = [tempMessage for tempMessage in tempMessage if tempMessage not in string.punctuation]
        tempMessage = [tempMessage for tempMessage in tempMessage if tempMessage not in puntuacion]

        for word in tempMessage:
            if word in wordCount:
                wordCount[word] = wordCount[word] + 1
                wordStrange[word] = name_file
            else:
                wordCount[word] = 1
                wordStrange[word] = name_file

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
    dataArray['date'].append(file['date'])
    dataArray['label'].append(file['label'])

    # TODO agregar signos que no están es string.punctuation


readJSONFiles()

# Convertir a Data Frame y guardar como JSON
finalFile = pd.DataFrame(dataArray.items())
finalFile.to_json(f'{PATH_CSVLINKEDFILES}ab.json')

# Ordenar de mayor a menor las palabras
wordCount = dict(sorted(wordCount.items(), key=lambda item: item[1], reverse=True))
word = list(wordCount.keys())
count = list(wordCount.values())

# Crear una tabla de las palabras tokenizadas y su cantidad
wordCountTab = go.Figure(data=[go.Table(header=dict(values=['Word', 'Count']),
                                        cells=dict(values=[word, count])
                                        )])
# Mostrar tabla
wordCountTab.show()

# Crear Data Frame y guardar CSV de las palabras y su cantidad
df = pd.DataFrame(wordCount.items(), columns=['Palabra', 'Cantidad'])
df.to_csv(f'{PATH_CSVLINKEDFILES}new.csv')
