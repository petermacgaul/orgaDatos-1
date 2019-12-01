from sklearn import metrics
import category_encoders as ce
import time

from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

import category_encoders as ce

import re
import string
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

stopwords = get_stop_words('spanish')

def remove_stopwords(text,stopwords):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stopwords] 
    filtered_sentence = []
    for w in word_tokens: 
        if w not in stopwords: 
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)

def clean_text_round(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[&;.*?¿\‘’“”…«»]\%;', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\x95', ' ', text)
    text = re.sub('acute', '', text)
    text = re.sub('tilde', '', text)
    text = re.sub(' p ', '', text)
    text = re.sub('nbsp', '', text)
    text = re.sub('á', 'a', text)
    text = re.sub('é', 'e', text)
    text = re.sub('í', 'i', text)
    text = re.sub('ó', 'o', text)
    text = re.sub('ú', 'u', text)
    return text

def limpiar_texto(text):
    return remove_stopwords(clean_text_round(text),stopwords)


def getDropCols():
    return ['titulo', 'descripcion', 'direccion'] # temporal


def getOneHotCols():
    return ['tipodepropiedad', 'provincia', "anio", "mes", "garages"]

def getBinaryCols():
    return ['ciudad', 'idzona']

def getSumCols():
    return ['banos','habitaciones']


def getAllCols():
    return ['ciudad', 'idzona', 'tipodepropiedad', 'provincia']  

def getTarget1Cols():
    return ['ciudad_target_precio', 'idzona_target_precio', 'tipodepropiedad_target_precio', 'provincia_target_precio','antiguedad_target_precio','banos_target_precio','habit_target_precio']  

def getTarget2Cols():
    return ['tipodepropiedad_target_antiguedad','idzona_target_antiguedad']  

def getTarget3Cols():
    return ['ciudad_target_mtot', 'idzona_target_mtot', 'tipodepropiedad_target_mtot', 'provincia_target_mtot']  

def getTarget4Cols():
    return ['ciudad_target_banos', 'idzona_target_banos', 'tipodepropiedad_target_banos', 'provincia_target_banos']  

def getTarget5Cols():
    return ['ciudad_target_habit', 'idzona_target_habit', 'tipodepropiedad_target_habit', 'provincia_target_habit']  

def getTarget6Cols():
    return ['ciudad_target_banoshabit', 'idzona_target_banoshabit', 'tipodepropiedad_target_banoshabit', 'provincia_target_banoshabit']  


def fill_m2(df):
    df['metrostotales'].fillna(df['metroscubiertos'], inplace=True)
    df['metroscubiertos'].fillna(df['metrostotales'], inplace=True)
    df["metroscubiertostotales"]=df["metroscubiertos"]+df["metrostotales"]
    df["m_cuadrado"]=df["metroscubiertos"]*df["metrostotales"]
    df["m_tot_log"] = df['metrostotales'].transform(lambda x: np.log(x))
    df["m_cub_log"] = df['metroscubiertos'].transform(lambda x: np.log(x))
    df["m2_log"]   = df['metroscubiertostotales'].transform(lambda x: np.log(x))
    df["m_cuad_log"]   = df['m_cuadrado'].transform(lambda x: np.log(x))
    df["m_tot_sqrt"] = df['metrostotales'].transform(lambda x: np.sqrt(x))
    df["m_cub_sqrt"] = df['metroscubiertos'].transform(lambda x: np.sqrt(x))
    df["m2_sqrt"]   = df['metroscubiertostotales'].transform(lambda x: np.sqrt(x))
    df["banosPorHabit"] = df["banos"]/df["habitaciones"]
    df["mtotPorHabit"] = df["metrostotales"]/df["habitaciones"]
    df["mtotPorBanos"] = df["metrostotales"]/df["banos"]
    df["mtotPorAmbiente"] = df["metrostotales"]/df["ambientes"]
    return df

def init_test(x):
    features = x.copy()
    
    features['fecha'] = pd.to_datetime(features['fecha'])
    
    features['anio'] = features['fecha'].dt.year
    features["mes"] = features['fecha'].dt.month
    features["dia"] = features['fecha'].dt.day
    
    features = features.drop(columns=["lat","lng"])
  
    features["descripcion"] = features["descripcion"].fillna("").transform(lambda x: limpiar_texto(x))
    features["titulo"] = features["titulo"].fillna("").transform(lambda x: limpiar_texto(x))
    features["direccion"] = features["direccion"].fillna("").transform(lambda x: limpiar_texto(x))
    features_palabras(features)
    return features
    
def preprocess(x, encode1, encode2, encodingType,
               encode3=None,encode4=None,encode5=None,encode6=None,encode7=None,encode8=None,encode9=None,
               y1 = None ):

    start_time = time.time()
   
    features = x.copy()
    
    drop_cols = getDropCols()
    features = features.drop(drop_cols, axis=1)

    features['fecha'] = pd.to_datetime(features['fecha'], errors='coerce').astype(int) / 10**9

    features = fill_m2(features)
        
    if (encodingType == 'train'):
        if encode3:
            encode3.set_params(cols=getTarget1Cols())
            features = encode3.fit_transform(features,y1)
            
        if encode4:
            y2 = features.antiguedad
            encode4.set_params(cols=getTarget2Cols())
            features = encode4.fit_transform(features,y2)
            
        if encode5:
            y3 = features.metrostotales
            encode5.set_params(cols=getTarget3Cols())
            features = encode5.fit_transform(features,y3)
            
        if encode6:
            y4 = features.banos
            encode6.set_params(cols=getTarget4Cols())
            features = encode6.fit_transform(features,y4)
        
        if encode8:
            y5 = features.habitaciones
            encode8.set_params(cols=getTarget5Cols())
            features = encode8.fit_transform(features,y5)
            
        if encode9:
            y6 = features.banosPorHabit
            encode9.set_params(cols=getTarget6Cols())
            features = encode9.fit_transform(features,y6)
            
        
        encode1.set_params(cols=getOneHotCols())
        encode2.set_params(cols=getBinaryCols())
        encode7.set_params(cols=getSumCols())
        
        features = encode1.fit_transform(features)
        features = encode2.fit_transform(features)
        features = encode7.fit_transform(features)
        
    else:
        if encode3:
            features = encode3.transform(features)
        if encode4:
            features = encode4.transform(features)
        if encode5:
            features = encode5.transform(features)
        if encode6:
            features = encode6.transform(features)
        if encode8:
            features = encode8.transform(features)
        if encode9:
            features = encode9.transform(features)
            
        features = encode1.transform(features)
        features = encode2.transform(features)
        features = encode7.transform(features)


    features_with_nans = features.columns[features.isna().any()].tolist()

    for feature in features_with_nans:
        features[feature] = features[feature].fillna(0)

    print("--- %s seconds ---" % (time.time() - start_time))
    return features



def predecir(model, train_features, train_labels, test_features, test_labels):
    predict = model.predict(test_features)
    error = mean_absolute_error(test_labels, predict)
    score = model.score(test_features,test_labels)

    print('Entrenamiento: {:0.4f}%'.format(model.score(train_features, train_labels)*100))
    print('Testeo: {:0.4f}%.'.format(score*100))
    print('Mean abs error: {:0.4f}.'.format(error))

def contiene(df, columna, palabra):
    return df[columna].str.contains(palabra).astype(int)

def contiene_alguna(df, columna, palabras):
    result = df[columna].apply(lambda x: 0)
    for palabra in palabras:
        result = result | contiene(df, columna, palabra)
    return result

def features_palabras(df):
    df["palabra_hermosa"] = contiene_alguna(df, "descripcion", ["hermosa", "bonita", "bonito", "linda", "cholula", "cholulo", "preciosa", "precioso"]) | contiene_alguna(df, "titulo", ["hermosa", "bonita", "bonito", "linda", "cholula", "cholulo", "precioso", "preciosa"])
    df["palabra_excelente"] = contiene_alguna(df, "descripcion", ["excelente", "excelentes"]) | contiene_alguna(df, "titulo", ["excelente", "excelentes"])
    df["palabra_mejor"] = contiene_alguna(df, "descripcion", ["mejor", "mejores"]) | contiene_alguna(df, "titulo", ["mejor", "mejores"])
    df["palabra_grande"] = contiene_alguna(df, "descripcion", ["grande", "gran", "amplia", "amplias", "amplio", "amplios"]) | contiene_alguna(df, "titulo", ["grande", "gran", "amplia", "amplias", "amplio", "amplios"])
    df["palabra_equipada"] = contiene_alguna(df, "descripcion", ["equipada", "equipado", "completa", "completo"]) | contiene_alguna(df, "descripcion", ["equipada", "equipado", "completa", "completo"])
    df["palabra_vestidor"] = contiene_alguna(df, "descripcion", ["vestidor", "closet"]) | contiene_alguna(df, "titulo", ["vestidor", "closet"])
    df["palabra_credito"] = contiene_alguna(df, "descripcion", ["credito", "crédito", "créditos", "creditos", "banco", "banca", "bancario", "bancarios", "hipoteca"]) | contiene_alguna(df, "titulo", ["credito", "crédito", "créditos", "creditos", "banco", "banca", "bancario", "bancarios", "hipoteca"])
    df["palabra_privada"] = contiene_alguna(df, "descripcion", ["privada", "privado"]) | contiene_alguna(df, "titulo", ["privada", "privado"])
    df["palabra_bodega"] = contiene_alguna(df, "descripcion", ["bodega"]) | contiene_alguna(df, "titulo", ["bodega"])
    df["palabra_club"] = contiene_alguna(df, "descripcion", ["club"]) | contiene_alguna(df, "titulo", ["club"])
    df["palabra_cerrada"] = contiene_alguna(df, "descripcion", ["cerrada", "cerrado"]) | contiene_alguna(df, "titulo", ["cerrada", "cerrado"])
    df["palabra_jardin"] = contiene_alguna(df, "descripcion", ["jardin", "jardín", "garden", "patio"]) | contiene_alguna(df, "titulo", ["jardin", "jardín", "garden", "patio"])
    df["palabra_oportunidad"] = contiene_alguna(df, "descripcion", ["oportunidad"]) | contiene_alguna(df, "titulo", ["oportunidad"])
    df["palabra_tv"] = contiene_alguna(df, "descripcion", ["tv", "tele", "television", "televisión", "televisor"]) | contiene_alguna(df, "titulo", ["tv", "tele", "television", "televisión", "televisor"])
    df["palabra_juegos"] = contiene_alguna(df, "descripcion", ["juego"]) | contiene_alguna(df, "titulo", ["juego"])
    df["palabra_niño"] = contiene_alguna(df, "descripcion", ["niño", "niña", "infantil"]) | contiene_alguna(df, "titulo", ["niño", "niña", "infantil"])
    df["palabra_transporte"] = contiene_alguna(df, "descripcion", ["transporte"]) | contiene_alguna(df, "titulo", ["transporte"])
    df["palabra_estudio"] = contiene_alguna(df, "descripcion", ["estudio"]) | contiene_alguna(df, "titulo", ["estudio"])
    df["palabra_terraza"] = contiene_alguna(df, "descripcion", ["terraza"]) | contiene_alguna(df, "titulo", ["terraza"])
    df["palabra_balcon"] = contiene_alguna(df, "descripcion", ["balcón", "balcon"]) | contiene_alguna(df, "titulo", ["balcón", "balcon"])
    df["palabra_lote"] = contiene_alguna(df, "descripcion", ["lote", "terreno"]) | contiene_alguna(df, "titulo", ["lote", "terreno"])
    df["palabra_fraccionamiento"] = contiene_alguna(df, "descripcion", ["fraccionamiento", "fracc"]) | contiene_alguna(df, "titulo", ["fraccionamiento", "fracc"])
    df["palabra_local"] = contiene_alguna(df, "descripcion", ["local", "tienda", "comercial"]) | contiene_alguna(df, "titulo", ["local", "tienda", "comercial"])
    df["palabra_seguridad"] = contiene_alguna(df, "descripcion", ["vigilancia", "vigilador", "seguridad", "guardia"]) | contiene_alguna(df, "titulo", ["vigilancia", "vigilador", "seguridad", "guardia"])
    df["palabra_garage"] = contiene_alguna(df, "descripcion", ["garage", "auto", "estacionamiento"]) | contiene_alguna(df, "titulo", ["garage", "auto", "estacionamiento"])
    df["palabra_centro"] = contiene_alguna(df, "descripcion", ["centro", "central", "cercano", "cercania", "minuto"]) | contiene_alguna(df, "titulo", ["centro", "central", "cercano", "cercania", "minuto"])
    df["palabra_techada"] = contiene_alguna(df, "descripcion", ["techada", "techado", "roof"]) | contiene_alguna(df, "titulo", ["techada", "techado", "roof"])
    df["palabra_estancia"] = contiene_alguna(df, "descripcion", ["estancia"]) | contiene_alguna(df, "titulo", ["estancia"])
    df["palabra_alberca"] = contiene_alguna(df, "descripcion", ["alberca"]) | contiene_alguna(df, "titulo", ["alberca"])
    df["palabra_servicios"] = contiene_alguna(df, "descripcion", ["servicios"]) | contiene_alguna(df, "titulo", ["servicios"])
    df["palabra_servicio"] = contiene_alguna(df, "descripcion", ["servicio"]) | contiene_alguna(df, "titulo", ["servicio"])
    df["palabra_estilo"] = contiene_alguna(df, "descripcion", ["estilo"]) | contiene_alguna(df, "titulo", ["estilo"])
    df["palabra_frente"] = contiene_alguna(df, "descripcion", ["frente"]) | contiene_alguna(df, "titulo", ["frente"])
    df["palabra_vista"] = contiene_alguna(df, "descripcion", ["vista"]) | contiene_alguna(df, "titulo", ["vista"])
    df["palabra_visitas"] = contiene_alguna(df, "descripcion", ["visita"]) | contiene_alguna(df, "titulo", ["visita"])
    df["palabra_parque"] = contiene_alguna(df, "descripcion", ["parque", "plaza", "verde"]) | contiene_alguna(df, "titulo", ["parque", "plaza", "verde"])
    df["palabra_areas"] = contiene_alguna(df, "descripcion", ["area", "área"]) | contiene_alguna(df, "titulo", ["area", "área"])
    df["palabra_estrenar"] = contiene_alguna(df, "descripcion", ["estrenar", "estreno", "estrene"]) | contiene_alguna(df, "titulo", ["estrenar", "estreno", "estrene"])
    df["palabra_infonavit"] = contiene_alguna(df, "descripcion", ["infonavit"]) | contiene_alguna(df, "titulo", ["infonavit"])
    df["palabra_residencial"] = contiene_alguna(df, "descripcion", ["residencia"]) | contiene_alguna(df, "titulo", ["residencia"])
    df["palabra_escuela"] = contiene_alguna(df, "descripcion", ["escuela", "colegio", "educacion", "educación", "uni", "universidad", "facultad"]) | contiene_alguna(df, "titulo", ["escuela", "colegio", "educacion", "educación", "uni", "universidad", "facultad"])
    df["palabra_exclusivo"] = contiene_alguna(df, "descripcion", ["exclusivo", "exclusividad"]) | contiene_alguna(df, "titulo", ["exclusivo", "exclusividad"])
    df["palabra_lujo"] = contiene_alguna(df, "descripcion", ["lujo"]) | contiene_alguna(df, "titulo", ["lujo"])
    df["palabra_esquina"] = contiene_alguna(df, "descripcion", ["esquina"]) | contiene_alguna(df, "titulo", ["esquina"])
    df["palabra_refaccion"] = contiene_alguna(df, "descripcion", ["refaccion", "refacción", "reacondicionado", "remodelada", "remodelado"]) | contiene_alguna(df, "titulo", ["refaccion", "refacción", "reacondicionado", "remodelada", "remodelado"])
    df["palabra_country"] = contiene_alguna(df, "descripcion", ["country"]) | contiene_alguna(df, "titulo", ["country"])
    df["palabra_barra"] = contiene_alguna(df, "descripcion", ["barra"]) | contiene_alguna(df, "titulo", ["barra"])
    df["palabra_lavado"] = contiene_alguna(df, "descripcion", ["lavado"]) | contiene_alguna(df, "titulo", ["lavado"])
    df["palabra_renta"] = contiene_alguna(df, "descripcion", ["renta", "alquiler", "alquilar"]) | contiene_alguna(df, "titulo", ["renta", "alquiler", "alquilar"])
    df["palabra_super"] = contiene_alguna(df, "descripcion", ["super"]) | contiene_alguna(df, "titulo", ["super"])
    df["palabra_lago"] = contiene_alguna(df, "descripcion", ["lago"]) | contiene_alguna(df, "titulo", ["lago"])
    df["palabra_bosque"] = contiene_alguna(df, "descripcion", ["bosque", "arbol", "árbol"]) | contiene_alguna(df, "titulo", ["bosque", "arbol", "árbol"])
    df["palabra_avenida"] = contiene_alguna(df, "descripcion", ["av", "avenida"]) | contiene_alguna(df, "titulo", ["av", "avenida"])
    df["palabra_hospital"] = contiene_alguna(df, "descripcion", ["hospital", "medicina", "medico", "médico", "farmacia"]) | contiene_alguna(df, "titulo", ["hospital", "medicina", "medico", "médico", "farmacia"])
    df["palabra_pileta"] = contiene_alguna(df, "descripcion", ["pileta", "piscina", "jacuzzi"]) | contiene_alguna(df, "titulo", ["pileta", "piscina", "jacuzzi"])
    df["palabra_solarium"] = contiene_alguna(df, "descripcion", ["solarium"]) | contiene_alguna(df, "titulo", ["solarium"])
    df["palabra_gas"] = contiene_alguna(df, "descripcion", ["gas", "estufa"]) | contiene_alguna(df, "titulo", ["gas", "estufa"])
    df["direc_privada"] = contiene_alguna(df, "direccion", ["privada","priv","privado"]) 
    df["direc_av"] = contiene_alguna(df, "direccion", ["av", "avenida"]) 
    df["direc_camino"] = contiene_alguna(df, "direccion", ["playa"]) 
    df["direc_playa"] = contiene_alguna(df, "direccion", ["camino"]) 