import pandas as pd

#SEEDS
X_SEED = 1

#TOPS
X_TOPS = 100

#PARAMETROS
x = X_SEED

#FUNCION PREDICTORA
def predecirPrecio( df ):
    return x*df['metroscubiertos']

#SETUP
train = pd.read_csv("train.csv", sep=",")

#TRAINING
train['prediccion'] = predecirPrecio(train)
train['errorCuadraticoMedioActual'] = (abs(train['precio'] - train['prediccion']))^2
mejorPredictor = x
for x in range(X_SEED, X_TOPS):
    train['prediccion'] = predecirPrecio(df)
    train['errorCuadraticoMedioNuevo'] = (abs(train['precio'] - train['prediccion']))^2
    print('Parametro X = ', x, 'con error total = ', train['errorCuadraticoMedioActual'])
    if (train['errorCuadraticoMedioActual'].sum() > train['errorCuadraticoMedioNuevo'].sum()):
        mejorPredictor = x
        train['errorCuadraticoMedioActual'] = train['errorCuadraticoMedioNuevo']
        print('Nuevo mejor predictor encontrado')
