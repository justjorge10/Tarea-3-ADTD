# Diego Sanjur 8-1024-2362
# Jorge Valderrama 8-1023-157
# Carlos Reyes 8-849-624
# Aaron Burac 8-1049-1605

"""
Backend Tarea 3 - Tratamiento de datos faltantes con imputacion KNN.
Dataset: Forest Fires (Cortez y Morais, 2007).

Aqui vive toda la logica de datos. El dashboard (frontend.py) solo consume
estas funciones, para mantener separado el back del front.
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Carpeta donde vive este archivo (para encontrar el CSV sin importar
# desde donde se ejecute el programa)
_BASE = os.path.dirname(os.path.abspath(__file__))

# Columnas numericas sobre las que se corre la imputacion
COLUMNAS_NUMERICAS = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']


def cargar_datos(path_csv='forestfires.csv'):
    """Lee el CSV. Si la ruta es relativa, la busca junto a este archivo."""
    if not os.path.isabs(path_csv):
        path_csv = os.path.join(_BASE, path_csv)
    return pd.read_csv(path_csv)


def introducir_nulos(df, columna='temp', proporcion=0.10, semilla=42):
    """
    Mete valores faltantes de forma artificial en una columna (por defecto temp),
    para luego poder demostrar la imputacion. Devuelve el df modificado y la
    mascara booleana de que filas quedaron nulas.
    """
    df = df.copy()
    np.random.seed(semilla)
    mascara = np.random.rand(len(df)) < proporcion
    df.loc[mascara, columna] = np.nan
    return df, mascara


def imputar_knn(df, columnas=None, n_vecinos=5):
    """Rellena los nulos de las columnas numericas usando KNNImputer."""
    if columnas is None:
        columnas = COLUMNAS_NUMERICAS
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_vecinos)
    df[columnas] = imputer.fit_transform(df[columnas])
    return df


def preparar_datos(path_csv='forestfires.csv', n_vecinos=5):
    """
    Flujo completo: carga -> introduce ~10% de nulos en temp -> imputa con KNN.
    Equivalente al script original de la tarea.
    """
    df = cargar_datos(path_csv)
    df_nulos, _ = introducir_nulos(df)
    df_limpio = imputar_knn(df_nulos, n_vecinos=n_vecinos)
    return df_limpio


if __name__ == "__main__":
    datos = preparar_datos()
    print(datos.head())
    print(datos.info())
    datos.to_csv(os.path.join(_BASE, 'forestfires_limpio.csv'), index=False)
    print("Archivo exportado: forestfires_limpio.csv")