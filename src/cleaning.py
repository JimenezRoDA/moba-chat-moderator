import pandas as pd 

def inspect_dataframe(df, name="DataFrame"):
    """
    Realiza una inspección básica de un DataFrame de pandas, mostrando:
    - Las primeras 5 filas
    - Información general (info(), incluyendo tipos de datos y valores no nulos)
    - Estadísticas descriptivas (describe())
    - Conteo de valores únicos para columnas con pocos valores únicos.
    """
    print(f"\n--- Inspección de: {name} ---")
    print("\n>>> Primeras 5 filas:")
    print(df.head())

    print(f"\n>>> Últimas 5 filas:")
    print(df.tail())

    print("\n>>> Información general (info()):")
    df.info()

    print("\n>>> Estadísticas descriptivas (describe()):")
    print(df.describe(include='all')) # include='all' para ver también no numéricas

    print("\n>>> Conteo de valores únicos (Value Counts) para columnas clave:")
    for column in df.columns:
        if df[column].nunique() < 50 and df[column].dtype == 'object': # Para columnas categóricas o con pocas opciones
            print(f"\nColumna '{column}':")
            print(df[column].value_counts(dropna=False)) # dropna=False para ver también NaN

    print(f"\n--- Fin de la inspección de: {name} ---")