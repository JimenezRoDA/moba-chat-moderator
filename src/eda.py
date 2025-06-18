import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import re
from wordcloud import WordCloud, STOPWORDS

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.append(os.path.abspath('.'))
from utils import PALETA_ATARDECER_GRIETA

sys.path.append(os.path.abspath('.'))
from cleaning import clean_text

def display_total_comments(dataframe):
    """
    Muestra el número total de comentarios en el DataFrame.
    """
    print(f"\n--- Estadísticas Básicas ---")
    print(f"\nNúmero total de comentarios: {dataframe.shape[0]}")

def display_label_distribution(dataframe, column_name):
    if column_name == 'binary_label':
        dataframe[column_name] = dataframe[column_name].astype(int)
    counts = dataframe[column_name].value_counts()
    percentages = dataframe[column_name].value_counts(normalize=True) * 100
    distribution_table = pd.DataFrame({
        'Conteo': counts,
        'Porcentaje (%)': percentages
    })
    print(distribution_table)


def display_comment_lengths(dataframe, text_column='text_cleaned'):
    """
    Calcula y muestra las estadísticas de longitud de los comentarios (caracteres y palabras).

    Args:
        dataframe (pd.DataFrame): El DataFrame a analizar.
        text_column (str): El nombre de la columna que contiene el texto limpio.
    """
    print(f"\n--- Longitud de Comentarios ({text_column}) ---")

    # Asegúrate de que la columna de texto no tenga valores NaN o vacíos
    dataframe[text_column] = dataframe[text_column].fillna('')

    # Longitud por número de caracteres
    dataframe['char_length'] = dataframe[text_column].apply(len)
    print(f"Longitud promedio de comentarios (caracteres): {dataframe['char_length'].mean():.2f}")
    print(f"Longitud mínima de comentarios (caracteres): {dataframe['char_length'].min()}")
    print(f"Longitud máxima de comentarios (caracteres): {dataframe['char_length'].max()}")


def run_basic_eda(dataframe):
    """
    Ejecuta todas las funciones de estadísticas básicas para el EDA.
    """

    display_total_comments(dataframe)


    display_label_distribution(dataframe, 'binary_label')
    print("Observación: La clase 'No Tóxico' es significativamente más abundante.")


    display_label_distribution(dataframe, 'multi_label')
    print("Observación: Las clases de toxicidad ('Levemente Tóxico', 'Gravemente Tóxico') son minoritarias.")


    display_label_distribution(dataframe, 'original_label')
    print("Observación: Esto muestra la proporción de cada tipo de etiqueta del dataset original en el unificado.")


    # Longitud de Comentarios
    display_comment_lengths(dataframe, 'text_cleaned')



PALETA_BINARIA = [PALETA_ATARDECER_GRIETA['azul_suave'], PALETA_ATARDECER_GRIETA['rojo_profundo']]
PALETA_MULTICLASE = [
    PALETA_ATARDECER_GRIETA['azul_suave'],          # No Tóxico
    PALETA_ATARDECER_GRIETA['naranja_ambar'],       # Levemente Tóxico
    PALETA_ATARDECER_GRIETA['rojo_profundo']        # Gravemente Tóxico
]

PALETA_GENERAL = [
    PALETA_ATARDECER_GRIETA['azul_suave'],
    PALETA_ATARDECER_GRIETA['ocre_claro'],
    PALETA_ATARDECER_GRIETA['verde_oliva_suave'],
    PALETA_ATARDECER_GRIETA['naranja_ambar'],
    PALETA_ATARDECER_GRIETA['terracota'],
    PALETA_ATARDECER_GRIETA['rojo_profundo'],
    PALETA_ATARDECER_GRIETA['purpura_oscuro']
]

CUSTOM_STOPWORDS = set(STOPWORDS) 
CUSTOM_STOPWORDS.add('sepa')

def run_eda_visualizations(dataframe, text_column='text_cleaned', label_column='binary_label'):
    """
    Ejecuta las visualizaciones de EDA (gráficos de barras y nubes de palabras), usando tu paleta de colores personalizada.
    """
    print("\n--- Comienza la Sección de Visualizaciones EDA ---")

    print("\n--- Generando gráficos de barras de distribuciones ---")
    plt.figure(figsize=(18, 6))

    # Gráfico para binary_label
    plt.subplot(1, 3, 1)
    binary_data = dataframe['binary_label'].astype(int).value_counts().rename(index={0: 'No Tóxico', 1: 'Tóxico'})
    sns.barplot(x=binary_data.index, y=binary_data.values, hue=binary_data.index, palette=PALETA_BINARIA, legend=False)
    plt.title('Distribución de Etiqueta Binaria')
    plt.xlabel('Etiqueta')
    plt.ylabel('Conteo de Comentarios')
    plt.xticks(rotation=45, ha='right')

    # Gráfico para multi_label
    plt.subplot(1, 3, 2)
    multi_data = dataframe['multi_label'].value_counts()
    ordered_multi_labels = ['No Tóxico', 'Levemente Tóxico', 'Gravemente Tóxico']
    multi_data = multi_data.reindex(ordered_multi_labels).fillna(0)
    sns.barplot(x=multi_data.index, y=multi_data.values, hue=multi_data.index, palette=PALETA_MULTICLASE, legend=False)
    plt.title('Distribución de Etiqueta Multi-Clase')
    plt.xlabel('Etiqueta')
    plt.ylabel('Conteo de Comentarios')
    plt.xticks(rotation=45, ha='right')

    # Gráfico para original_label
    plt.subplot(1, 3, 3)
    original_data = dataframe['original_label'].value_counts()
    sns.barplot(x=original_data.index, y=original_data.values, hue=original_data.index, palette=PALETA_GENERAL, legend=False)
    plt.title('Distribución de Etiquetas Originales')
    plt.xlabel('Etiqueta')
    plt.ylabel('Conteo de Comentarios')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Nubes de palabras
    print("\n--- Generando nubes de palabras ---")
    dataframe[text_column] = dataframe[text_column].fillna('')
    dataframe[label_column] = dataframe[label_column].astype(int)

    text_no_toxic = " ".join(dataframe[dataframe[label_column] == 0][text_column].astype(str))
    text_toxic = " ".join(dataframe[dataframe[label_column] == 1][text_column].astype(str))

    wordcloud_no_toxic = WordCloud(width=800, height=400, background_color='white',
                                   collocations=False, min_font_size=10,
                                   stopwords=CUSTOM_STOPWORDS).generate(text_no_toxic)
    wordcloud_toxic = WordCloud(width=800, height=400, background_color='black',
                                color_func=lambda *args, **kwargs: PALETA_ATARDECER_GRIETA['rojo_profundo'],
                                collocations=False, min_font_size=10,
                                stopwords=CUSTOM_STOPWORDS).generate(text_toxic)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_no_toxic, interpolation='bilinear')
    plt.title('Nube de Palabras: Comentarios No Tóxicos')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_toxic, interpolation='bilinear')
    plt.title('Nube de Palabras: Comentarios Tóxicos')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

sid = SentimentIntensityAnalyzer()

def classify_sentiment_vader(text, threshold=0.05):
    """
    Clasifica el sentimiento de un texto usando VADER.
    Retorna 'Positivo', 'Negativo (No Tóxico)', o 'Neutral'.
    """
    if not isinstance(text, str):
        return 'Neutral' 

    scores = sid.polarity_scores(text)
    compound_score = scores['compound'] # El score compuesto es el más importante

    if compound_score >= threshold:
        return 'Positivo'
    elif compound_score <= -threshold:
        return 'Negativo (No Tóxico)' # Negativo pero no clasificado como tóxico binario
    else:
        return 'Neutral'

def display_sentiment_breakdown(dataframe, text_column='text', label_column='binary_label', label_value=0, title_suffix=""):
    """
    Calcula y muestra la distribución de sentimiento para comentarios
    filtrados por una etiqueta y un valor de etiqueta específicos, utilizando VADER.
    """
    label_name = "No Tóxicos" if label_value == 0 else "Tóxicos"
    print(f"\n--- Distribución de Sentimiento en Comentarios {label_name} ({text_column}){title_suffix} ---")

    # Filtrar comentarios por el valor de etiqueta proporcionado
    filtered_comments = dataframe[dataframe[label_column] == label_value].copy()

    if not filtered_comments.empty:
        # Aplicar el clasificador de sentimiento de VADER
        filtered_comments['sentiment'] = filtered_comments[text_column].apply(classify_sentiment_vader)
        sentiment_counts = filtered_comments['sentiment'].value_counts()
        sentiment_percentages = filtered_comments['sentiment'].value_counts(normalize=True) * 100

        sentiment_table = pd.DataFrame({
            'Conteo': sentiment_counts,
            'Porcentaje (%)': sentiment_percentages
        })
        print(sentiment_table)

        plt.figure(figsize=(8, 5))
        sentiment_palette = {
            'Positivo': PALETA_ATARDECER_GRIETA['verde_oliva_suave'],
            'Negativo (No Tóxico)': PALETA_ATARDECER_GRIETA['naranja_ambar'],
            'Neutral': PALETA_ATARDECER_GRIETA['azul_suave']
        }
        order = ['Positivo', 'Negativo (No Tóxico)', 'Neutral']
        sns.barplot(x=sentiment_table.index, y=sentiment_table['Conteo'],
                    hue=sentiment_table.index, palette=sentiment_palette, order=order, legend=False)
        plt.title(f'Distribución de Sentimiento en Comentarios {label_name} (VADER){title_suffix}')
        plt.xlabel('Sentimiento')
        plt.ylabel('Conteo de Comentarios')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No hay comentarios {label_name} para analizar el sentimiento.")


def display_extreme_comments_by_sentiment(dataframe, text_column='text', label_column='binary_label', num_examples=3):
    """
    Muestra ejemplos de comentarios con el sentimiento más extremo (positivo y negativo)
    y algunos ejemplos neutrales, dentro de las etiquetas binarias (No Tóxico y Tóxico).
    """
    print("\n--- Ejemplos de Comentarios por Sentimiento Extremo y Toxicidad ---")

    # Asegurarse de que la columna 'sentiment' exista
    # y que 'compound_score' exista, ya que la usaremos para ordenar.
    # Es más eficiente calcular ambas aquí de una vez para todo el DataFrame.
    if 'compound_score' not in dataframe.columns or 'sentiment' not in dataframe.columns:
        print(f"Calculando 'sentiment' y 'compound_score' usando '{text_column}' para todo el DataFrame...")
        dataframe['compound_score'] = dataframe[text_column].apply(lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0.0)
        dataframe['sentiment'] = dataframe[text_column].apply(classify_sentiment_vader)
        print("Columnas 'sentiment' y 'compound_score' calculadas.")
    else:
        print("Las columnas 'sentiment' y 'compound_score' ya existen.")


    # Definir las categorías a explorar
    toxic_labels = {
        0: "No Tóxico",
        1: "Tóxico"
    }

    for label_value, label_name in toxic_labels.items():
        print(f"\n----- Comentarios {label_name} -----")
        current_toxic_df = dataframe[dataframe[label_column] == label_value].copy()

        if current_toxic_df.empty:
            print(f"No hay comentarios {label_name} para mostrar.")
            continue

        # --- Comentarios más Positivos ---
        positive_comments = current_toxic_df[current_toxic_df['sentiment'] == 'Positivo'].sort_values(by='compound_score', ascending=False)
        print("\n  > Más Positivos:")
        if positive_comments.empty:
            print("    No hay comentarios positivos en esta categoría.")
        else:
            for i, row in enumerate(positive_comments.head(num_examples).itertuples()):
                print(f"    {i+1}. (Score: {row.compound_score:.3f}) {getattr(row, text_column)}")

        # --- Comentarios más Negativos (No Tóxicos en sentido VADER) ---
        negative_comments = current_toxic_df[current_toxic_df['sentiment'] == 'Negativo (No Tóxico)'].sort_values(by='compound_score', ascending=True)
        print("\n  > Más Negativos (según VADER):")
        if negative_comments.empty:
            print("    No hay comentarios negativos en esta categoría.")
        else:
            for i, row in enumerate(negative_comments.head(num_examples).itertuples()):
                print(f"    {i+1}. (Score: {row.compound_score:.3f}) {getattr(row, text_column)}")

        # --- Comentarios Neutrales (muestras aleatorias) ---
        neutral_comments = current_toxic_df[current_toxic_df['sentiment'] == 'Neutral']
        print("\n  > Neutrales (ejemplos aleatorios):")
        if neutral_comments.empty:
            print("    No hay comentarios neutrales en esta categoría.")
        else:
            # Muestra un número limitado de muestras aleatorias
            samples = neutral_comments.sample(min(num_examples, len(neutral_comments)), random_state=42)
            for i, comment in enumerate(samples[text_column]):
                print(f"    {i+1}. {comment}")

    print("\n--- Ejemplos de Comentarios Extremos Completados ---")

def mark_specific_toxic_phrases(df):
    toxic_phrases_raw_for_processing = [
    "invoker what the fuck are u playing?", # Vuelve el '?'
    "u just made me cum",
    "habla bien burro",
    "nob team",
    "what is wrong with you guys",
    "like your father",
    "rly [SEPA] look at these scrubs [SEPA] then complain", # Vuelve el '[SEPA]'
    "his father was a cow farmer...", # Vuelven los '...'
    "ebalo [SEPA] shdi davay", # Vuelve el '[SEPA]'
    "comprende? [SEPA] no [SEPA] just say ok daddy", # Vuelve el '?' y '[SEPA]'
    "cada ves tocan [SEPA] mas basuras en el dota", # Vuelve el '[SEPA]'
    "chmo ebanoe",
    "cm thinks trans people struggle",
    "chinese inbred",
    "gg [SEPA] WP [SEPA] NICE [SEPA] STORM [SEPA] he knows [SEPA] how to use his passive xD [SEPA] XD [SEPA] XD [SEPA] THIS GUY iS HILARIOUS [SEPA] SOMEBODY TELL HIM [SEPA] HOW HIS PASSIVE WORKS XD", # Vuelve el '[SEPA]' y se mantiene el 'xD' original
    "SOMeBODY TELL THIS STORM HOW TO USE HIs PASSIVE XD [SEPA] XD [SEPA] XD [SEPA] XD", # Vuelve el '[SEPA]' y se mantiene el 'XD' original
    "gg [SEPA] our storm [SEPA] doesnt know how his passive works xD [SEPA] XD [SEPA] XD [SEPA] XD" # Vuelve el '[SEPA]' y se mantiene el 'xD' original
    ]
    processed_toxic_phrases = [clean_text(phrase) for phrase in toxic_phrases_raw_for_processing]

    if 'text_cleaned' not in df.columns or 'binary_label' not in df.columns or 'multi_label' not in df.columns:
        raise ValueError("El DataFrame debe contener las columnas 'text_cleaned', 'binary_label' y 'multi_label'.")
    df_modified = df.copy()

    # Función auxiliar para aplicar a cada fila
    def check_and_update_labels(row):
        text_to_check = str(row['text_cleaned'])

        is_toxic_by_rule = False
        for toxic_phrase_pattern in processed_toxic_phrases:
            if re.search(re.escape(toxic_phrase_pattern), text_to_check):
                is_toxic_by_rule = True
                break

        if is_toxic_by_rule:
            if row['binary_label'] == 0:
                row['binary_label'] = 1
                row['multi_label'] = 'Gravemente Tóxico'
            elif row['multi_label'] == 'Levemente Tóxico':
                 row['binary_label'] = 1
                 row['multi_label'] = 'Gravemente Tóxico'
        return row

    df_modified = df_modified.apply(check_and_update_labels, axis=1)

    return df_modified