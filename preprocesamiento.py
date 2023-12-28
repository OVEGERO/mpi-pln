from mpi4py import MPI
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
import mysql.connector


def cargar_dataset(path, start_line=0, end_line=None):
    # Modificación para cargar un rango específico de líneas
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        num_ensayo_actual = None
        contador_filas = 0
        for line in file:
            if end_line is not None and contador_filas >= end_line:
                break
            if start_line <= contador_filas:
                if line.startswith('###'):
                    num_ensayo_actual = line[3:].strip()
                elif '\t' in line:
                    seccion, texto = line.strip().split('\t', 1)
                    data.append([num_ensayo_actual, seccion, texto])
            contador_filas += 1
    return pd.DataFrame(data, columns=['num_ensayo', 'seccion', 'text'])

def limpieza_incial(df):
    return df.dropna()

def tokenizar_texto(text):
    return word_tokenize(text)

def preprocesar_tokens(tokens, lemmatizer, stuff_to_be_removed):
    filtered_tokens = [re.sub(r"&lt;/?.*?&gt;|[^a-zA-Z]|(\d|\W)+", '', token).lower() for token in tokens if isinstance(token, str)]
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens if token not in stuff_to_be_removed and token.strip()]
    return cleaned_tokens

def aplicar_preprocesamiento(df):
    lemmatizer = WordNetLemmatizer()
    stuff_to_be_removed = set(stopwords.words("english")).union(set(punctuation))
    df['text'] = df['text'].apply(tokenizar_texto)
    df['text'] = df['text'].apply(lambda x: preprocesar_tokens(x, lemmatizer, stuff_to_be_removed))
    return df

def construir_bolsa_de_palabras(df):
    all_tokens = sum(df['text'], [])
    return Counter(all_tokens)

def inserciones_base_de_datos(df_procesado, bolsa_de_palabras, conn):
    cursor = conn.cursor()

    # Insertar en la tabla num_ensayo_clinico
    num_ensayos_unicos = df_procesado['num_ensayo'].unique()
    for num in num_ensayos_unicos:
        cursor.execute("INSERT INTO num_ensayo_clinico (num_ensayo) VALUES (%s)", (num,))
        conn.commit()

        # Obtener el ID del ensayo clínico recién insertado
        cursor.execute("SELECT num_ensayo_clinico_id FROM num_ensayo_clinico WHERE num_ensayo = %s", (num,))
        num_ensayo_id = cursor.fetchone()[0]

        # Insertar en la tabla section
        secciones_unicas = df_procesado[df_procesado['num_ensayo'] == num]['seccion'].unique()
        for seccion in secciones_unicas:
            cursor.execute("INSERT INTO section (name_section, num_ensayo_clinico_id) VALUES (%s, %s)", (seccion, num_ensayo_id,))
            conn.commit()

    # Insertar en la tabla token y section_token
    for _, row in df_procesado.iterrows():
        cursor.execute("SELECT section_id FROM section WHERE name_section = %s AND num_ensayo_clinico_id = (SELECT num_ensayo_clinico_id FROM num_ensayo_clinico WHERE num_ensayo = %s)", (row['seccion'], row['num_ensayo']))
        section_id = cursor.fetchone()[0]

        for token in row['text']:
            cursor.execute("INSERT INTO token (token_name) VALUES (%s) ON DUPLICATE KEY UPDATE token_id=LAST_INSERT_ID(token_id)", (token,))
            token_id = cursor.lastrowid
            cursor.execute("INSERT INTO section_token (token_id, section_id) VALUES (%s, %s)", (token_id, section_id))
            conn.commit()

    # Insertar en la tabla bolsa_palabras
    for token, repeticiones in bolsa_de_palabras.items():
        cursor.execute("SELECT token_id FROM token WHERE token_name = %s", (token,))
        result = cursor.fetchone()
        if result:
            token_id = result[0]
            cursor.execute("INSERT INTO bolsa_palabras (token_id, repeticiones) VALUES (%s, %s)", (token_id, repeticiones))
            conn.commit()

    cursor.close()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size <= 1:
        raise ValueError("El tamaño del comunicador MPI debe ser mayor que 1.")

    if rank == 0:
        conn = mysql.connector.connect(
            host="hostname",
            user="username",
            password="password",
            database="dbname"
        )
        df = cargar_dataset('./pubmed-rct-master/PubMed_200k_RCT/train.txt')
        split_data = np.array_split(df, size-1)
        for i in range(1, size):
            comm.send(split_data[i-1], dest=i)
    else:
        data = comm.recv(source=0)
        df_limpio = limpieza_incial(data)
        df_procesado = aplicar_preprocesamiento(df_limpio)
        bolsa_de_palabras = construir_bolsa_de_palabras(df_procesado)
        # Envía tanto la bolsa de palabras como el dataset preprocesado
        comm.send((bolsa_de_palabras, df_procesado), dest=0)

    if rank == 0:
        bolsa_palabras_total = Counter()
        df_total = pd.DataFrame()
        for i in range(1, size):
            bolsa_de_palabras, df_procesado = comm.recv(source=i)
            bolsa_palabras_total.update(bolsa_de_palabras)
            df_total = pd.concat([df_total, df_procesado], ignore_index=True)
        
        try:
            inserciones_base_de_datos(df_total, bolsa_palabras_total, conn)
        finally:
            conn.close()  # Asegúrate de cerrar la conexión cuando hayas terminado

if __name__ == "__main__":
    main()

