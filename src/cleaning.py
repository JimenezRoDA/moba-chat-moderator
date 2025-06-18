import pandas as pd 
import re

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


# 1. Mapeo BINARIO

# Diccionario de mapeo para CONDA
conda_binary_mapping = {
    'E': 1,  # Explícito -> Tóxico
    'I': 1,  # Implícito -> Tóxico
    'A': 0,  # Acción -> No Tóxico
    'O': 0   # Otro -> No Tóxico
}

# Diccionario de mapeo para Hugging Face
hf_binary_mapping = {
    0: 0, # No tóxico -> No Tóxico
    1: 1, # Moderado/Leve -> Tóxico
    2: 1  # Máx. toxicidad -> Tóxico
}

def apply_binary_mapping(df, source_type):
    """Aplica el mapeo binario a la columna 'original_label'."""
    if source_type == 'conda':
        df['binary_label'] = df['original_label'].map(conda_binary_mapping)
    elif source_type == 'hf':
        df['binary_label'] = df['original_label'].map(hf_binary_mapping)
    else:
        raise ValueError("source_type debe ser 'conda' o 'hf'")
    return df


# 2. Mapeo MULTI-CLASE (Opcional pero recomendado para más granularidad)

# Diccionario de mapeo para CONDA (multi-clase)
conda_multi_mapping = {
    'O': 'No Tóxico',       # Other
    'A': 'Acción/Juego',    # Action (relacionado con el juego, no tóxico)
    'I': 'Levemente Tóxico', # Implicit toxicity
    'E': 'Gravemente Tóxico' # Explicit toxicity
}

# Diccionario de mapeo para Hugging Face (multi-clase)
hf_multi_mapping = {
    0: 'No Tóxico',         # Non-toxic
    1: 'Levemente Tóxico',   # Mildly toxic / potentially toxic
    2: 'Gravemente Tóxico'  # Severely toxic
}

# Función para aplicar el mapeo multi-clase
def apply_multi_mapping(df, source_type):
    """Aplica el mapeo multi-clase a la columna 'original_label'."""
    if source_type == 'conda':
        df['multi_label'] = df['original_label'].map(conda_multi_mapping)
    elif source_type == 'hf':
        df['multi_label'] = df['original_label'].map(hf_multi_mapping)
    else:
        raise ValueError("source_type debe ser 'conda' o 'hf'")
    return df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_manual_toxic_overrides(df):
    """
    Aplica reglas manuales para sobrescribir etiquetas de toxicidad
    basadas en palabras clave o patrones identificados.
    """
    # Lista de palabras o frases fuertemente tóxicas que invalidan etiquetas no tóxicas
    # Puedes ampliar esta lista con lo que has encontrado en tu revisión manual
    toxic_keywords_strong = [
        r'\bautist\b', r'\bmonkeys\b', r'\bdumbfuck\b', r'\brata\b', r'\bfucker\b',
        r'\bscum\b', r'\bheretics\b', r'finger you', r'\bputins\b', r'\bpendejo\b',
        r'\bdildo\b', r'\bgay\b', r'\bwhore\b', r'\bruzzians\b', r'\bsex\b', r'\bdownies\b',
        r'\bmuslim dont commend\b', r'\bnigga\b', r'\bsuicide over a game\b', r'\bprostit\b',
        r'\bmrd\b', r'\bshit\b', r'\bpissy\b', r'\bclowns\b', r'\btrashcans\b',
        r'\bpussy\b', r'\bfaggot\b', r'\bidiot\b', r'\bcancerous\b', r'\bhdp\b',
        r'\bchupap\b', r'\bpinga\b', r'\bfucking\b' # Add more from your review!
    ]

    # Patrones para identificar acoso o ataques específicos
    toxic_patterns = [
        r'reported ur mom', # "i reported ur mom last night"
        r'report the peruvians', # racial/xenophobic
        r'break ur hand or brain', # threat of harm
        r'u talk too much .* like a girl', # sexist
        r'typical woman gamer', # sexist
        r'im reporting you sepa gay', # homophobic
        r'son of a whore', # extreme insult
        r'death and woe to all', # hate speech
        r'ill give u sex', # harassment
        r'team rat', # dehumanizing insult
        r'pobre lloron', # belittling
        r'you are all heretics', # derogatory towards group
        r'so fat', # body shaming
        r'get suicide', # trivializing suicide
        r'the jerk store called', # derogatory phrase
        r'play like bots', # skill shaming
        r'suck it', # vulgar/sexual
        r'my wit and your mom have that in common', # vulgar personal attack
        r'two dads are coming', # potentially homophobic depending on context
        r'stop playing', # highly unwelcoming/derogatory
        r'dumb', r'idiot', r'stupid', # common insults
        r'trash', r'noob', r'bad', # skill-based insults
        r'cry', r'crying', # mocking
        r'cancer', # derogatory use of illness
        r'autist', 'downies' # ableist insults
    ]
    grave_toxic_patterns = [
        r'\bautist\b', r'\bmonkeys\b', r'\bdumbfuck\b', r'\brata\b', r'\bfucker\b',
        r'\bscum\b', r'\bheretics\b', r'finger you', r'\bputins\b', r'\bpendejo\b',
        r'\bdildo\b', r'\bgay\b', r'\bwhore\b', r'\bruzzians\b', r'\bsex\b', r'\bdownies\b',
        r'\bmuslim dont commend\b', r'\bnigga\b', r'\bsuicide over a game\b',
        r'reported ur mom', r'report the peruvians', r'break ur hand or brain',
        r'son of a whore', r'death and woe to all', r'ill give u sex',
        r'drow has downies', r'win nigga', r'hes gonna suicide over a game sad',
        r'\bchupap\b', r'\bpinga\b', r'\bfaggot\b', r'\bhdp\b', r'\bfucking\b'
        # Puedes añadir más aquí de tu revisión que sean claramente graves
    ]
    gravely_toxic_patterns = toxic_keywords_strong + toxic_patterns + grave_toxic_patterns
    lightly_toxic_patterns = [
        r'\bnoob\b', r'\bclowny\b', r'\bcumback\b', r'\bcry\b',
        r'\bidiot\b', r'\bstupid\b', r'\btrash\b', r'\bbad\b', r'\bsuck\b',
        r'\bclowns\b', r'\btrashcans\b', r'\bpissy\b', r'\blame\b', r'\btoxic\b',
        r'\bcancerous\b', r'\bpotato servers\b', r'\bbots\b', r'\blame\b',
        r'\bjerk store\b', r'\blloron\b', r'\basquerosa\b', r'\bmrd\b',
        r'u talk too much .* like a girl', r'typical woman gamer', # pueden ser leves o graves dependiendo del contexto
        r'play genshin', r'stop playing', # a veces burlas, otras veces solo sugerencias
        r'good handicap for us', r'pobre lloron', r'sad' # sarcasmo/burla
        # Añade más aquí de tu revisión que sean levemente tóxicos
    ]

    # 1. Identificar comentarios GRAVEMENTE TÓXICOS
    is_gravely_toxic = df['text_cleaned'].apply(lambda x: any(re.search(pattern, x) for pattern in gravely_toxic_patterns))

    # 2. Identificar comentarios LEVEMENTE TÓXICOS
    # OJO: Excluimos aquellos que ya identificamos como GRAVEMENTE TÓXICOS para evitar sobrescribir
    is_lightly_toxic = df['text_cleaned'].apply(lambda x: any(re.search(pattern, x) for pattern in lightly_toxic_patterns))
    is_lightly_toxic = is_lightly_toxic & ~is_gravely_toxic # Solo si NO es ya gravemente tóxico

    # 3. Aplicar sobrescrituras (la orden importa: graves primero)

    # Sobreescribir binary_label: Si es tóxico (grave o leve), que sea 1
    # Esto debe aplicarse a ambos grupos para que todo lo identificado como tóxico sea 1 en binario
    df.loc[is_gravely_toxic | is_lightly_toxic, 'binary_label'] = 1


    # Sobreescribir multi_label para GRAVEMENTE TÓXICOS
    df.loc[is_gravely_toxic, 'multi_label'] = 'Gravemente Tóxico'

    # Sobreescribir multi_label para LEVEMENTE TÓXICOS
    df.loc[is_lightly_toxic, 'multi_label'] = 'Levemente Tóxico'

    return df