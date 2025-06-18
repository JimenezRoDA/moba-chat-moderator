
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
import json
import time
import random

# --- Configuración inicial de la página de Streamlit ---
# Esta debe ser la PRIMERA llamada a st. en todo el script, después de los imports.
st.set_page_config(
    page_title="Moderador de Chat IA",
    layout="centered",
    page_icon="🤖",
    initial_sidebar_state="collapsed" 
)

# --- 0. Configuración inicial y carga del modelo/tokenizador ---
FINETUNED_MODEL_PATH = './models/modelo_toxicidad_guardado'
BASE_MODEL_NAME_FOR_TOKENIZER = 'distilbert-base-uncased'

num_classes = 4
id_to_label = {
    0: 'Acción/Juego',
    1: 'Gravemente Tóxico',
    2: 'Levemente Tóxico',
    3: 'No Tóxico'
}

label_to_id = {v: k for k, v in id_to_label.items()}

@st.cache_resource
def load_resources_cached(model_path_cached, base_model_name_for_tokenizer_cached, num_labels_cached):
    """
    Carga el tokenizador y el modelo finetuneado, y lo mueve al dispositivo adecuado.
    Esta función se cachea con st.cache_resource para una carga única.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_for_tokenizer_cached)
        model = AutoModelForSequenceClassification.from_pretrained(model_path_cached, num_labels=num_labels_cached)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        
        return tokenizer, model, device

    except Exception as e:
        st.error(f"¡Ups! Error al cargar el modelo o tokenizador desde '{model_path_cached}': {e}")
        st.info("Asegúrate de que el modelo esté entrenado y guardado correctamente en esa ruta.")
        st.info("Si estás ejecutando la app por primera vez o has movido archivos, verifica la ruta y los permisos.")
        return None, None, None

# Llama a la función cacheada para cargar los recursos
tokenizer, model, device = load_resources_cached(FINETUNED_MODEL_PATH, BASE_MODEL_NAME_FOR_TOKENIZER, num_classes)

if model is None or tokenizer is None:
    st.stop() # Detiene la ejecución de la app si el modelo no se pudo cargar

# --- Inicialización de variables de estado de Streamlit ---
# Todas las variables de sesión deben inicializarse para evitar errores.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'grave_count' not in st.session_state:
    st.session_state.grave_count = 0
if 'leve_count' not in st.session_state:
    st.session_state.leve_count = 0
if 'is_banned' not in st.session_state:
    st.session_state.is_banned = False
if 'chat_input' not in st.session_state: # Para el widget de texto del chat
    st.session_state.chat_input = ""

# Umbrales configurables por la empresa para la Demostración Interactiva
if 'max_graves' not in st.session_state:
    st.session_state.max_graves = 3 # Valor por defecto
if 'max_leves' not in st.session_state:
    st.session_state.max_leves = 5 # Valor por defecto
if 'ban_threshold_graves' not in st.session_state:
    st.session_state.ban_threshold_graves = 10 # Umbral de baneo por graves
if 'ban_threshold_total_toxic' not in st.session_state:
    st.session_state.ban_threshold_total_toxic = 15 # Umbral de baneo por toxicidad total

# --- Funciones Auxiliares ---
def get_prediction(text, tokenizer, model, device, id_to_label):
    if model is None or tokenizer is None:
        return "Modelo no cargado", None, None

    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    predicted_id = np.argmax(probabilities).item()
    predicted_label = id_to_label[predicted_id]

    return predicted_label, probabilities, predicted_id

# Función para resetear la demo (usada en la Demostración Interactiva)
def reset_demo():
    st.session_state.grave_count = 0
    st.session_state.leve_count = 0
    # Aseguramos que también se resetean los contadores de aliados/enemigos
    st.session_state.grave_count_ally = 0
    st.session_state.leve_count_ally = 0
    st.session_state.grave_count_enemy = 0
    st.session_state.leve_count_enemy = 0
    st.session_state.is_banned = False
    st.session_state.chat_history = []
    st.session_state.chat_input = "" 

# Función para crear burbujas de mensaje HTML
def create_bubble(message, sender="tú", msg_type="normal"):
    colors = {
        "grave": "#ffcccc", # Rojo claro
        "leve": "#ffe0b2",  # Naranja claro
        "no_toxic": "#e0f7fa", # Azul muy claro
        "action": "#e8f5e9", # Verde claro
        "system": "#eeeeee", # Gris muy claro
        "enemy": "#f3e5f5",  # Morado claro
        "ally": "#dcedc8",   # Verde pistacho claro
        "warning": "#fff3cd", # Amarillo claro
        "ban": "#f8d7da",    # Rojo muy claro, más pastel
    }
    style = f"background-color:{colors.get(msg_type, '#f0f0f0')};border-radius:10px;padding:10px;margin:6px 0"
    return f"""
    <div style='{style}'>
        <strong>{sender}:</strong> {message}
    </div>
    """

# --- NAVEGACIÓN CON PESTAÑAS (TABS) ---
tab_intro, tab_demo, tab_csv, tab_performance, tab_conclusions = st.tabs([
    "1. Introducción",
    "2. Demostración Interactiva",
    "3. Herramienta de Análisis",
    "4. Rendimiento ",
    "5. Próximos Pasos"
])

# --- Contenido de las Páginas ---

with tab_intro:
    st.title("🛡️ Moderador de Chat Inteligente")
    st.subheader("Tu Aliado en la Moderación Online")
    st.markdown("---")

    with st.container():
        st.header("🎯 ¿Por qué un Moderador de Chat con IA?")
        st.markdown("""
        La toxicidad en los entornos online, especialmente en videojuegos multijugador, se ha convertido en un problema creciente que afecta tanto a los usuarios como a las plataformas que los albergan.

        Según recientes investigaciones, más del **80%** de los jugadores han experimentado algún tipo de acoso en juegos en línea. De estos, un **68%** ha recibido amenazas físicas, acecho o acoso sostenido, mientras que el **53%** ha sido víctima de ataques basados en raza, género, orientación sexual u otros aspectos personales.

        Este tipo de comportamiento no solo deteriora la experiencia del usuario, sino que también genera consecuencias serias a nivel social, incluyendo aislamiento, ansiedad e incluso pensamientos depresivos en los afectados.

        En este contexto, un Moderador de Chat Inteligente basado en IA no solo es una herramienta para detectar y filtrar mensajes tóxicos, sino una solución clave para proteger a las comunidades digitales y garantizar un entorno seguro y saludable para todos.
        """)

    st.markdown("---")

    with st.container():
        st.header("🏢 ¿Por qué esto importa a las empresas?")
        st.markdown("""
        La toxicidad online no se limita a los videojuegos. Plataformas con interacción constante — foros, redes sociales, e-commerce o servicios de atención al cliente — enfrentan el desafío de mantener ambientes positivos y seguros.

        Incorporar un moderador de chat con IA ofrece múltiples ventajas comerciales:

        - **Mejora la experiencia del usuario**: Los entornos saludables incrementan la satisfacción, retención y engagement.
        - **Protección de la marca**: Evita que contenidos dañinos o polémicos afecten la reputación de la empresa.
        - **Escalabilidad y eficiencia**: La IA modera grandes volúmenes de mensajes en tiempo real, sin fatiga ni sesgos.
        - **Reducción de costos operativos**: Disminuye la dependencia de equipos humanos para tareas repetitivas.
        - **Análisis y prevención**: Detecta patrones de comportamiento tóxico para anticipar y mitigar problemas futuros.
        """)

    st.markdown("---")

    with st.container():
        st.subheader("📚 Fuentes y estudios relevantes")
        with st.expander("🔎 Consulta las investigaciones que respaldan estos datos"):
            st.markdown("""
            - 📈 *El Telégrafo (2024)* reporta que el acoso en juegos online aumentó un **74%** en el último año. Más del **80%** de jugadores han sufrido acoso, y el **68%** recibió amenazas físicas, acecho o acoso sostenido.  
            [Ver artículo](https://www.eltelegrafo.com.ec/noticias/sociedad/6/acoso-juegos-online-aumento-ultimo-ano)

            - 📊 *Pew Research Center (2021)* señala que el **41%** de los adultos en EE.UU. ha sido acosado online, un problema transversal en múltiples plataformas.  
            [Ver estudio](https://www.pewresearch.org/internet/2021/01/13/the-state-of-online-harassment/)
            """)

with tab_demo:

    with st.sidebar:
        st.header("⚙️ Ajustes de umbrales")

        # Asegúrate de que los sliders usan las variables de st.session_state para sus valores
        st.session_state.max_graves = st.slider(
            "🔴 Umbral de advertencia grave", 1, 10,
            value=st.session_state.max_graves, key="slider_max_graves" # Añadido key
        )
        st.session_state.max_leves = st.slider(
            "🟠 Umbral de advertencia leve", 1, 10,
            value=st.session_state.max_leves, key="slider_max_leves" # Añadido key
        )
        st.session_state.ban_threshold_graves = st.slider(
            "🚫 Umbral para ban por graves", 1, 10,
            value=st.session_state.ban_threshold_graves, key="slider_ban_graves" # Añadido key
        )
        st.session_state.ban_threshold_total_toxic = st.slider(
            "🚫 Umbral para ban por total tóxicos", 1, 20,
            value=st.session_state.ban_threshold_total_toxic, key="slider_ban_total" # Añadido key
        )

        if st.button("Reiniciar Demo", key="reset_demo_button_sidebar"):
            reset_demo()
           

    st.title("🤖 Demo Interactivo: Moderador de Chat")
    st.markdown("---")

    st.header("🤔 ¿Cómo funciona este demo?")
    st.markdown(f"""
    1.  **Escribe un mensaje** en la caja de texto.
    2.  Haz clic en **\"Enviar mensaje\"** o presiona `Enter`.
    3.  El moderador clasificará el mensaje en una de las **{num_classes} categorías** ({', '.join(id_to_label.values())}).
    4.  Si se detecta toxicidad, verás una **advertencia**.
    5.  Los mensajes se acumulan como **'Tóxicos Graves'** y **'Tóxicos Leves'**. Si superas los umbrales configurados en la barra lateral (actualmente **{st.session_state.max_graves} tóxicos graves** y **{st.session_state.max_leves} tóxicos leves**, o **{st.session_state.ban_threshold_graves} tóxicos graves** o **{st.session_state.ban_threshold_total_toxic} mensajes tóxicos totales** para ban), se bloqueará el chat.
    6.  Usa el botón **\"Reiniciar Demo\"** en la barra lateral para empezar de nuevo.
    """)

    st.markdown("---")

    st.subheader("Simulador de chat")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tus mensajes tóxicos graves", st.session_state.grave_count)
        st.metric("Tus mensajes tóxicos leves", st.session_state.leve_count)
    with col2:
        # Asegurarse de que estas variables estén inicializadas en reset_demo o al inicio del script
        if 'grave_count_ally' not in st.session_state: st.session_state.grave_count_ally = 0
        if 'leve_count_ally' not in st.session_state: st.session_state.leve_count_ally = 0
        st.metric("Aliado - tóxicos graves", st.session_state.grave_count_ally)
        st.metric("Aliado - tóxicos leves", st.session_state.leve_count_ally)
    with col3:
        # Asegurarse de que estas variables estén inicializadas en reset_demo o al inicio del script
        if 'grave_count_enemy' not in st.session_state: st.session_state.grave_count_enemy = 0
        if 'leve_count_enemy' not in st.session_state: st.session_state.leve_count_enemy = 0
        st.metric("Enemigo - tóxicos graves", st.session_state.grave_count_enemy)
        st.metric("Enemigo - tóxicos leves", st.session_state.leve_count_enemy)

    # --- Lógica de baneo y chat ---
    # La visualización del historial de chat se mueve aquí, antes del input,
    # para que siempre sea visible, incluso si el usuario está baneado.
    st.markdown("### Historial del Chat:")
    # Mostrar el chat histórico (en orden cronológico si se prefiere)
    # Si quieres el más nuevo arriba, usa reversed(st.session_state.chat_history)
    for bubble in st.session_state.chat_history:
        st.markdown(bubble, unsafe_allow_html=True)

    if st.session_state.is_banned:
        st.markdown(create_bubble("🚫 ¡HAS SIDO BLOQUEADO DEL CHAT! 🚫 Excediste los límites de toxicidad.", "🛑 Sistema", "ban"), unsafe_allow_html=True)
        if st.button("Desbloquear (Reiniciar Demo)", key="unban_button"):
            reset_demo()
            # st.experimental_rerun() # No es necesario aquí

    else:
        # --- Función Callback para el envío de mensaje ---
        def send_message_and_update_demo():
            user_message = st.session_state.chat_input # Accede al valor del input con su key

            if not user_message: # Si el mensaje está vacío
                st.warning("Por favor, escribe un mensaje.")
                return # Salir de la función si no hay mensaje

            # Realizar predicción: ¡Aquí es donde se usa el modelo!
            predicted_label, probabilities, predicted_id = get_prediction(user_message, tokenizer, model, device, id_to_label)

            # --- Añadir mensaje del usuario al historial ---
            if predicted_label == id_to_label[1]:  # Gravemente Tóxico
                st.session_state.grave_count += 1
                bubble = create_bubble(user_message + f"<br><em>Clasificado como: <b>{predicted_label}</b></em>", "🧑 Tú", "grave")
                st.session_state.chat_history.append(bubble)
                if st.session_state.grave_count >= st.session_state.max_graves:
                    warning = create_bubble("🚨 ADVERTENCIA: Demasiados mensajes tóxicos graves.", "⚠️ Sistema", "warning")
                    st.session_state.chat_history.append(warning)
                if st.session_state.grave_count >= st.session_state.ban_threshold_graves:
                    st.session_state.is_banned = True
                    ban_msg = create_bubble("🚫 Has sido bloqueado por toxicidad grave.", "🛑 Sistema", "ban")
                    st.session_state.chat_history.append(ban_msg)

            elif predicted_label == id_to_label[2]:  # Levemente Tóxico
                st.session_state.leve_count += 1
                bubble = create_bubble(user_message + f"<br><em>Clasificado como: <b>{predicted_label}</b></em>", "🧑 Tú", "leve")
                st.session_state.chat_history.append(bubble)
                if st.session_state.leve_count >= st.session_state.max_leves:
                    warning = create_bubble("⚠️ AVISO: Varios mensajes tóxicos leves.", "⚠️ Sistema", "warning")
                    st.session_state.chat_history.append(warning)
                if (st.session_state.grave_count + st.session_state.leve_count) >= st.session_state.ban_threshold_total_toxic:
                    st.session_state.is_banned = True
                    ban_msg = create_bubble("🚫 Has sido bloqueado por toxicidad acumulada.", "🛑 Sistema", "ban")
                    st.session_state.chat_history.append(ban_msg)

            elif predicted_label == id_to_label[3]:  # No Tóxico
                bubble = create_bubble(user_message + f"<br><em>Clasificado como: <b>{predicted_label}</b></em>", "🧑 Tú", "no_toxic")
                st.session_state.chat_history.append(bubble)

            else: # Acción/Juego
                bubble = create_bubble(user_message + f"<br><em>Clasificado como: <b>{predicted_label}</b></em>", "🧑 Tú", "action")
                st.session_state.chat_history.append(bubble)

            # --- Mensajes aleatorios de Aliado/Enemigo ---
            def ally_message_generator(): # Renombrado para evitar conflicto con la función ally_message() anterior
                options = [
                    ("¡Buen trabajo en la línea!", "no_toxic"),
                    ("¡Reagrupémonos en mid!", "action"),
                    ("Cuidado con el jungla enemigo...", "leve"),
                    ("¡Necesito ayuda, me están aplastando!", "grave"),
                ]
                text, msg_type = random.choice(options)
                if msg_type == "grave":
                    st.session_state.grave_count_ally += 1
                elif msg_type == "leve":
                    st.session_state.leve_count_ally += 1
                return create_bubble(text, "🟢 Aliado", "ally") # Usar "ally" para el tipo de burbuja

            def enemy_message_generator(): # Renombrado
                options = [
                    ("¿Qué haces, inútil?", "leve"),
                    ("Te voy a aplastar", "grave"),
                    ("Vamos a ganar esta partida.", "no_toxic"),
                    ("Cállate y farmea.", "action"),
                ]
                text, msg_type = random.choice(options)
                if msg_type == "grave":
                    st.session_state.grave_count_enemy += 1
                elif msg_type == "leve":
                    st.session_state.leve_count_enemy += 1
                return create_bubble(text, "🔴 Enemigo", "enemy") # Usar "enemy" para el tipo de burbuja

            # Añadir mensajes de IA al historial
            st.session_state.chat_history.append(ally_message_generator())
            st.session_state.chat_history.append(enemy_message_generator())

            # Vaciar el input de texto después de todo el procesamiento
            st.session_state.chat_input = ""
            # No necesitas st.experimental_rerun() aquí, on_click maneja el refresco.


        # El campo de entrada de texto
        # Ya no usamos on_change para limpiar, se limpia dentro del callback
        st.text_input("Escribe tu mensaje aquí:", key="chat_input")

        # El botón que activa la función callback para enviar y procesar el mensaje
        st.button("Enviar mensaje", on_click=send_message_and_update_demo, key="send_message_button_demo")

    # Mover la visualización del historial de chat arriba para que siempre sea visible
    # st.markdown("### Historial del Chat:") # Ya lo puse arriba

with tab_csv:
    st.title("🛠️ Herramienta de Análisis de Toxicidad por Lotes")
    st.markdown("---")

    st.header("Carga tu CSV de Comentarios")
    st.write("""
    Esta herramienta te permite subir un archivo CSV con una columna de comentarios
    y el modelo de IA añadirá una nueva columna clasificando cada comentario
    en 'Gravemente Tóxico', 'Levemente Tóxico', 'No Tóxico' o 'Acción/Juego'.
    """)

    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("Archivo CSV cargado exitosamente.")
            st.dataframe(df_upload.head())

            # Permite al usuario seleccionar la columna si no es obvio
            column_to_analyze = st.selectbox(
                "Selecciona la columna que contiene los comentarios:",
                df_upload.columns,
                key="csv_column_selector" # Añadido key
            )

            if st.button("Analizar Comentarios", key="analyze_csv_button"): # Añadido key
                if model is not None and tokenizer is not None:
                    with st.spinner("Analizando comentarios... Esto puede tardar unos minutos para archivos grandes."):
                        predictions = []
                        # Asegurarse de que los comentarios son strings y manejar NaN/vacíos
                        for comment in df_upload[column_to_analyze].astype(str).fillna(''):
                            predicted_label, _, _ = get_prediction(comment, tokenizer, model, device, id_to_label)
                            predictions.append(predicted_label)

                        df_upload['clasificacion_toxicidad'] = predictions
                        st.subheader("Resultados del Análisis:")
                        st.dataframe(df_upload)

                        # Opción de descarga
                        csv_output = df_upload.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar CSV con Clasificaciones",
                            data=csv_output,
                            file_name="comentarios_clasificados.csv",
                            mime="text/csv",
                            key="download_classified_csv" # Añadido key
                        )
                else:
                    st.error("El modelo no está cargado. No se puede realizar el análisis.")
        except Exception as e:
            st.error(f"Error al leer o procesar el archivo CSV: {e}")
            st.info("Asegúrate de que el archivo es un CSV válido y que no contiene caracteres problemáticos.")


with tab_performance:
    st.title("📈 Rendimiento del Modelo: Fiabilidad y Capacidades")
    st.markdown("---")

    st.header("¿Hasta qué punto es fiable nuestro moderador IA?")
    st.write("""
    Nuestro modelo de Inteligencia Artificial es el corazón de este sistema de moderación. Ha sido entrenado
    con mucho cuidado y probado a fondo para asegurar que es **eficaz y fiable**. Aquí te explicamos, de
    forma sencilla, qué tan bien funciona y dónde brilla más.
    """)

    # --- DATOS DE RENDIMIENTO HARDCODEADOS DIRECTAMENTE EN EL CÓDIGO ---
    # Estos son los datos que antes se cargaban del JSON. Ahora están aquí.
    hardcoded_report_data = {
        "Acción/Juego": {"precision": 0.739, "recall": 0.845, "f1-score": 0.788, "support": "N/A"}, # 'support' es el número de ejemplos
        "Gravemente Tóxico": {"precision": 0.872, "recall": 0.886, "f1-score": 0.879, "support": "N/A"},
        "Levemente Tóxico": {"precision": 0.793, "recall": 0.718, "f1-score": 0.753, "support": "N/A"},
        "No Tóxico": {"precision": 0.954, "recall": 0.954, "f1-score": 0.954, "support": "N/A"},
        "accuracy": 0.9174,
        "macro avg": {"precision": 0.8395, "recall": 0.85075, "f1-score": 0.8435, "support": "N/A"},
        "weighted avg": {"precision": 0.9175, "recall": 0.9174, "f1-score": 0.9175, "support": "N/A"}
    }

    # Creamos el DataFrame directamente desde los datos hardcodeados
    report_df = pd.DataFrame(hardcoded_report_data).transpose()

    # Redondeamos las columnas numéricas para una mejor visualización
    # y manejamos el caso de 'accuracy' que no es un diccionario
    for col in ['precision', 'recall', 'f1-score']:
        if col in report_df.columns:
            report_df[col] = pd.to_numeric(report_df[col], errors='coerce').round(3)
    if 'accuracy' in report_df.index:
        report_df.loc['accuracy'] = pd.to_numeric(report_df.loc['accuracy'], errors='coerce').round(4) # Accuracy puede tener más decimales


    st.subheader("Métricas de Evaluación Detalladas:")
    st.dataframe(report_df)

    st.markdown("""
    ### **¿Cómo se comporta el modelo? Un resumen para todos:**

    Imagina que el modelo es como un **experto revisor de chat**. Analiza los mensajes y decide
    si son tóxicos o no. Basado en nuestras pruebas, aquí te contamos lo que hace genial y dónde puede mejorar:

    ---

    #### **✅ Lo que el modelo hace EXCEPCIONALMENTE bien (Puntos Fuertes):**

    * **Detecta la toxicidad más grave (¡Y muy bien!):** Cuando un mensaje es **"Gravemente Tóxico"**, nuestro modelo es **muy, muy bueno** identificándolo. Tiene una fiabilidad de casi el **88%** en esta tarea. Esto es crucial porque significa que los mensajes más dañinos y problemáticos son rápidamente detectados y gestionados. Protege a tu comunidad de lo peor.

    * **Reconoce lo que NO es tóxico (¡Casi perfecto!):** El modelo es **excelente** a la hora de decir cuándo un mensaje es **"No Tóxico"**. Con una fiabilidad del **95.4%**, casi nunca se equivoca al dejar pasar conversaciones normales y saludables. Esto es vital para no molestar a los usuarios que se comunican bien.

    * **Precisión general impresionante:** En total, el modelo acierta en casi **9 de cada 10 mensajes** (un **91.74%** de precisión general). Esto nos dice que, en la mayoría de los casos, puedes confiar en su criterio para mantener el chat limpio.

    ---

    #### **⚠️ Dónde el modelo puede ser más "cauteloso" (Áreas de Oportunidad):**

    * **Toxicidad leve (el matiz es complicado):** Identificar la toxicidad **"Levemente Tóxica"** es el reto más grande. Piensa en el sarcasmo, las indirectas o los comentarios un poco molestos pero no tan agresivos. Aquí el modelo es **razonablemente bueno** (alrededor del **75.3%**), pero a veces puede **dejar pasar** algunos mensajes sutiles. Esto es normal en la IA, ya que la línea entre un comentario "regular" y uno "ligeramente tóxico" puede ser muy fina y subjetiva para una máquina.

    * **Mensajes de "Acción/Juego":** Aunque clasifica bien la mayoría, en los mensajes de **"Acción/Juego"** (como "¡Vamos a por el objetivo!"), su precisión es un poco más baja (alrededor del **78.8%**). A veces, puede confundir estos mensajes con otros "No Tóxicos". Esto se debe a que el lenguaje del juego puede ser muy similar al de las conversaciones normales.

    ---

    #### **En resumen:**

    Este moderador es una **herramienta poderosa y muy fiable** para detectar los mensajes más peligrosos y para asegurar que el chat normal fluye sin problemas. Para las toxicidades más sutiles o las categorías de contenido específicas, es una base excelente que puede mejorarse con el tiempo y el uso.

    ---
    """)

    # --- La sección de descarga del PDF se mantiene igual ---
    st.markdown("---")
    st.header("¿Quieres ver los números exactos en detalle?")
    st.write("""
    Para aquellos interesados en los detalles técnicos, las métricas completas y un análisis exhaustivo
    del rendimiento de nuestro modelo, puedes descargar el informe completo de evaluación aquí.
    Contiene todos los datos que los expertos necesitan:
    """)

    try:
        # Asegúrate de que 'informe_modelo.pdf' esté en la ruta './reports/'
        with open("./reports/informe_modelo.pdf", "rb") as file:
            st.download_button(
                label="Descargar Informe Completo (PDF)",
                data=file,
                file_name="informe_moderador_ia.pdf",
                mime="application/pdf",
                key="download_report_pdf"
            )
    except FileNotFoundError:
        st.warning("El archivo del informe (`informe_modelo.pdf`) no se encontró en la ruta `./reports/`. Por favor, verifica que la carpeta `reports` y el archivo existan.")
    except Exception as e:
        st.error(f"Error al preparar el archivo para descarga: {e}")


with tab_conclusions:
    st.title("🔮 Conclusiones y Próximos Pasos: El Futuro de la Moderación")
    st.markdown("---")

    st.header("Explorando el Horizonte de la Moderación Inteligente")
    st.write("""
    Este moderador de chat es un componente fundamental para una estrategia de moderación de contenido robusta.
    Mirando hacia el futuro, se pueden explorar varias vías de mejora e integración:
    """)
    st.markdown("""
    * **Arquitectura de Integración (APIs):** El modelo entrenado puede ser desplegado como un **microservicio** al que las plataformas de chat se conectarían a través de una API REST (por ejemplo, con Flask, FastAPI). Esto permite una **moderación en tiempo real** donde cada mensaje enviado pasa por la IA antes de ser visible.
    * **IA Conversacional para Intervención:** Imagina una IA que no solo detecta, sino que también interviene.
        * **Feedback Personalizado:** "Lo que has dicho es ofensivo. Por favor, modera tu lenguaje."
        * **Filtrado Activo con Reescritura (IA de Traducción de Toxicidad):** Esto es un concepto innovador. Un usuario escribe: "*me cago en tu ... no sabes jugar tendrias que hacer esto manco*". El sistema detecta la toxicidad y, antes de que el mensaje sea enviado al chat público, lo reescribe a algo como: "*Podrías intentar esto en su lugar. Me estoy frustrando con el juego.*" Esto permite a los usuarios expresar sus emociones, pero asegura que el mensaje que llega al chat sea constructivo y no tóxico, manteniendo un flujo de comunicación positivo. Es un filtro proactivo que "traduce" la toxicidad en constructividad.

    Este proyecto representa un punto de partida poderoso. La evolución hacia sistemas de IA más contextuales y proactivos es el camino hacia comunidades digitales verdaderamente seguras y acogedoras.
    """)