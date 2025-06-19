# 💬 Moderador de Chat Inteligente – Por un Chat Libre de Toxicidad

## 🧩 Contexto del Proyecto

> El Moderador de Chat Inteligente es una solución impulsada por Machine Learning diseñada para combatir la **toxicidad en entornos online**, especialmente en plataformas de juegos cooperativos como los MOBAs.

> Como jugadora activa de League of Legends, he experimentado de primera mano cómo el lenguaje ofensivo y disruptivo puede arruinar la experiencia de juego y desmotivar a los usuarios. Esta problemática es recurrente en la mayoría de los chats online y afecta directamente la salud de las comunidades.

> **Problemática:** La proliferación de comentarios tóxicos crea ambientes hostiles, daña la reputación de las plataformas y disminuye la retención de usuarios.

> **Objetivo:**
> Desarrollar un clasificador de texto robusto y adaptable que identifique automáticamente diferentes niveles de toxicidad, permitiendo a las plataformas crear entornos más seguros y positivos para sus usuarios.

> [!NOTE]
> Este proyecto fue desarrollado como parte de un Bootcamp de Análisis de Datos, replicando un caso real de análisis y modelado de datos textuales.

---

<details>
<summary>📦 <strong>Resumen del Dataset</strong></summary>

El modelo fue entrenado con un extenso dataset de conversaciones de chat, crucial para capturar la diversidad del lenguaje online:

- **Origen Principal:** Datos de chat de juegos, incluyendo un conjunto significativo del repositorio **"Dota 2 Toxic Chat Data" de Kaggle**, que contiene una amplia gama de interacciones en el contexto de juegos multijugador online.
- **Diversidad:** El dataset abarca una variedad de mensajes que reflejan la dinámica real de los chats, desde comentarios inofensivos hasta lenguaje ofensivo y tóxico.
- **Adaptabilidad:** Aunque los datos provienen de Dota 2, la naturaleza del lenguaje tóxico es transversal. Esto hace que el modelo sea **altamente adaptable y escalable para funcionar eficazmente en cualquier otro MOBA** (como League of Legends, Smite, etc.) y, en general, en cualquier plataforma de chat o comunidad online que busque una moderación de contenido avanzada.

</details>

---

<details>
<summary>🧹 <strong>Limpieza y Preparación de Datos</strong></summary>

Para asegurar la calidad y el balance del modelo, se aplicaron los siguientes pasos clave:

- **Estandarización:** Normalización y limpieza del texto para manejar variaciones como mayúsculas/minúsculas, puntuación y caracteres especiales.
- **Tokenización:** Conversión del texto a un formato numérico (tokens) comprensible para el modelo, utilizando el tokenizador de **DistilBERT**.
- **Manejo de Desequilibrio de Clases:** Se utilizó el algoritmo **RandomOverSampler** (parte de `imblearn`) para sobremuestrear las categorías minoritarias (por ejemplo, "Levemente Tóxico" y "Gravemente Tóxico"). Esto es crucial para que el modelo aprenda a identificar estas clases importantes con mayor precisión, sin estar sesgado por la abundancia de mensajes no tóxicos.
- **Preparación para Hugging Face:** Adaptación de los datos al formato `Dataset` compatible con la librería `Hugging Face Transformers` para el entrenamiento.

</details>

---

<details>
<summary>📊 <strong>Modelo y Entrenamiento</strong></summary>

- **Algoritmo Base:** Se utilizó **DistilBERT**, una versión más pequeña y rápida de BERT, optimizada para tareas de clasificación de texto. DistilBERT es un modelo de última generación pre-entrenado en grandes volúmenes de texto, lo que le permite entender el contexto y los matices del lenguaje de forma excepcional.
- **Técnica:** **Fine-tuning** (ajuste fino) de DistilBERT. Esto significa que el modelo pre-entrenado se adaptó específicamente a la tarea de clasificar mensajes de chat tóxicos, aprendiendo de nuestros datos etiquetados.
- **Clasificación Multicategoría:** El modelo clasifica los mensajes en cuatro categorías principales:
    - **No Tóxico:** Mensajes inofensivos y respetuosos.
    - **Acción/Juego:** Mensajes relacionados con la dinámica del juego sin toxicidad.
    - **Levemente Tóxico:** Mensajes con tono negativo o sarcasmo sutil, pero no gravemente ofensivos.
    - **Gravemente Tóxico:** Mensajes con lenguaje vulgar, amenazas o insultos severos.
- **Plataforma de Entrenamiento:** Se utilizó el `Trainer` de la librería `Hugging Face Transformers` para una gestión eficiente del entrenamiento.

</details>

---

<details>
<summary>📈 <strong>Resultados Clave y Fiabilidad del Modelo</strong></summary>

- **Alta Precisión en Detección Crítica:** El modelo demuestra una **alta fiabilidad** en la clasificación de mensajes, especialmente en la detección de contenido "**Gravemente Tóxico**" y "**No Tóxico**". Esto significa que es muy efectivo en identificar el contenido más dañino y en diferenciarlo de los mensajes seguros.
- **Manejo de Sutiles Matices:** Gracias al entrenamiento con DistilBERT y el sobremuestreo, el modelo también tiene una capacidad significativa para identificar el contenido "**Levemente Tóxico**" y de "**Acción/Juego**", aunque estas categorías pueden presentar un mayor desafío debido a su naturaleza más ambigua o su menor representación inicial en los datos.
- **Fiabilidad en Contextos Empresariales:** La robustez del modelo lo hace idóneo para su aplicación en entornos reales, proporcionando una base sólida para la automatización de la moderación de contenido.

**[Aquí se insertarán las métricas detalladas y su explicación simplificada una vez que estén disponibles y el Streamlit funcione.]**

---

### 📄 Informe Detallado del Modelo

Para un análisis técnico más profundo del rendimiento del modelo y las decisiones de diseño, puedes descargar el informe completo aquí:

[Enlace de descarga del informe (PDF/Markdown) - ¡Pronto disponible!]

</details>

---

<details>
<summary>🧭 <strong>Recomendaciones y Uso</strong></summary>

El Moderador de Chat Inteligente es una herramienta potente, pero su implementación óptima requiere considerar:

- **Herramienta de Apoyo, No Sustituto Total:** Debe complementar, no reemplazar por completo, la moderación humana, especialmente para casos complejos que requieran juicio contextual.
- **Umbrales Configurables:** Las empresas pueden ajustar los umbrales de toxicidad y baneo (como se demuestra en la aplicación Streamlit) para adaptar la sensibilidad del moderador a sus políticas específicas de comunidad.
- **Monitoreo Continuo:** Los modelos de IA necesitan ser monitoreados y reentrenados periódicamente con nuevos datos para adaptarse a la evolución del lenguaje y las nuevas formas de toxicidad.
- **Feedback Loop:** Implementar un sistema donde los moderadores humanos puedan corregir las predicciones del modelo, lo que a su vez retroalimenta y mejora futuras versiones del modelo.

</details>

---

## 🚀 Aplicación Web Interactiva con Streamlit

Hemos desarrollado una aplicación web interactiva utilizando **Streamlit** que te permite experimentar directamente con el moderador de chat:

- **Demo de Chat en Vivo:** Escribe mensajes y observa cómo el modelo los clasifica en tiempo real. Experimenta con diferentes niveles de toxicidad para ver cómo el sistema reacciona y aplica avisos o incluso "baneos" simulados según los umbrales configurables.
- **Umbrales Personalizables para Empresas:** Ajusta fácilmente los límites de toxicidad para ver cómo las reglas de moderación impactan la experiencia del usuario, demostrando la flexibilidad de la herramienta para adaptarse a las políticas de cada plataforma.
- **Herramienta de Análisis CSV:** Sube tu propio archivo de comentarios (`.csv`) y obtén una clasificación instantánea de la toxicidad de cada mensaje, ideal para auditorías de contenido o análisis de grandes volúmenes de datos.

👉 [Prueba la app online aquí](https://moba-chat-moderator-gcnvlaagvoqyzxjker459w.streamlit.app/)

---

## ¡Explora Nuestra Presentación!

Hemos preparado una presentación detallada para explicar el funcionamiento, la problemática y las consideraciones clave detrás de nuestro Moderador de Chat Inteligente. ¡Descubre cómo la IA puede transformar las comunidades online!

[![Visita la presentación en Prezi](https://img.shields.io/badge/Prezi-Ver%20Presentaci%C3%B3n-blue?style=for-the-badge&logo=prezi)](https://prezi.com/view/QNpmIlJ4cWOItZ1t90Re/)

---

## Próximos pasos
- 📊 Ampliar el dataset: Incluir más ejemplos variados de toxicidad permitirá mejorar la precisión del modelo y adaptarse a diferentes formas de comunicación.
- 🌍 Multilingüismo: Adaptar el modelo para detectar toxicidad en otros idiomas, no solo en inglés, ampliando su utilidad a nivel global.
- 🧪 Pruebas en entornos reales: Aplicar el sistema con supervisión humana en plataformas reales para identificar errores, ajustar el modelo y alimentarlo con nuevos casos.
- 🧰 APIs modulares: Desarrollar APIs robustas y plug-and-play que faciliten su integración por parte de terceros (empresas, desarrolladores, plataformas).

---

<details>
  <summary>
    <h2>👤 Autora y Tecnologías</h2>
  </summary>

[![Rocío](https://img.shields.io/badge/@JimenezRoDA-GitHub-181717?logo=github&style=flat-square)](https://github.com/JimenezRoDA)

---

![Python](https://img.shields.io/badge/Python-3.12.7-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Tools-green?logo=scikitlearn)
![Imblearn](https://img.shields.io/badge/Imblearn-Imbalanced%20Data-purple)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

[🔝 Volver arriba](#-moderador-de-chat-inteligente--por-un-chat-libre-de-toxicidad)
</details>
