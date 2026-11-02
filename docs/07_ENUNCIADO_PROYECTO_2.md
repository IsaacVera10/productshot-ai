# Proyecto #2: Sistema Generativo Interactivo para Variaciones Visuales con Feedback Humano y Explicación mediante Agentes de IA

**Departamento de Ciencia de la Computación — Universidad de Ingeniería y Tecnología (UTEC)**  
**Curso:** Planificación y Toma de Decisiones en IA  
**Profesor:** Dra. Aurea Soriano-Vargas

> **Fuente:** `Evaluación_Enunciado Proyecto2.pdf` (5 páginas). Transcripción de consulta con formato Markdown; se han normalizado los saltos de línea y palabras partidas al fin de línea. El documento PDF original es la fuente normativa.

## 1. Introducción

En los últimos años, los avances en inteligencia artificial han transformado profundamente la manera en que los sistemas computacionales generan, interpretan y utilizan información. En particular, los modelos generativos, impulsados por arquitecturas profundas y modelos fundacionales, han permitido la creación automática de contenido visual de alta calidad, abriendo nuevas posibilidades en aplicaciones como diseño, comunicación visual y sistemas interactivos.

Sin embargo, la generación automática de contenido plantea nuevos desafíos en términos de toma de decisiones, control del proceso creativo y confiabilidad de los resultados. A diferencia de los enfoques tradicionales de inteligencia artificial centrados en la predicción, los sistemas generativos requieren mecanismos que permitan evaluar, seleccionar y justificar las salidas producidas, especialmente cuando estas impactan directamente en contextos humanos o institucionales.

En este contexto, la integración del ser humano en el ciclo de decisión se vuelve fundamental. La colaboración humano–IA permite aprovechar las capacidades complementarias de ambos: mientras los modelos generativos exploran un amplio espacio de soluciones posibles, los usuarios aportan criterio, contexto y validación. Este enfoque híbrido no solo mejora la calidad de las decisiones, sino que también incrementa la transparencia y la confianza en los sistemas.

El presente proyecto propone el desarrollo de un sistema generativo interactivo capaz de producir variaciones visuales a partir de una imagen de entrada proporcionada directamente por el usuario, integrando explícitamente la retroalimentación del usuario en el proceso de selección. A través de una interfaz gráfica, el usuario podrá cargar una imagen, evaluar las alternativas generadas y aceptar o rechazar cada una de ellas. Posteriormente, un agente basado en modelos de lenguaje será responsable de sintetizar las decisiones tomadas, generando una explicación textual coherente que describa el proceso de selección.

De esta manera, el proyecto no solo aborda la generación de contenido visual, sino también aspectos clave de la inteligencia artificial moderna, tales como la interacción humano-IA, la explicabilidad, la toma de decisiones asistida y el diseño de sistemas inteligentes integrados. Este enfoque permite a los estudiantes experimentar con tecnologías de vanguardia en un contexto aplicado, reflexionando además sobre sus implicaciones técnicas, sociales y éticas.

## 2. Indicaciones generales

### 2.1. Organización del trabajo

El proyecto deberá desarrollarse en equipos de tres a cinco estudiantes. Cada grupo será responsable de diseñar, implementar y evaluar un sistema generativo interactivo completo, integrando los distintos componentes requeridos. Se espera una distribución equilibrada de responsabilidades, abarcando aspectos como la generación de imágenes, el desarrollo de la interfaz, la integración del agente basado en modelos de lenguaje y la evaluación del sistema. Todas las decisiones técnicas deberán estar debidamente justificadas en el documento final.

### 2.2. Enfoque metodológico

El proyecto deberá combinar fundamentos conceptuales con implementación práctica. No se trata únicamente de generar imágenes, sino de diseñar un sistema de toma de decisiones que integre generación, interacción y explicación. El sistema deberá producir exactamente cinco variaciones visuales a partir de una imagen de entrada cargada por el usuario, las cuales serán evaluadas mediante una interfaz gráfica. El usuario deberá aceptar o rechazar cada una de las opciones generadas, y estas decisiones deberán ser registradas de manera estructurada.

Posteriormente, un agente basado en modelos de lenguaje deberá generar un resumen textual que describa explícitamente qué opciones fueron aceptadas y cuáles fueron rechazadas. Este resumen deberá ser coherente con las decisiones registradas y reflejar fielmente el proceso de selección realizado por el usuario.

### 2.3. Niveles de implementación

El proyecto contempla tres niveles de complejidad técnica:

- **Nivel básico:** uso de modelos generativos preentrenados o APIs (por ejemplo, Stable Diffusion vía API, DALL·E u otros servicios) para producir variaciones visuales a partir de la imagen proporcionada por el usuario.
- **Nivel intermedio:** uso de modelos generativos controlables o técnicas de transformación de imágenes que permitan modificar atributos específicos, tales como estilo, iluminación o composición. Se espera el uso de herramientas como Stable Diffusion con img2img, ControlNet o técnicas similares. Se sugiere explorar modelos preentranados como los disponibles en Hugging Face o Civit AI.
- **En el nivel avanzado**, los estudiantes podrán implementar o adaptar modelos generativos propios, explorando enfoques como variantes de GANs (por ejemplo, DCGAN, StyleGAN), modelos de difusión (Diffusion Models), o arquitecturas de traducción de imagen a imagen como pix2pix, CycleGAN o ControlNet. Se espera que los estudiantes analicen los desafíos asociados al entrenamiento, control de generación y estabilidad de estos modelos, así como su impacto en la calidad y diversidad de las variaciones generadas.

Todos los proyectos deberán incluir interacción humano-IA, procesamiento de una imagen de entrada proporcionada por el usuario y generación de explicaciones mediante un agente basado en modelos de lenguaje. Los estudiantes deberán elegir 2 de los 3 enfoques para implementar en el proyecto, de modo que puedan funcionar como alternativas de respaldo o permitir que el usuario elija cómo generar la imagen. Asimismo, se sugiere contar con más de un modelo de lenguaje natural para comparar los resultados.

### 2.4. Selección de datasets

Cada grupo deberá seleccionar un dataset adecuado para el problema, justificando su elección en función de los objetivos del sistema, la calidad de los datos y las consideraciones éticas. Se sugiere el uso de datasets de rostros, ya que estos permiten explorar de manera natural variaciones visuales relevantes como iluminación, expresión o estilo, y facilitan la evaluación de resultados. Sin embargo, los grupos también podrán utilizar datasets de otras categorías, como objetos, escenas o elementos del entorno, siempre que el problema planteado sea coherente con la generación de variaciones visuales y la posterior toma de decisiones.

En todos los casos, se deberá considerar el uso responsable de los datos, evitando información sensible o privada sin consentimiento explícito, y reflexionando sobre posibles implicancias éticas del sistema desarrollado. Debe incluir también mecanismos para minimizar estos riesgos.

### 2.5. Requisitos del sistema

El sistema desarrollado deberá incluir un módulo generativo capaz de producir cinco variaciones visuales por cada imagen de entrada, una interfaz gráfica que permita visualizar dichas variaciones y registrar las decisiones del usuario, y un agente basado en modelos de lenguaje que genere un resumen final de las decisiones tomadas. Este resumen deberá indicar de forma clara y explícita qué opciones fueron aceptadas y cuáles fueron rechazadas.

### 2.6. Evaluación del sistema

Los grupos deberán definir métricas que permitan evaluar el desempeño del sistema en términos de calidad visual, diversidad de las variaciones generadas, coherencia entre las decisiones del usuario y el resumen producido por el agente, y usabilidad de la interfaz. Se valorará especialmente la capacidad de analizar críticamente el comportamiento del sistema y sus limitaciones.

### 2.7. Consideraciones éticas

El proyecto deberá incluir una reflexión explícita sobre aspectos éticos relacionados con la generación de imágenes, la posible manipulación de identidad visual, la presencia de sesgos en los datos o en los modelos, y el impacto del uso de estos sistemas en contextos reales. Esta reflexión será parte integral de la evaluación del proyecto.

## 3. Actividades del proyecto

El desarrollo del proyecto se organizará en cinco etapas principales que reflejan el ciclo de diseño, implementación, interacción y evaluación de un sistema generativo interactivo.

### 3.1. Definición del problema y diseño del sistema

En la primera etapa, cada grupo deberá definir claramente el problema que abordará el sistema, especificando el contexto de uso, los objetivos y las restricciones relevantes. Se espera que los estudiantes planteen un caso de uso coherente con la generación de variaciones visuales y su posterior selección por parte de un usuario.

Asimismo, deberán proponer el diseño general del sistema, describiendo cómo se integran sus componentes principales, incluyendo el módulo generativo, la interfaz gráfica, el mecanismo de interacción con el usuario y el agente basado en modelos de lenguaje. Este diseño deberá representarse mediante un diagrama que muestre el flujo de información y las decisiones dentro del sistema.

### 3.2. Selección y preparación de datos

En la segunda etapa, los grupos deberán seleccionar el dataset a utilizar y justificar su elección en función del problema planteado. Se espera que realicen un análisis exploratorio que permita comprender las características de los datos y su adecuación para la generación de variaciones visuales.

Además, deberán definir y aplicar las técnicas necesarias de preprocesamiento, tales como normalización, redimensionamiento o filtrado de imágenes, de modo que los datos puedan ser utilizados correctamente por el modelo generativo. En esta etapa también deberán considerar aspectos relacionados con la calidad de los datos y posibles implicancias éticas.

### 3.3. Implementación del modelo generativo

En la tercera etapa, los estudiantes deberán implementar el módulo generativo responsable de producir exactamente cinco variaciones visuales a partir de cada imagen de entrada proporcionada por el usuario. La implementación dependerá del nivel elegido por el grupo, pudiendo incluir el uso de modelos preentrenados, modelos generativos controlables o la construcción de un modelo propio.

Se espera que los estudiantes analicen el comportamiento del modelo, evaluando la calidad y diversidad de las imágenes generadas, así como las limitaciones observadas. Esta etapa deberá incluir una justificación de las decisiones tomadas en relación con la generación de imágenes.

### 3.4. Desarrollo de la interfaz e interacción humano–IA

En la cuarta etapa, los grupos deberán desarrollar una interfaz gráfica que permita visualizar las cinco variaciones generadas y registrar las decisiones del usuario. La interfaz deberá ser clara y funcional, facilitando la evaluación de cada imagen y permitiendo su aceptación o rechazo.

Asimismo, el sistema deberá almacenar de manera estructurada las decisiones tomadas por el usuario, de modo que puedan ser utilizadas posteriormente en el proceso de generación de explicaciones. En esta etapa, se busca enfatizar la integración efectiva entre el usuario y el sistema de IA.

### 3.5. Generación de explicaciones y evaluación del sistema

En la quinta etapa, los estudiantes deberán integrar un agente basado en modelos de lenguaje que genere un resumen textual de las decisiones tomadas por el usuario. Este resumen deberá indicar explícitamente qué variaciones fueron aceptadas y cuáles fueron rechazadas, manteniendo coherencia con la información registrada.

Finalmente, los grupos deberán evaluar el sistema en su conjunto, analizando la calidad de las imágenes generadas, la utilidad de la interfaz, la coherencia de las explicaciones y el comportamiento general del sistema. Se espera una reflexión crítica que identifique fortalezas, limitaciones y posibles mejoras, incluyendo consideraciones técnicas, de interacción y éticas.

## 4. Producto esperado

El proyecto deberá culminar con la entrega de un sistema generativo interactivo funcional, acompañado de la documentación técnica correspondiente y materiales de apoyo que permitan comprender, reproducir y evaluar el trabajo desarrollado.

- **Código del sistema:** implementación completa del sistema generativo interactivo, incluyendo el módulo de generación de variaciones visuales, la interfaz gráfica, el registro de decisiones del usuario y el agente basado en modelos de lenguaje. El código deberá estar organizado, documentado y ser reproducible, con instrucciones claras para su ejecución.
- **Documento técnico:** informe de un máximo de diez páginas que describa de manera estructurada todas las etapas del proyecto. Deberá incluir la definición del problema, el diseño del sistema, la justificación de las decisiones técnicas, la descripción del modelo generativo utilizado, la implementación de la interfaz y el análisis de resultados. Asimismo, deberá incorporar una reflexión crítica sobre limitaciones y consideraciones éticas.    
- **Elementos visuales:** inclusión de diagramas de arquitectura, representaciones del flujo del sistema, ejemplos de imágenes generadas y evidencia de la interacción con el usuario. Estos elementos deberán integrarse de manera coherente dentro del documento.
- **Demostración del sistema:** evidencia del funcionamiento del sistema, que deberá presentarse mediante un video grabado y una presentación en vivo, en los que se observe claramente la generación de cinco variaciones visuales por cada imagen de entrada, la interacción del usuario mediante la aceptación o el rechazo de cada opción y la generación automática del resumen final por parte del agente.

El video, al permitir la edición para eliminar los tiempos muertos, deberá tener una duración de 3 minutos o menos. La presentación en vivo deberá durar 6 minutos o menos.

La grabación del video deberá realizarse con un docente de la UTEC o un estudiante de una carrera de pregrado distinta de Ciencia de Datos o Ciencia de la Computación. Se recomienda que el video se centre principalmente en la captura de pantalla del funcionamiento del sistema. Únicamente al inicio se deberán dedicar 10 segundos para que el docente o estudiante se presente y brinde su consentimiento para participar en la grabación.
