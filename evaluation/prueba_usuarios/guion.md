# Guion de la prueba con usuarios

Protocolo de `docs/06_EVALUACION_Y_ETICA.md`, sección «Prueba con usuarios». Una sesión dura entre 15 y 20 minutos. Quien modera lee en voz alta lo que está entre comillas y no ayuda durante las tareas.

## Antes de empezar el día

- [ ] `.env` con `GENERATION_BACKEND=remote`, `SEGMENTATION_BACKEND=remote`, `SCREENING_BACKEND=remote` y `SUMMARY_BACKEND=gemini` con la clave propia.
- [ ] Agente: `GEMINI_MODEL_ID=gemini-3.5-flash` y `GEMINI_FALLBACK_MODEL_ID=gemini-3.5-flash-lite`. Los modelos más nuevos (3.7 y 3.8) respondían 503 (saturados) el 2026-11-16; si el resumen sale como `deterministic_fallback`, revisar cuál responde.
- [ ] La app corriendo: `uv run productshot`, en `http://127.0.0.1:8000/ui`.
- [ ] Productos de respaldo a mano (vela, taza, caja con texto), por si alguien no trae uno.
- [ ] Hoja de registro abierta: `registro.csv`, una fila por participante.
- [ ] Cronómetro.

## Antes de cada participante

- [ ] Calentar el worker unos 2 minutos antes, para que la persona no espere el arranque en frío: `uv run python evaluation/prueba_usuarios/calentar.py`.
- [ ] Asignar el código (P01, P02, …) y el modo: el intermedio para todos si el avanzado no está disponible. Anotarlo en el registro.

## 1. Consentimiento (antes de mostrar la pantalla)

«Gracias por ayudarnos. Estamos probando una herramienta de un curso de la universidad, no a ti: si algo es confuso, es un problema de la herramienta.

- Vas a subir la foto de un producto. La foto se usa solo para esta prueba y se borra al terminar el proyecto, salvo que nos autorices a guardarla.
- Tus decisiones y los motivos que escribas, nunca la foto, se envían al servicio Gemini de Google para escribir un resumen. En su plan gratuito, Google puede usar ese texto para mejorar sus productos.
- Anotamos tiempos, errores y tus respuestas con un código, sin tu nombre. No grabamos tu cara.
- Puedes parar cuando quieras, sin dar explicaciones.

¿Estás de acuerdo en participar?»

Anotar en el registro `consentimiento = si`, y si autoriza a guardar la foto. Sin consentimiento, la sesión no se hace.

## 2. Contexto

«Imagina que vendes este producto por Instagram y quieres mejores fotos para publicarlo. Mientras usas la herramienta, dime en voz alta lo que piensas.»

Pedir la foto: puede tomarla con su celular en ese momento o elegir uno de los productos de respaldo.

## 3. Tareas

Al iniciar cada tarea, empezar el cronómetro; detenerlo cuando la termine. La espera de generación se anota aparte. Si queda bloqueada más de 2 minutos, se le ayuda y la tarea cuenta como **no completada**. Después de cada tarea, preguntar la SEQ:

> «Del 1 al 7, ¿qué tan fácil o difícil fue esta tarea? (1 = muy difícil, 7 = muy fácil)»

| Tarea | Lo que se le dice | Termina cuando |
|---|---|---|
| **U1** | «Sube la foto de tu producto y prepara la generación de las fotos.» (Si pregunta por el modo, indicarle el asignado.) | Ve la máscara lista y pulsa «Generar» |
| **U2** | «Revisa las cinco fotos. Acepta las que publicarías y rechaza las que no; en al menos una, escribe por qué.» | Guardó las cinco decisiones |
| **U3** | «Pide el resumen y léelo.» Luego: «¿Refleja lo que decidiste? ¿Por qué?» | Leyó el resumen y respondió |
| **U4** | «Pide otra ronda de fotos tomando en cuenta lo que decidiste.» Al ver la nueva ronda: «¿Estas fotos tienen en cuenta lo que elegiste antes? ¿Por qué?» | Ve la segunda ronda y respondió |

Anotar también:

- Los **errores**: clic equivocado, retroceso o una pregunta a quien modera.
- La **espera percibida**: «Del 1 al 5, ¿el tiempo de espera fue aceptable?»
- Cualquier problema, con su gravedad: bloquea, molesta o es cosmético.

## 4. Cuestionario SUS (al final)

«Para terminar, dime qué tan de acuerdo estás con cada frase, del 1 (totalmente en desacuerdo) al 5 (totalmente de acuerdo).»

1. Creo que me gustaría usar esta herramienta con frecuencia.
2. La herramienta me pareció innecesariamente complicada.
3. La herramienta me pareció fácil de usar.
4. Creo que necesitaría ayuda de una persona con conocimientos técnicos para usarla.
5. Las distintas funciones de la herramienta están bien integradas.
6. Hay demasiadas inconsistencias en la herramienta.
7. Imagino que la mayoría de las personas aprendería a usarla muy rápido.
8. La herramienta me pareció muy incómoda de usar.
9. Me sentí muy seguro(a) usando la herramienta.
10. Necesité aprender muchas cosas antes de poder usarla.

## 5. Preguntas abiertas

1. «¿Qué fue lo más confuso?»
2. «¿Qué cambiarías?»
3. «¿Usarías estas fotos o la herramienta para tus ventas? ¿Por qué?»

## Al terminar

- [ ] Borrar la sesión desde la API (`DELETE /api/v1/sessions/{id}`), salvo que haya autorizado guardar la foto.
- [ ] Revisar que la fila del registro esté completa.

Cuando estén las diez filas: `uv run python evaluation/prueba_usuarios/analizar.py evaluation/prueba_usuarios/registro.csv`.

## Video

La persona de otra carrera que pide el enunciado puede ser una de las participantes. Al inicio del video (unos 10 segundos) dice su nombre, su carrera y que acepta participar en la grabación. Después, la grabación muestra sobre todo la pantalla: las cinco fotos, las decisiones y el resumen.
