# UI (Gradio)

Paquete `productshot_ui`. Se monta en `/ui` del mismo servidor y habla con la API `/api/v1` por HTTP, sin tocar la base de datos.

Flujo: subir foto y elegir modo → crear sesión y segmentar (con reintento) → generar 5 variaciones → aceptar o rechazar cada una, con motivo opcional → guardar → generar resumen.

- Los resultados simulados llevan la etiqueta «SIMULADA» y la franja roja en la imagen.
- El resumen se muestra como texto plano: los motivos del usuario nunca se renderizan como Markdown o HTML.
- Un doble clic en «Generar» reutiliza la misma `Idempotency-Key` y no crea un segundo trabajo.
