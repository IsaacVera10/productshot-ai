# Interfaces gráficas

El sistema tiene dos interfaces. Ambas se sirven desde el mismo proceso que la API (`uv run productshot`) y hablan con `/api/v1` por HTTP, sin tocar la base de datos.

| Interfaz | URL | Código | Para quién |
|---|---|---|---|
| Web app | `/app` | [`backend/productshot/web/`](../backend/productshot/web/) | La persona que vende. Pensada para el celular: puede tomar la foto con la cámara. Usa HTML, CSS y JS sin dependencias, con tema claro y oscuro |
| Panel Gradio | `/ui` | [`productshot_ui/app.py`](productshot_ui/app.py) (este paquete) | Demostración y moderación de la prueba con usuarios. Expone todos los controles, incluidas la máscara manual y la vista previa del plan de la siguiente ronda |

## Flujo

1. Subir una foto (o tomarla, en la web app), elegir la categoría del producto y el modo (intermedio o avanzado).
2. Segmentar el producto. En Gradio se puede reintentar la segmentación o subir una máscara PNG (blanco = fondo, negro = producto).
3. Generar las cinco variaciones. La web app muestra el progreso y avisa si el worker está arrancando en frío.
4. Aceptar o rechazar cada variación, con un motivo opcional. En la web app cada decisión se guarda al marcarla.
5. Ver el resumen del agente. A partir de ese momento las decisiones de la ronda quedan congeladas.
6. Opcional: escribir una nota de ajuste y crear la siguiente ronda (hasta 3). En Gradio, «Ver qué se entendió» muestra el plan antes de crearla.
7. Borrar la sesión, con todas sus rondas y fotos.

## Garantías

- **Resultados simulados:** si el servidor usa el mock, cada imagen lleva la franja «SIMULACIÓN» y la interfaz lo indica.
- **Resumen:** se muestra como texto (la web app lo escapa). Los motivos que escribe la persona nunca se interpretan como Markdown ni HTML.
- **Autoría del resumen:** la web app indica si lo escribió el agente y fue verificado (`llm_validated`, `hybrid`) o si es el resumen automático de respaldo (`deterministic_fallback`).
- **Doble clic en «Generar» (Gradio):** reutiliza la misma `Idempotency-Key`, así que no crea un segundo trabajo.
- **Cambios sin guardar:** «Generar resumen» guarda antes los cambios pendientes del formulario. Esto lo cubre `tests/test_ui.py`.
