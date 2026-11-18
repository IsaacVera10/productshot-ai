# Criterios de la revisión del piloto de BG60k

Se aplican en `training/bg60k_pilot_review.py` (docs/10). Cada estilo describe el entorno del producto, no el producto, y se acepta solo lo que coincide con su preset (`backend/productshot/services/presets.py`), porque el caption de entrenamiento parte de ese texto.

| Estilo | Se acepta | Colores |
|---|---|---|
| Estudio | Fondo continuo liso, luz suave de estudio, sombra suave, sin escena | Gris claro neutro (también blanco grisáceo o gris cálido muy suave). Los fondos de estudio de color saturado o pastel y los oscuros no cuentan |
| Natural | Superficie natural (piedra, roca, madera rústica) en exterior, luz de día, vegetación desenfocada | Verdes, tonos de piedra y madera |
| Cálido | Superficie de madera, luz dorada, bokeh suave | Ámbar, dorado, naranja suave, marrones |
| Lifestyle | Cocina, comedor o sala de una casa luminosa, algunos objetos de fondo, profundidad de campo baja | Tonos claros de interior |
| Minimalista | Fondo plano, mucho espacio vacío, sin accesorios, una sombra suave | Blanco roto |

Motivos de rechazo, sea cual sea el estilo: no corresponde al estilo (incluido un color fuera de la paleta), transparente o reflectante (no mate), más de un producto, comida servida o sin envase, texto o logo grande en el fondo, otro problema.

Ante la duda, se rechaza: los candidatos sobran y un ejemplo dudoso enseña mal.

Decisión de Isaac del 2026-11-15: estudio se limita a fondos neutros claros. Abrirlo a estudios de color exigiría captions con el color real de cada imagen.
