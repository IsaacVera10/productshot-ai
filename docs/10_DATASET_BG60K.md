# 10: Curación de BG60k para el LoRA del modo avanzado

**Estado (2026-11-15):** protocolo acordado y revisado con Codex en dos rondas. Están hechas la etapa 1 (manifiesto) y la preclasificación de las etapas 2 y 3 (nicho por categoría y alertas de máscara; resultados abajo, revisados con Codex). Las demás etapas siguen pendientes. Cumple el Hito 4 de `04_ROADMAP_Y_TESTS.md` y alimenta la etapa `train` de `training/README.md`.

## Qué hay en la descarga

| Dato | Valor | Cómo se comprobó |
|---|---|---|
| Zip descargados | 19 de 22, desde el JD Whale Disk del README de BG60k | Directorio central legible; el manifiesto lee cada miembro completo, así que también comprueba el CRC |
| Zip faltantes | `bg60k_imgs_2`, `bg60k_masks_1`, `bg60k_masks_4` | Lista de archivos |
| Reparto | Shard `i` = imágenes con `id mod 10 == i` | Primeros y últimos ids de cada zip |
| Pares imagen y máscara completos | 44 304 de 63 293 (70 %), shards 0, 3, 5, 6, 7, 8 y 9 | `bg60k_info.txt` cruzado con los shards |
| Formato | PNG 800×800 RGB; máscara PNG RGBA 800×800 | Muestra de los ids 10, 20, 30 y 40 |
| Máscara | Es un recorte: el **canal alfa** es el matte suave del producto y el RGB es el producto recortado (0 fuera del alfa) | Revisado en cuatro muestras del shard 0 (Claude y Codex); el manifiesto lo verifica en todos los pares (`mask_rgb_err`) |
| Categorías | 1 981 en total, 1 880 con pares disponibles; muy desbalanceadas (cell phone 1 427, Men casual shoes 1 304, Liquor 1 268) | `bg60k_info.txt` |
| `bg1k` y `bgpair_info.txt` | Pares de referencia de productos **distintos** de la misma categoría (cuatro refrigeradoras diferentes); no sirven para identidad de producto y quedan fuera del entrenamiento | Revisión visual de bg1k 0, 1, 3 y 4 |

**Efecto de los shards faltantes.** En las 136 categorías con al menos 100 imágenes, la fracción disponible va de 0,624 a 0,809 (mediana 0,700). Esas desviaciones son compatibles con una selección aleatoria del mismo tamaño (Codex: 10 000 selecciones, semilla 20261115, p ≈ 0,21). No demuestra ausencia de sesgo en categorías raras, en estilos ni después de curar. El orden de los ids **no es aleatorio**: el 21,9 % de los ids consecutivos comparte categoría, frente al 0,4 % esperado al azar.

## Etapas

| # | Etapa | Entrada | Salida | Dónde corre |
|---|---|---|---|---|
| 1 | Manifiesto: metadatos, máscara, calidad y hashes de cada par | Zip de `data/raw/bg60k` | `data/bg60k/manifest.parquet` y `manifest_summary.json` | Laptop, CPU |
| 2 | Nicho: excluir categorías fuera del alcance de `docs/00` (productos grandes, ropa en personas, transparentes, muy reflectantes, alimentos preparados) | Manifiesto y reglas de `training/bg60k_categories.toml` | Columnas `niche` y `niche_rule`; `training/bg60k_categories_review.csv` | Laptop |
| 3 | Máscara: alertas geométricas (componentes, área, bordes tocados, fondo visible) | Manifiesto | Columna `alerts` | Laptop |
| 4 | Identidad y duplicados: DINOv2 ViT-B/14 sobre el recorte del producto (con padding a cuadrado, sin recorte central) y pHash del recorte; grupos indivisibles | Candidatos de 2 y 3 | `groups.parquet` y auditoría | Laptop, GPU |
| 5 | Estilo: CLIP contra cinco descripciones paralelas a los presets, con paráfrasis y abstención («ninguno de los cinco»); propone, no decide | Candidatos | Puntajes y margen por estilo | Laptop, GPU |
| 6 | OCR del fondo (texto fuera del producto) | Candidatos de estilo | Alerta `bg_text` | Laptop |
| 7 | Piloto: 150 por estilo, cuotas contadas por grupo y tope por categoría, revisado a mano en una grilla HTML | 4, 5 y 6 | `pilot.parquet` con decisión humana | Laptop |
| 8 | Captions: preset del estilo más una descripción breve del producto, revisada | Piloto | Columna `caption` | Laptop |
| 9 | Split por grupo, 80/10/10 aproximado, priorizando la integridad de los grupos y la cobertura por estilo; auditoría entre particiones antes de congelar | Piloto y grupos | `split` congelado | Laptop |
| 10 | Exportar **solo train** al imagefolder (`metadata.jsonl`); val y test van aparte | Split | `dataset.tar` | Laptop |

## Reglas acordadas

- **Umbrales:** los de máscara empiezan como alertas y se calibran con revisión visual. La nitidez no se usa como filtro global, porque `warm` y `lifestyle` piden desenfoque de fondo (`presets.py`); solo se mide la del producto.
- **Estilo:** la etiqueta final es humana. Además de los ejemplos con margen alto, la revisión incluye márgenes bajos. La evaluación del LoRA no se apoya solo en el mismo CLIP que eligió los ejemplos: la medida principal es humana y ciega (`06_EVALUACION_Y_ETICA.md`).
- **Identidad:** BG60k no trae identidad de producto. Los grupos salen de aristas por pHash del recorte y por coseno de DINOv2 ≥ τ, unidas por union-find. τ se calibra con pares revisados a mano en bandas de similitud, con pares de mismo producto con cambios grandes y con productos distintos muy parecidos. Como union-find encadena (A~B y B~C unen A con C), se revisan los componentes grandes y las aristas que conectan subgrupos. Un grupo nunca se divide para cumplir cuotas. Las variantes de color de un mismo modelo se agrupan como familia cuando se reconocen.
- **Afirmación permitida en el informe**: «piloto seleccionado por revisión humana con un radio de exclusión de identidad visual; segunda pasada sobre las aceptadas y auditoría de pares entre particiones realizadas por un asistente de IA sobre miniaturas, sin revisión humana de los pares; los casos dudosos se trataron como coincidencias; persiste riesgo de variantes no detectadas». Reemplaza a la anterior, que suponía una auditoría hecha por una persona (ver «Resultados del piloto, el split y la exportación»).
- **Equilibrio:** las categorías se reparten entre estilos para que el LoRA no asocie un estilo con un tipo de producto.
- **Pruebas:** el test de BG60k es estético y reservado. Las fotos caseras se dividen en desarrollo (para elegir checkpoint, `strength`, escala y duración) y prueba final, que no se usa para ajustar nada.

## Entrenamiento (ajustes en `training/`)

- Hecho: `colab_job.py` ya no usa `--random_flip` (voltea textos y logos), guarda hasta 8 checkpoints y el resultado trae los pesos de todos, para validarlos por estilo. `run_colab.py` sube el dataset en partes de 32 MiB y el VM verifica su sha256.
- Primera corrida: 1200 pasos sobre 209 imágenes de train (unas 5,7 épocas con lote 1), con checkpoints en 300, 600, 900 y 1200; es un presupuesto de costo, no un óptimo. La duración se elige validando checkpoints por estilo, no a priori. Rank 16 y LR 1e-4 son el punto de partida del smoke test, no un óptimo.
- Ampliar `--max-minutes` (hoy 50, con 10 de reserva): por el smoke test (200 s por 100 pasos), 1500 pasos son unos 50 minutos solo de entrenamiento.
- 1024 px por continuidad con el smoke test: el script oficial reescala con interpolación bilineal (800 → 1024, 1,28×) y recorta al cuadrado. Si el LoRA aprende desenfoque, se mide; no se supone.
- Pérdida estándar en el piloto. La pérdida enmascarada al fondo queda como ablación si aparecen asociaciones indeseadas con productos (el script oficial usa MSE global).
- Si un estilo queda sin cobertura, el backend necesita escala del LoRA por estilo (hoy se fija por modo antes del bucle de estilos en `sdxl_generation.py`).

## Etapa 1: manifiesto

```bash
uv run --group dataset python training/bg60k_manifest.py --raw data/raw/bg60k --out data/bg60k
```

Una fila por par completo, con estas columnas:
- **origen:** shard, zip y miembro, y sha256 de la imagen y de la máscara;
- **imagen:** tamaño y modo;
- **máscara (alfa > 127):** área del producto, fracción de borde suave, componentes que superan el 1 % de la imagen y fracción del componente mayor, bbox y bordes que toca, centroide;
- **verificación de la máscara:** error medio entre el RGB de la máscara y la imagen dentro del producto (`mask_rgb_err`) y RGB fuera del producto;
- **brillo:** del fondo y del producto, y desviación del fondo como medida de textura;
- **nitidez:** varianza del laplaciano dentro del producto;
- **hashes:** pHash de la imagen completa y del recorte del producto sobre gris neutro;
- **estado:** `status` con el error si un archivo no se puede leer.

`manifest_summary.json` guarda los conteos, las distribuciones y las versiones de las herramientas. `data/` no se versiona.

### Resultado de la etapa 1 (2026-11-15)

Los 44 304 pares se leyeron completos, con el CRC verificado y sin errores. Todas las imágenes son de 800×800 y todas las máscaras son RGBA. En el 100 % de los pares, el RGB de la máscara vale 0 fuera del producto. Dentro del producto coincide con la imagen: el error medio es 0,002 de 255 en la mediana, 0,079 en el percentil 99 y llega a 16 en el peor par, que hay que revisar. La lectura del canal alfa queda confirmada en todos los shards.

| Medida (alfa > 127) | Resultado | Qué implica para las etapas siguientes |
|---|---|---|
| Componentes que superan el 1 % de la imagen | 1: 37 002 (83,5 %) · 2: 5 362 · 3 o más: 1 934 · 0: 6 (no están vacías: su producto ocupa entre el 1,6 % y el 9,2 %, pero en piezas menores al 1 % de la imagen) | Varios productos o máscaras partidas en el 16 %: alerta de la etapa 3 |
| Bordes tocados por el bbox | 0: 35 553 (80,2 %) · 1: 4 669 · 2: 2 734 · 3: 903 · 4: 445 | Producto posiblemente recortado por el encuadre en el 20 % |
| Área del producto | p1 5,3 % · p5 10,4 % · mediana 31,3 % · p95 61,4 % · p99 77,4 % · máx. 100 % | Fondo visible escaso en la cola alta; las máscaras del 100 % se revisan |
| Brillo del fondo (0 a 255) | p5 22,5 · mediana 163,9 · p95 237,9 | Hay fondos casi negros: relevantes para el estilo y para G12 |
| Textura del fondo (desviación estándar) | p5 6,8 · mediana 34,3 · p95 70,3 | Fondos lisos de catálogo junto a escenas con textura |
| Imágenes idénticas (mismo sha256) | 1 547 repetidas | Duplicados exactos: se agrupan en la etapa 4 |
| Mismo pHash del recorte del producto | 3 007 repetidas | El mismo producto aparece con varios fondos: la agrupación por identidad es necesaria |

Corrida en la laptop con 1 proceso, CPU limitada al 60 % de un núcleo (cgroup), prioridad mínima y pausa automática a 80 °C. Con más procesos la laptop llegó a 100 °C y llegó a apagarse. Tardó unos 28 minutos de cómputo efectivo.

## Etapas 2 y 3: nicho y alertas de máscara

```bash
uv run --group dataset python training/bg60k_filters.py
uv run --group dataset python training/bg60k_filters.py --sheet truncated --n 24 --seed 0 --out-sheet hoja.jpg
```

Es una **preclasificación**: no descarta nada ni demuestra que cada imagen cumpla el nicho. Escribe `data/bg60k/candidates.parquet` (el manifiesto más `niche`, `niche_rule` y `alerts`) y `filters_summary.json`, con los hashes del manifiesto y de las reglas y los umbrales usados. También escribe `training/bg60k_categories_review.csv`, versionado, con el nivel, la regla y los conteos de cada categoría.

**Nicho por categoría** (`training/bg60k_categories.toml`). Primero se aplican las excepciones exactas y después las palabras clave, con límite de palabra y plural opcional. Si coinciden varias, gana la frase más específica, la que contiene a la otra («tv accessories» frente a «tv»); si no, `out` gana a `near`, y `near` a `core`.

| Nivel | Qué incluye | Pares | Sin alertas geométricas |
|---|---|---|---|
| `core` | Pequeño, rígido y opaco **por categoría**: tazas, teteras, cosméticos, decoración, electrónica pequeña, papelería, utensilios | 14 912 | 9 932 |
| `near` | Tamaño parecido, pero blando, transparente, reflectante, joyería o alimento que suele mostrarse sin envase | 15 806 | 9 459 |
| `out` | Grande, ropa puesta, personas, alimentos frescos o preparados, plantas y animales, lo que no es un producto | 10 395 | 7 230 |
| `unclassified` | Nombre ambiguo (`notebook`, `Early childhood education`, `object`) | 3 191 | 2 055 |

El nicho de `docs/00` pide superficies mates o semimates, y la categoría no lo garantiza: teléfonos, laptops y lentes, por ejemplo, tienen pantallas o vidrios. El material, el producto único y la ausencia de comida servida se verifican **imagen por imagen** en la revisión del piloto (etapa 7). Prueba de ello: en una muestra de `core` sin alertas aparecieron un frasco transparente (36596), dos fundas de teléfono (20403), varias vistas del mismo teléfono (43299) y un plato servido (57945).

**Revisión de las reglas.** Se revisó a mano la asignación de las ~170 categorías más frecuentes de cada nivel, unas 510 categorías que cubren más del 80 % de los pares. Las correcciones quedaron como excepciones con motivo en el TOML, y las de la revisión de Codex como pruebas de regresión en `tests/test_bg60k_filters.py`:
- `shower room` y `Outdoor sun protection clothing` se colaban en `core` por palabras generales;
- `instant food`, plantas y flores estaban en `near` pese a ser exclusiones del nicho;
- los alimentos sin envase estaban en `core`.

Las categorías por debajo de ese corte no se revisaron una por una.

**Alertas geométricas** (sobre el alfa > 127; no descartan):

| Alerta | Regla | Pares |
|---|---|---|
| `truncated` | El bbox queda a 2 px o menos de algún borde | 8 751 |
| `multi_object` | 2 o más componentes de al menos el 1 % de la imagen | 7 296 |
| `fragmented` | El componente mayor tiene menos del 85 % del área | 6 726 |
| `large_product` | Producto de más del 60 % de la imagen | 2 538 |
| `tiny_product` | Producto de menos del 5 % | 388 |
| `mask_mismatch` | El RGB de la máscara difiere de la imagen en más de 1/255 | 203 |

Las hojas de contacto muestran la imagen junto a su superposición (fondo en magenta según el alfa) y devuelven los ids con la semilla, así la revisión se puede repetir. Lo observado hasta ahora es **exploratorio, no una precisión medida**:
- con 2 componentes suelen verse composiciones con accesorios (hervidor con taza, teclado con mouse);
- cuando el producto toca 1 borde, suele ser un cable o un mango que sale del cuadro, o un primer plano;
- un solo componente no garantiza un solo producto, porque dos objetos que se tocan forman un único componente.

**Política provisional** (revisada con Codex):
- Pool principal: `core` sin alertas geométricas. Tiene 9 932 pares, pero hay imágenes repetidas y falta agrupar por identidad, así que el total no prueba que cada estilo llegue a 150.
- Antes de congelar la política se mide la cobertura por estilo, tanto de los candidatos aceptados como de los rechazados, por familia y por alerta, con revisión humana de ambos lados. Ninguna alerta leve se admite automáticamente.
- `near` entra solo con tope fijado antes de seleccionar y declarado en el informe. `out` y `unclassified` quedan fuera.
- Los topes del piloto se cuentan por familias de categorías, por ejemplo teléfonos nuevos y de segunda mano juntos, con un mapa versionado, y se comprueba que la mezcla de familias sea comparable entre estilos.

## Etapas 4 y 5: embeddings en Colab

La GPU de la laptop no se usa, para no recalentarla. El trabajo se reparte así:
- **En la laptop:** se arma un paquete de imágenes reducidas.
- **En el Colab Pro de Pedro:** se calculan los embeddings, con el mismo driver del LoRA, que garantiza un VM único, presupuesto de tiempo y apagado.
- **De vuelta en la laptop:** se analizan los resultados.

```bash
uv run --group dataset python training/bg60k_pack.py
PYTHONPATH=backend python training/run_colab.py --stage embed --pack data/bg60k/pack.tar --max-minutes 30
```

**Paquete** (`training/bg60k_pack.py`). Incluye todos los candidatos `core` y `near`, con y sin alertas, porque la cobertura por estilo se mide en aceptados y rechazados. Deja una copia por imagen repetida y registra el mapa de duplicados. Cada candidato lleva:
- la foto completa a 224 px (Lanczos, JPEG 92), para CLIP;
- solo los candidatos sin alertas: el recorte del producto sobre gris 128, con padding a cuadrado y sin recorte central, para DINOv2;
- 128 originales de referencia, para comparar este preprocesado con el oficial de CLIP.

La firma del preprocesado queda en `preprocess.json`: una corrida con otros parámetros no reutiliza archivos viejos. Las escrituras son atómicas, y al reanudar se valida cada JPEG.

**Trabajo en el VM** (`training/colab_embed.py`):
- **Entrada:** comprueba que el paquete coincida con el sha256 calculado en la laptop. Registra la revisión exacta (SHA) de `openai/clip-vit-large-patch14` y de `facebook/dinov2-base`, y las versiones de torch, transformers y Pillow del VM.
- **CLIP:** usa las proyecciones del modelo directamente, para no depender de lo que devuelva `get_*_features` en cada versión de transformers. Las imágenes se procesan con autocast fp16 y el texto en fp32.
- **Comprobaciones previas:** antes del lote completo prueba 64 imágenes (dimensiones, valores finitos, normas) y compara las 128 referencias contra sus originales.
- **DINOv2:** normalización ImageNet manual sobre el recorte de 224, usando el `pooler_output`.
- **Resultado:** guarda `clip.npz` antes de empezar DINOv2, para no perder CLIP si DINOv2 falla.

El driver exige que `clip.npz`, `dino.npz` e `index.json` coincidan con su sha256. Además valida el contenido: ids iguales a los del índice y sin repetidos, dimensiones, valores finitos y normas.

**Estilo** (`training/bg60k_style_prompts.toml`, revisado con Codex):
- Tres paráfrasis por estilo, paralelas a los presets. Su prototipo es la media de los embeddings de texto normalizados, renormalizada.
- `minimal` insiste en el espacio vacío y en la ausencia de accesorios, para separarse de un estudio claro.
- `other` son subclases independientes (degradado, neón, negro puro, collage, póster, comida, salpicaduras). Funcionan como un veto calibrado por separado, más un umbral absoluto del mejor estilo, porque un margen alto no garantiza que algún estilo encaje.
- CLIP solo propone el estilo. La decisión es humana y ciega.

**Identidad:** DINOv2 base es el punto de partida. Su suficiencia se mide con pares difíciles: el mismo producto con cambios grandes, y productos distintos muy parecidos. Large se prueba sobre ese conjunto antes de recalcular todo. Las aristas por pHash también se calibran, porque pueden fusionar familias aunque DINOv2 esté bien.

### Resultado de los embeddings (2026-11-15)

Corrida `20261115-022838-embed` en una L4 del Colab Pro de Pedro: unos 8 minutos desde que se pidió el VM y **0,23 unidades** (de 217,61 a 217,38). El VM quedó liberado.

- **Subida por partes.** El primer intento falló al subir el paquete (error 500), sin costo apreciable. El CLI de Colab 0.8.0 sube cada archivo en una sola petición JSON en base64, y 792 MB no pasan. Desde entonces el driver lo sube en partes de 32 MB y el VM las une y verifica el sha256.
- **Entorno del VM:** torch 2.14.1+cu130, transformers 5.22.1 (la versión en la que `get_*_features` ya no devuelve un tensor, como anticipó Codex), Pillow 11.3.0 y Python 3.13.
- **Modelos:** `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41` y `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- **CLIP:** 29 600 fotos en 217 s. **DINOv2:** 18 602 recortes en 51 s. Ambos de 768 dimensiones, con valores finitos y normas entre 0,9999998 y 1,0000001.
- **Preprocesado frente al oficial.** Se comparó el embedding CLIP de la foto reducida del paquete (Lanczos más JPEG 92) con el del PNG original pasado por el procesador oficial. En 128 referencias, el coseno dio media 0,949, percentil 5 de 0,915 y mínimo 0,866. La diferencia no es despreciable, y puede mover decisiones de estilo cercanas a un umbral. Esta corrida guardó solo los cosenos, no los embeddings de los originales: para medir si cambia el estilo asignado, la próxima corrida debe guardar ambos.

### Primera lectura de estilo e identidad (2026-11-15, exploratoria)

`training/bg60k_style.py` y `training/bg60k_identity.py` trabajan sobre la corrida `20261115-022838-embed`. Nada de esto es todavía una decisión: falta la calibración humana.

**Estilo.** Con el puntaje crudo, `studio` se lleva el 65 % de `core` sin alertas: casi cualquier foto de producto le parece de estudio a CLIP. La mediana del margen entre los dos mejores estilos es 0,018. Además, la subclase «fondo negro puro» supera al mejor estilo en 5 478 imágenes. Se estandariza cada estilo contra su propia distribución (z-score) y se ordena por ese valor.

| `core` sin alertas | studio | lifestyle | natural | warm | minimal |
|---|---|---|---|---|---|
| Mejor estilo crudo | 6 076 | 2 004 | 602 | 393 | 288 |
| Mejor estilo estandarizado | 1 288 | 2 858 | 2 148 | 1 424 | 1 645 |

En una revisión visual de los 12 mejores de cada estilo (estandarizado), hecha por Claude y no por una persona:
- `warm` y `lifestyle` se ven bien;
- `natural` es aceptable, aunque muchos fondos son renders 3D de vegetación;
- `studio` trae cortinas pastel en vez del gris continuo del preset;
- `minimal` es heterogéneo.

Estos dos últimos dependen más de la revisión humana o de mejores descripciones.

**Identidad.** La mediana del coseno de DINOv2 con el vecino más cercano es 0,861, y 2 241 recortes tienen un vecino a 0,98 o más. En hojas de pares por banda:
- entre 0,85 y 0,90 se mezclan variantes del mismo producto (color, ángulo) con productos distintos de la misma categoría;
- entre 0,70 y 0,80 todavía aparecen productos idénticos (el mismo juego de tetera, 36305 y 27345, a 0,767).

Un umbral único de DINOv2 no separa identidad de categoría: si es conservador, une categorías enteras por encadenamiento.

Hay una señal complementaria: **ids cercanos**. Entre los vecinos más cercanos, la fracción a 10 ids o menos va de 0,03 (coseno < 0,7) a 0,26 (≥ 0,95), frente a 0,0003 al azar. BG60k parece guardar juntas varias imágenes de una misma publicación. Esa señal cubre solo una parte de los casos, así que sirve como arista adicional y no reemplaza a la revisión.

### Revisión humana de identidad

```bash
uv run --group dataset python training/bg60k_review.py sample       # una vez; no se rehace
uv run --group dataset python training/bg60k_review.py serve --reviewer <nombre>
```

Es una página local en `http://127.0.0.1:8765`: las imágenes de BG60k no salen de la laptop. La muestra (`data/bg60k/review/identity_sample.json`, semilla 0) tiene 320 pares de vecinos más cercanos de DINOv2:
- 40 por banda de coseno: [0,6, 0,7), [0,7, 0,8), [0,8, 0,85), [0,85, 0,9), [0,9, 0,95), [0,95, 0,98) y [0,98, 1];
- 40 pares con pHash del recorte a una distancia de 6 o menos.

Guarda el tamaño de cada estrato para poder ponderar después. La revisión es ciega: la página no muestra el coseno ni la banda, y el orden es aleatorio. Las etiquetas son *mismo producto*, *variante del mismo modelo* (color, talla o empaque), *distinto* y *no sé*. Cada decisión se agrega a `identity_decisions.csv` con la hora y quién revisó; si se corrige un par, vale la última.

Con estas etiquetas se calibran τ de DINOv2, el pHash y la arista por ids cercanos antes de formar los grupos. Las variantes se agrupan como familia (docs/10, reglas).

### Resultado de la revisión de identidad y cambio de protocolo (2026-11-15)

Isaac revisó los 320 registros de la muestra, que son 318 pares únicos (dos estaban en un estrato de coseno y también en el de pHash), y no marcó ningún «no sé». La muestra y las decisiones están versionadas en `training/review/`, sin imágenes; el repo es privado. BG60k no dice nada sobre redistribuir anotaciones derivadas, así que publicarlas fuera del equipo queda pendiente.

| Estrato (coseno DINOv2 del vecino más cercano) | Mismo | Variante | Distinto | Mismo o variante |
|---|---|---|---|---|
| [0,6, 0,7) | 2 | 4 | 34 | 0,15 |
| [0,7, 0,8) | 5 | 3 | 32 | 0,20 |
| [0,8, 0,85) | 4 | 8 | 28 | 0,30 |
| [0,85, 0,9) | 13 | 10 | 17 | 0,57 |
| [0,9, 0,95) | 16 | 6 | 18 | 0,55 |
| [0,95, 0,98) | 18 | 13 | 9 | 0,78 |
| [0,98, 1] | 40 | 0 | 0 | 1,00 |
| pHash del recorte ≤ 6 | 37 | 2 | 1 | 0,97 |

Las tasas de pHash y de ids cercanos mezclan estratos sobremuestreados: describen esta muestra y no son precisiones de la población. Cuarenta de cuarenta tampoco demuestra una precisión perfecta.
- **pHash del recorte < 3:** 76 de 76 registros (74 pares únicos) son mismo producto o variante.
- **Ids a 10 o menos:** 36 de 39. Los tres falsos positivos son de la misma categoría (56507–56506, 11355–11356 y 50097–50099).

Ningún umbral intermedio separa identidad de categoría. Hay además mismos productos muy lejos en DINOv2, como 25487–4260 (coseno 0,633, pHash 34, ids lejanos), y 1 669 pares únicos de vecinos con coseno < 0,6 quedaron fuera de los estratos revisados. Por eso, en vez de agrupar todo el dataset (etapa 4 original):

1. **Selección del piloto con radio de exclusión** (`training/bg60k_pilot.py`, revisado con Codex). Es en línea y por turnos fijos entre estilos (orden studio, natural, warm, lifestyle, minimal). Cada estilo presenta su siguiente mejor candidato por z de CLIP que cumpla dos condiciones:
   - **Topes**, contados sobre aceptados y pendientes: como máximo 30 `near` de los 150 y 4 por familia de producto. Las familias salen de `training/bg60k_families.toml`, que agrupa las categorías que la regla de nicho deja separadas.
   - **Ningún conflicto** con un aceptado o pendiente: misma imagen, coseno DINOv2 ≥ 0,75, pHash del recorte < 7, o ids a 10 o menos de la misma categoría.

   El radio de 0,75 es provisional y no es una garantía: 25487–4260 pasaría igual. Se eligió porque, aceptando todo, los radios de 0,75 a 0,90 alcanzan la misma disponibilidad: 150 por estilo, de 75 a 89 familias por estilo y 30 `near` (17 en lifestyle), con entre 2 681 y 3 734 candidatos aún en cola. Que alcance después de la revisión humana se comprueba al terminarla. El orden de los turnos influye poco (al empezar por otro estilo cambiaron dos imágenes): se fija y se declara.
2. **Revisión humana del piloto** (`training/bg60k_pilot_review.py`, página local). Cada candidato se acepta o se rechaza con un motivo: no corresponde al estilo, no es mate, más de un producto, comida, texto en el fondo u otro. Un rechazo hace que la misma cola presente el siguiente. El estado se reconstruye reproduciendo el registro de decisiones, y el proceso se detiene si cambió la configuración. La procedencia (hashes de candidatos, puntajes, embeddings y familias, más la configuración) queda en `pilot_provenance.json`.
3. **Split y auditoría** (`training/bg60k_split.py`; resultados abajo):
   - **Split:** aleatorio por estilo, con semilla fija.
   - **Auditoría:** para cada imagen de val y test, sus 10 vecinos DINOv2 entre las imágenes de las otras dos particiones juntas (así quedó implementado; calcular por separado los vecinos en train y en la otra partición de evaluación agregaría 315 pares que no se auditaron). Se suma todo par entre particiones que tenga ids a 10 o menos, o pHash < 15 (la **unión** de ambas señales). La revisión es ciega, en la misma página.
   - **Coincidencias:** las confirmadas forman componentes (solo con etiquetas humanas, sin encadenar por similitud) y cada componente va entero a una partición. Se recalcula y se repite hasta no encontrar coincidencias nuevas.
   - **Riesgo residual:** se estima con una muestra aleatoria de pares entre particiones fuera de los auditados, guardando su probabilidad de selección. Con 100 pares y ningún positivo, la cota superior unilateral al 95 % es 2,95 % de los pares, demasiado amplia para afirmar que no hay fuga.

## Resultados del piloto, el split y la exportación (2026-11-15)

**Revisión del piloto.** Isaac tomó 2 706 decisiones (408 aceptadas, 1 990 rechazadas, la gran mayoría por no corresponder al estilo) y la cortó antes de la meta de 150 por estilo: 61 en estudio, 62 en natural, 63 en cálido, 145 en lifestyle y 77 en minimalista. En los últimos minutos el ritmo pasó de unas 18 a más de 100 decisiones por minuto; los rechazos rápidos solo descartan candidatos, pero las aceptadas de ese tramo tenían más errores.

**Segunda pasada sobre las aceptadas (asistida).** Se revisaron las 408 en hojas de miniaturas y se excluyeron 78 que no cumplen los criterios: fondo fuera de la paleta (20), transparente (20), más de un producto (23) y comida o bebida servida (15). Quedan en `training/review/pilot_exclusions.csv`, aparte del registro de decisiones, que no se toca. Isaac no devolvió ninguna.

**Selección.** Las aceptadas menos las exclusiones, recortadas a 60 por estilo en orden de aceptación (solo afecta a lifestyle, que tenía 133; minimalista quedó en 58): 257 imágenes. El orden de aceptación sigue el z de CLIP, así que la selección favorece lo más típico de cada estilo, y el tope absoluto de 30 `near`, pensado para 150, deja proporciones distintas por estilo (15 de 44 en estudio, 5 de 60 en lifestyle).

**Split.** 80/10/10 por estilo con semilla 0. La auditoría tuvo 589 pares: 488 entre particiones (10 vecinos DINOv2 por imagen de val y test, más ids a 10 o menos o pHash < 15) 100 pares al azar de los 10 487 restantes y 1 par nuevo tras mover componentes. La revisó un asistente de IA (revisor `asistido` en `split_decisions.csv`), no una persona, aunque el protocolo pedía etiquetas humanas: 587 pares distintos y 2 dudosos (1728–45185, pulseras de cuentas; 37868–53718, papel), que se trataron como coincidencias y se movieron enteros a train. De los 100 pares aleatorios, 93 siguen cruzando particiones en el split final y ninguno es coincidencia según esa revisión. No se presenta como cota del riesgo: mide pares, no productos, y no incluye los errores de la propia revisión. Resultado en `training/review/split.csv`:

| Estilo | Train | Val | Test |
|---|---|---|---|
| Estudio | 36 | 4 | 4 |
| Natural | 42 | 5 | 5 |
| Cálido | 36 | 3 | 4 |
| Lifestyle | 48 | 6 | 6 |
| Minimalista | 47 | 5 | 6 |
| **Total** | **209** | **23** | **25** |

Val y test son chicos: sirven para elegir checkpoint y para una evaluación agregada (cada producto de test con los cinco estilos y varias semillas); los resultados por estilo son indicativos.

**Captions y exportación** (`training/bg60k_export.py`). Caption = preset del estilo (el mismo prompt de la generación) + producto en inglés, que sale de la categoría de BG60k reducida a un sustantivo en `training/bg60k_captions.toml` (con correcciones por id). Los captions quedan en `training/review/captions.csv`. `data/bg60k/export/dataset.tar` (solo train, 73 MB, fotos originales sin recomprimir) y `data/bg60k/export/eval/` (val y test con foto y recorte) no se versionan; los sha256 están en `export.json`.

**Primera corrida de entrenamiento** (`20261115-132928-train`, 2026-11-15). L4, 1200 pasos en 1 944 s, VRAM pico 10,7 GB, 1,01 unidades (217,38 → 216,37), VM liberado y 0 sesiones al terminar. Adaptador de 44,5 MB (sha256 `9724411b…`) y checkpoints en 300, 600, 900 y 1200 en `data/lora/20261115-132928-train/` (no versionado). La carga pasó en ambos pipelines (560 capas, error relativo 5,5e-6). En la taza sintética, el LoRA cambia el fondo 23,8/255 en el pipeline de cuatro canales (quita el equipo de estudio que agrega el modelo base) y 0,27/255 en el de nueve canales con `strength` 0,85: es la misma limitación del smoke y no cuenta como falla en `train`. La pérdida por paso es ruidosa (media de los primeros 10 pasos 0,112, de los últimos 10 0,116), como es normal en difusión; no sirve para elegir checkpoint. Falta: evaluar los cuatro checkpoints con val en los cinco estilos, con el LoRA apagado y encendido, y recién después usar test.

**Evaluación de los checkpoints con val** (`20261115-174621-eval`, 2026-11-15; `training/colab_eval.py` y `training/lora_eval_report.py`). Cada producto de val en los cinco estilos, con el LoRA apagado y con cada checkpoint, la misma semilla por producto y estilo, y el pipeline y los parámetros de producción (cuatro canales, `strength` 0,99, 30 pasos, peso 1,0). La entrada es el recorte de BG60k sobre gris 128 (experimento controlado: no incluye errores del segmentador real) y cada imagen se recompone como en G8. El estilo se mide con el margen CLIP sobre la imagen recompuesta (coseno al centroide de train del estilo pedido menos el mejor de los otros), con centroides calculados en el VM; como referencia, las fotos reales de val aciertan su estilo en 16 de 23. La regla de elección se fijó antes de ver los datos. Diseño revisado por Codex.

En L4 cada imagen tardó 14,4 s (no 8 a 10), así que el plazo de 120 minutos alcanzó para 17 de los 23 productos (425 imágenes, orden aleatorio fijo, todos completos y pareados). VRAM pico 12,6 GB. Costo: 3,37 unidades, más 0,35 de un primer intento en el que el CLI perdió la conexión al iniciar `exec` y quedó colgado (el trabajo no llegó a correr; hubo que matar el proceso para que el driver apagara el VM).

| Config | Δ margen vs apagado [IC 95 %] | Gana el estilo pedido | DINOv2 producto (antes de G8) |
|---|---|---|---|
| apagado | | 41/85 | 0,973 |
| 300 | −0,0105 [−0,0152, −0,0063] | 30/85 | 0,976 |
| 600 | −0,0112 [−0,0163, −0,0067] | 31/85 | 0,977 |
| 900 | −0,0103 [−0,0158, −0,0054] | 32/85 | 0,976 |
| 1200 | −0,0087 [−0,0130, −0,0042] | 29/85 | 0,976 |

**Resultado: ningún checkpoint mejora; la regla propone no usar el LoRA.** Todos empeoran el margen con IC bajo 0, sobre todo en lifestyle y warm; en studio y natural la diferencia es cercana a 0. Las imágenes lo confirman: con el LoRA los fondos colapsan a una pared lisa y apagada (verdosa o gris) en casi todos los estilos, y se pierden la cocina de lifestyle y el bokeh de warm. Sin el LoRA aparece el equipo de estudio que ya se había visto, pero los estilos se distinguen. Además, el LoRA agrega partes al producto fuera de la máscara (un pico y un asa a una olla, un casquillo a un peine), que G8 no corrige porque quedan en el fondo. La revisión humana ciega puede revisar esta propuesta, pero con este resultado no vale la pena gastar test en estos checkpoints.

Hipótesis a contrastar antes de otra corrida: el LoRA aprende una estética común de BG60k que se impone sobre el prompt (209 fotos, 5,7 épocas, rank 16, LR 1e-4, peso 1,0 al generar); el sustantivo del producto en los captions podría reforzar el agregado de partes. Opciones baratas: evaluar el checkpoint 300 con peso 0,3 a 0,6 (sin entrenar) o reentrenar con menos pasos o menor LR.

**Segunda ronda, exploratoria: peso 0,5** (`20261115-195014-eval`, 2026-11-15). Los mismos 17 productos, ckpt300 y ckpt1200 con peso 0,5 (Codex propuso probar los dos, porque la diferencia entre ellos a peso 1,0 no era concluyente: +0,0018 [−0,0019, +0,0055]). La fila del LoRA apagado salió idéntica a la de la primera ronda (0,7252, 41/85): la generación es determinista y las dos rondas se pueden comparar. 211 s por producto, 2,11 unidades (212,49 → 210,38), VM liberado.

| Config | Δ margen vs apagado [IC 95 %] | Gana el estilo pedido |
|---|---|---|
| apagado | | 41/85 |
| 300 a 0,5 | −0,0055 [−0,0100, −0,0019] | 38/85 |
| 1200 a 0,5 | −0,0042 [−0,0089, +0,0002] | 38/85 |

Con la mitad del peso el daño se reduce a la mitad, pero ninguna variante mejora; la regla vuelve a proponer no usar el LoRA. En las imágenes, a 0,5 vuelven en parte las escenas (la cocina de lifestyle con ckpt300), pero siguen apareciendo partes agregadas fuera de la máscara (el pico y el asa de la olla en warm, una llama en un peine). Al ser una segunda búsqueda sobre los mismos datos, los IC no se leen como confirmación, solo como orientación.

**Conclusión de esta línea:** el adaptador entrenado en `20261115-132928-train` no se usa; el modo avanzado sigue sin LoRA hasta que otro entrenamiento lo supere en val. Si se reentrena, los cambios a probar son captions sin el sustantivo del producto (sospecha de las partes agregadas), menos pasos o menor LR, y evaluar con este mismo procedimiento antes de abrir test.

**Segundo entrenamiento: captions sin producto y pérdida con máscara** (2026-11-15 y 16). Se probaron las dos hipótesis anteriores con dos corridas que solo difieren en la pérdida, ambas con captions iguales al prompt de inferencia (solo el preset), LR 5e-5, 800 pasos (3,8 épocas), rank 16:

- A (`v2a/20261115-231054-train`): `preset.tar`, pérdida normal. 0,97 unidades.
- B (`v2b/20261115-234005-train`): `preset_masked.tar`, pérdida solo sobre el fondo. `colab_job.py` parchea el script oficial con reemplazos exactos (`MASKED_PATCH`): la máscara del producto, sacada del alfa del recorte de BG60k, sigue el mismo resize y recorte que la foto, se reduce a la resolución latente y el error del producto pesa 0. El job valida antes de entrenar que cada máscara exista, tenga el tamaño de la foto y deje al menos 20 % de fondo (el producto ocupa entre 6 % y 57 %, mediana 22 %). Revisado por Codex.

Ambas cargan bien (560 capas) y cambian el fondo de la taza sintética en cuatro canales (20,9 y 21,7/255). Dos intentos fallaron porque `colab exec` perdió la conexión antes de empezar y quedó colgado (0,24 unidades y nada en el segundo); desde entonces el driver consulta el manifiesto del VM y reintenta `exec` si el trabajo no llegó a empezar.

Evaluación con val, decidida antes de ver los datos para que cupiera en el tiempo: 15 productos, checkpoints 400 y 800 a peso 1,0, y la misma regla. 208 s por producto, unas 2 unidades por corrida. El total de la ronda fue 5,18 unidades (210,14 → 204,96), VM liberado.

| Config | Δ margen vs apagado [IC 95 %] | Gana el estilo pedido | Deriva > 12 |
|---|---|---|---|
| apagado | | 37/75 | 5/75 |
| A 400 | −0,0060 [−0,0095, −0,0024] | 30/75 | 0/75 |
| A 800 | −0,0120 [−0,0159, −0,0085] | 24/75 | 1/75 |
| B 400 | −0,0057 [−0,0092, −0,0019] | 31/75 | 1/75 |
| B 800 | −0,0107 [−0,0142, −0,0070] | 27/75 | 1/75 |

Por estilo, studio y natural mejoran un poco (+0,002 a +0,007) y warm y lifestyle empeoran (−0,011 a −0,039). A y B se ven casi iguales: lifestyle pierde la cocina y queda una pared lisa, warm pierde el bokeh y la madera, y las partes agregadas siguen (el pico y el asa de la olla en warm, una rejilla sobre una caja en lifestyle). La pérdida con máscara no las elimina, así que no vienen de que el LoRA aprenda a dibujar productos. La explicación más probable es que BG60k enseña fotos de catálogo con el producto grande sobre un fondo simple, y ese fondo plano empuja al inpainting de cuatro canales a prolongar la silueta, que es la falla que ya tenía el modelo base (`05_COLAB_Y_GCP.md`).

**Resultado: según la regla fijada, se cierra la línea del LoRA.** Ninguna de las diez configuraciones evaluadas, de tres entrenamientos, mejora el margen de estilo en val, y todas agregan partes al producto. El modo avanzado queda sin LoRA y esto se reporta como resultado negativo medido. Para un trabajo futuro, haría falta un dataset de escenas (no de catálogo) para los estilos warm y lifestyle, o entrenar sobre el UNet de inpainting con el objetivo de inpainting, que el script oficial no cubre.

**Quinta línea: reencuadre, nueve canales y LoRA por autodestilación** (2026-11-16). El modo avanzado pasa a ser reencuadre (`services/layout.py`: el producto al 45 % del alto, apoyado al 85 %) más el UNet de inpainting de nueve canales con `strength` 1,0, porque con 0,99 sobrevivía el gris del lienzo en warm y studio (experimentos `integracion`, `integracion_nueve` e `integracion_strength` en Modal). El cuarto LoRA, de inpainting sobre fotos de BG60k (`v3inpaint`), tampoco mejoró en val (ningún checkpoint con IC sobre 0) y casi no cambia la imagen.

El quinto se entrena por autodestilación: 45 productos de train en los 5 estilos con 2 semillas, con el pipeline del avanzado; de cada producto y estilo se elige la candidata con mayor margen CLIP contra el texto del preset (que gane el estilo pedido y DINOv2 ≥ 0,9), y una revisión visual asistida excluye 56 con equipo de estudio visible o partes agregadas (`training/review/distill_exclusions.csv`). Quedan 131 pares (natural 42, warm 33, lifestyle 28, minimal 16, studio 12) en `distill.tar`. Entrenamiento: objetivo de inpainting, 800 pasos, LR 1e-4, rank 16.

Regla de decisión, fijada antes de ver resultados y revisada por Codex:

1. Primaria, sin cambios: Δ margen CLIP contra los centroides de train con IC 95 % sobre 0.
2. Si la primaria no se cumple, se elige el checkpoint con mayor Δ margen contra el texto del preset (desempate: el más temprano) y se adopta solo si cumple todo: (a) IC del Δ de texto sobre 0; (b) límite inferior del IC del Δ de centroides mayor que −0,005; (c) DINOv2 medio no baja más de 0,01 y la deriva > 12 no sube más de 3 de 75; (d) revisión humana ciega de Isaac sobre los 15 productos de val, un estilo por producto sorteado antes (semilla 0), apagado frente al checkpoint en orden aleatorio, prefiriendo el LoRA en al menos 12 de 15 (p ≈ 0,018 unilateral si fuera azar).
3. En ambas vías, revisión visual de partes agregadas, bordes y sombras antes de adoptar. El margen de texto está sesgado a favor de este LoRA (se eligió con él), por eso nunca decide solo.
4. Si se adopta, test se usa una sola vez con la configuración fijada.

**Resultado en val del LoRA por autodestilación** (`v4distill/20261116-170052-train`, eval `20261116-173240-eval`, 1,78 unidades): la regla primaria se cumple por primera vez. Δ margen frente a apagado: ckpt200 +0,0038 [−0,0007, +0,0088], ckpt400 +0,0044 [+0,0004, +0,0086], ckpt800 +0,0066 [+0,0021, +0,0113]; el estilo pedido gana en 32/75 sin LoRA y en 36 y 40/75 con ckpt400 y ckpt800; DINOv2 del producto 0,946 → 0,951. La regla propone ckpt400 (equivalente al mejor dentro de 0,005 y más temprano). La mejora se concentra en warm (madera y luces cálidas en todos los productos) y en menos equipo de estudio; en natural, lifestyle y minimal es pequeña. Revisión de Isaac sobre una grilla de 6 productos x 5 estilos: con LoRA la sombra tiene mejor dirección, el fondo queda más difuminado y centra el producto, y warm es claramente mejor. Partes agregadas: aparecen igual con y sin LoRA (vienen de la base con `strength` 1,0), salvo dos casos en warm (ámbar sobre una caja y la base de una lámpara convertida en frasco).

Confirmación en test, fijada antes de abrirlo y revisada por Codex: ckpt400 a peso 1,0 contra apagado, los 25 productos de test, una semilla por producto y estilo como en val (no varias: dos semillas no caben en el plazo). Solo cuenta una corrida completa (25 productos, 250 imágenes, sin parcial). Confirmado si el IC 95 % bootstrap del Δ margen de centroides, promediando primero los estilos y remuestreando productos, queda sobre 0; si cruza 0 con media positiva, "dirección consistente, no confirmada"; si la media es negativa, no confirmado. Además se cuentan los pares con partes agregadas en cada configuración y el LoRA no puede superar a apagado en más de 2 de 125. Test se empaqueta bajo `val/` del tar; `config.split` registra que es test (los campos `val_*` de `style_refs` corresponden a test en esa corrida).
