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
- **Afirmación permitida en el informe** (tras ejecutar la auditoría): «particiones por grupos de similitud visual, sin coincidencias de identidad detectadas en la auditoría descrita; persiste riesgo de variantes no detectadas».
- **Equilibrio:** las categorías se reparten entre estilos para que el LoRA no asocie un estilo con un tipo de producto.
- **Pruebas:** el test de BG60k es estético y reservado. Las fotos caseras se dividen en desarrollo (para elegir checkpoint, `strength`, escala y duración) y prueba final, que no se usa para ajustar nada.

## Entrenamiento (cambios pendientes en `training/`)

- Quitar `--random_flip` de `colab_job.py`: voltea textos y logos.
- 1500 pasos sobre unas 600 imágenes de train son unas 2,5 épocas con lote 1. La duración se elige validando checkpoints por estilo, no a priori. Rank 16 y LR 1e-4 son el punto de partida del smoke test, no un óptimo.
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
