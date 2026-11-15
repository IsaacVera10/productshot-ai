# 10: Curación de BG60k para el LoRA del modo avanzado

**Estado (2026-11-15):** protocolo acordado y revisado con Codex en dos rondas. La etapa 1 (manifiesto) está hecha sobre los 44 304 pares (`training/bg60k_manifest.py`, resultados abajo); las demás etapas siguen pendientes. Cumple el Hito 4 de `04_ROADMAP_Y_TESTS.md` y alimenta la etapa `train` de `training/README.md`.

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
| 2 | Nicho: excluir categorías fuera del alcance de `docs/00` (productos grandes, ropa en personas, transparentes, muy reflectantes, alimentos preparados) | Manifiesto y lista de categorías revisada a mano | Columna `niche` con motivo | Laptop |
| 3 | Máscara: alertas calibradas a mano (componentes, área, bordes tocados, fondo visible) | Manifiesto | Columna `mask_ok` con motivo | Laptop |
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
| Componentes que superan el 1 % de la imagen | 1: 37 002 (83,5 %) · 2: 5 362 · 3 o más: 1 934 · 0 (máscara vacía): 6 | Varios productos o máscaras partidas en el 16 %: alerta de la etapa 3 |
| Bordes tocados por el bbox | 0: 35 553 (80,2 %) · 1: 4 669 · 2: 2 734 · 3: 903 · 4: 445 | Producto posiblemente recortado por el encuadre en el 20 % |
| Área del producto | p1 5,3 % · p5 10,4 % · mediana 31,3 % · p95 61,4 % · p99 77,4 % · máx. 100 % | Fondo visible escaso en la cola alta; las máscaras del 100 % se revisan |
| Brillo del fondo (0 a 255) | p5 22,5 · mediana 163,9 · p95 237,9 | Hay fondos casi negros: relevantes para el estilo y para G12 |
| Textura del fondo (desviación estándar) | p5 6,8 · mediana 34,3 · p95 70,3 | Fondos lisos de catálogo junto a escenas con textura |
| Imágenes idénticas (mismo sha256) | 1 547 repetidas | Duplicados exactos: se agrupan en la etapa 4 |
| Mismo pHash del recorte del producto | 3 007 repetidas | El mismo producto aparece con varios fondos: la agrupación por identidad es necesaria |

Corrida en la laptop con 1 proceso, CPU limitada al 60 % de un núcleo (cgroup), prioridad mínima y pausa automática a 80 °C. Con más procesos la laptop llegó a 100 °C y llegó a apagarse. Tardó unos 28 minutos de cómputo efectivo.
