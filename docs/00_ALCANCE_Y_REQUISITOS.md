# 00 — Alcance, requisitos y exclusiones

## Obligatorio según enunciado del Proyecto 2

1. **Una imagen** cargada por la persona usuaria como entrada de la generación.
2. **Exactamente cinco variaciones visuales** por ejecución exitosa.
3. **Interfaz gráfica** para visualizar las cinco imágenes y aceptar/rechazar **cada una**.
4. **Registro estructurado** de las cinco decisiones.
5. **Agente basado en LLM** que genera un resumen indicando explícitamente cuáles fueron aceptadas y rechazadas; el resumen debe ser coherente con el registro.
6. **Implementar dos de los tres enfoques** definidos en la sección 2.3, como alternativas de respaldo o elección por el usuario.
7. **Seleccionar, justificar y preparar un dataset**; considerar calidad de datos, ética y restricciones.
8. **Evaluar** calidad visual, diversidad, coherencia del resumen, usabilidad y limitaciones éticas.
9. Código reproducible, evidencias/diagramas, informe técnico de **máximo 10 páginas**, video de **máximo 3 minutos** y presentación de **máximo 6 minutos**. El video requiere docente UTEC o estudiante pregrado de carrera distinta de CC/DS, con presentación y consentimiento al comienzo (~10 s).

### Lineamientos adicionales indicados en clase

Además del enunciado, el docente pidió en clase:

| Lineamiento | Dónde se cubre |
|---|---|
| Encontrar un nicho para el proyecto | Sección «Nicho» de este documento |
| Guardrails para manejar fallos | `08_ITERACIONES_Y_GUARDRAILS.md`, parte B (ADR-008) |
| Métricas de usabilidad | `06_EVALUACION_Y_ETICA.md`, sección «Usabilidad» |
| Probar con un mínimo de 10 usuarios ajenos a la carrera (familiares o conocidos) | `06_EVALUACION_Y_ETICA.md`, sección «Prueba con usuarios» |

### No obligatorio en el enunciado

- Comparación experimental A/B de los dos enfoques (la haremos porque aporta evidencia).
- Dos bases de datos operacionales o dos datasets obligatorios (son elecciones técnicas, no mandatos).
- Generación repetida ilimitada, aprendizaje online, reconstrucción 3D, multiagente, Kubernetes.
- Usar más de un LLM: **se sugiere**, no se exige.

## Objetivo

Generar, a partir de una foto casera de un producto, cinco propuestas de ambientación para catálogo o redes sociales, priorizando la conservación de sus rasgos visibles, para que la persona acepte o rechace cada una y reciba un resumen fiel de sus decisiones.

La aplicación no es una herramienta de identidad de marca: no crea logotipos ni aprende la marca del usuario. Un logotipo presente en el producto se trata como un rasgo visible a conservar.

## Caso escogido: ProductShot AI

Personas emprendedoras suben una foto casera de un artículo pequeño y obtienen cinco **ambientaciones**: `studio`, `natural`, `warm`, `lifestyle`, `minimal`. Son cinco presets **propuestos**, no impuestos por el curso. El sistema conserva orientación/perspectiva y características visibles; puede cambiar principalmente fondo y composición circundante. Para un producto con logotipo, texto o estampado, evitar editar esas regiones.

`minimal` debe distinguirse de `studio` por rasgos observables de fondo, espacio negativo y utilería. Los presets describen principalmente el entorno: preservar los píxeles del producto impide cambiar físicamente su iluminación.

### Nicho

**Quiénes:** personas que emprenden en Perú y venden productos pequeños, hechos a mano o de reventa, por Instagram, TikTok, WhatsApp o Facebook Marketplace. Toman las fotos con el celular, en casa, y no contratan sesiones de fotografía.

**Qué productos:** objetos pequeños, rígidos y de superficie mate o semimate que se fotografían sobre una mesa: tazas y vasos personalizados, velas, jabones artesanales, macetas, cerámica, artículos de decoración y papelería. Coinciden con los casos obligatorios de `04_ROADMAP_Y_TESTS.md`.

**Problema:** una foto casera con fondo desordenado o luz pobre transmite poca confianza en una publicación de venta. Las alternativas al alcance tienen límites: un removedor de fondo deja el producto sobre un fondo plano y sin contexto, y un editor generativo general puede alterar el producto, su color o su logotipo.

**Propuesta de valor:** en una sola sesión, cinco ambientaciones listas para catálogo o redes sociales, sin escribir prompts y conservando el producto tal como es.

**Persona tipo:** vende tazas sublimadas por Instagram desde su casa, publica varias veces por semana, usa el celular para todo y no tiene experiencia con herramientas de diseño. Quiere fotos con aspecto profesional sin aprender a usar un editor.

**Por qué este nicho:**

- Los productos pequeños y rígidos son los que la segmentación y el inpainting manejan mejor; ropa, alimentos o joyería brillante fallarían más.
- Las redes sociales piden varias imágenes distintas de un mismo producto, lo que da sentido a generar cinco ambientaciones.
- Es un público al que el equipo tiene acceso entre familiares y conocidos, lo que hace viable la prueba con usuarios reales.

**Filtro por tipo de producto (G15, 2026-11-16):** se rechaza lo que Instagram y TikTok no permiten anunciar y lo que facilita fraude (armas, tabaco y vapes, drogas, dinero y documentos, productos sexuales); lo que está fuera del nicho recibe un aviso, no un rechazo. Fotos de más de 10 MB (cámaras profesionales) se rechazan: no corresponden al nicho.

**Fuera del nicho en v1:** ropa puesta en modelos, alimentos preparados, productos grandes como muebles, objetos transparentes o muy reflectantes y fotos de personas.

### Input v1

- 1 archivo JPG/PNG RGB, máximo configurable (valor preliminar: **10 MB**).
- Un solo producto principal, completo y no obstruido; se advierte que fotos borrosas, transparentes, reflectantes o múltiples objetos pueden fallar.
- Redimensionamiento de trabajo con proporción conservada y padding si se necesita; preservar archivo original.
- La máscara se define **blanca para área editable (fondo), negra para región protegida (producto)** al comunicarse con el pipeline de inpainting; documentar transformaciones y revisar semántica de cada biblioteca.

## Modos planteados

| Modo de UI | Descripción | Trabajo propio |
|---|---|---|
| `intermediate` | Editor por instrucciones preentrenado (Qwen-Image-2.1-Turbo) que re-ilumina el producto para la escena, con escenas por categoría y candidatas verificadas contra la foto (ADR-010) | Experimentos de modelos, escenas por categoría, verificación calibrada, segmentación, guardrails e integración |
| `advanced` | Inpainting con modelo de difusión adaptado con LoRA especializado en estética comercial | Dataset curado, entrenamiento LoRA, integración y análisis del cambio |

**Niveles elegidos:** intermedio y avanzado. El modo `advanced` se presenta como modalidad utilizable y como experimento con pregunta propia: ¿entrenar un LoRA con estética comercial mejora las imágenes frente al modo intermedio?

**Plan B:** si el LoRA no resulta viable técnicamente (el equipo decidió el 2026-11-14 no esperar una confirmación escrita de la docente), implementar básico + intermedio, que también cumple la regla de dos de tres enfoques. Que el LoRA no mejore los resultados no activa el plan B: es un resultado negativo que se reporta como evidencia.

## Objetivo de adaptación del LoRA

Se busca adaptar el modelo a la estética de escena comercial: fondos, superficies y composición. No aprende la identidad del producto subido, las preferencias del usuario ni iluminación físicamente correcta, y puede aprender asociaciones indeseadas con productos o marcas. Si logra adaptarse y generalizar es una hipótesis a evaluar.

Se entrena un solo LoRA condicionado por captions que nombran el estilo, porque reduce la gestión de entrenamientos separados. Exige cobertura por estilo, captions consistentes y evaluación individual, para comprobar que responde a cada estilo sin mezclarlos ni colapsar hacia el dominante.

## Datos

| Conjunto | Contenido | Condiciones |
|---|---|---|
| Entrenamiento y validación estética | BG60k, candidato principal condicionado a auditoría | Piloto de 100–200 imágenes revisadas por estilo; solo se amplía si la validación lo justifica |
| Complemento | CommonCatalog CC-BY | Solo si se demuestra un déficit de cobertura; conserva licencia y atribución por imagen |
| Evaluación de segmentación | Finegrain Product Masks Lite | No se usa para entrenar |
| Prueba | Fotos caseras más los casos obligatorios de `04_ROADMAP_Y_TESTS.md` | Autorización que cubra las fotografías, no solo la propiedad del objeto |

Si el acceso reproducible a BG60k no se confirma (los permisos de publicación no se consultan: es un trabajo académico), la alternativa es ABO + CommonCatalog, con más trabajo de curación. La selección se decidió por consenso del equipo (ADR-006).

Los conjuntos no comparten productos, y las variantes casi duplicadas de un producto se agrupan en la misma partición. El análisis exploratorio se realiza en cualquier caso, porque el dataset es obligatorio. Si un estilo no tiene datos suficientes, el aprendizaje se limita a los estilos cubiertos: los demás se siguen generando con las capacidades del modelo preentrenado, y esa diferencia se declara.

## Alcance funcional de v1

- Una sesión contiene **una imagen, un modo y una generación** de cinco alternativas; una segunda generación requiere **otra sesión** (evita ambigüedad de decisiones). Cada variación tiene ID, índice 1..5, estilo, URI, seed, parámetros y estado de evaluación.
- El usuario debe decidir `accepted` o `rejected` para las cinco. `reason` es opcional.
- Se permite **0 aceptadas** y **5 rechazadas**; la sesión se puede resumir igualmente.
- No atribuir motivaciones si no hay motivos explícitos. Se puede informar una asociación descriptiva entre estilos y decisiones, sin afirmar causas psicológicas.
- El agente consulta los registros con herramientas de **solo lectura** y produce un resumen estructurado con referencias a variaciones y motivos. Python verifica cobertura, decisiones y citas asociadas al registro correcto, y renderiza la enumeración. La narrativa libre es opcional y su revisión automática es parcial.
- Si la segmentación falla, la persona puede reintentarla o subir una máscara manual en PNG antes de generar. No se incluye editor de máscaras.

## Excluido inicialmente

- Voltear producto, cambiar su forma o proyectar lados no visibles.
- Reentrenamiento por feedback tras cada usuario, edición libre de texto/logotipo, comercio público/multi-tenant, pagos, intervención manual sofisticada para máscaras, app móvil, live streaming y Kubernetes.
- Despliegue GCP en el primer PR.

## Requisito pedagógico y ética

Cumplir los requisitos de autoría, reproducción y evaluación establecidos por el enunciado y el sílabo.
