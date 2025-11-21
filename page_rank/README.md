# Guía de Configuración con Docker + Análisis de PageRank + ERD del code

Una aplicación en Docker para el análisis de grafos PageRank en conjuntos de datos de la web. Esta herramienta calcula PageRank, HITS (Hubs & Authorities) y genera visualizaciones interactivas.

> Se utilizaron herramientas de IA como ChatGPT, Claude y Copilot durante la creación de esta solución.

-----

## Requisitos Previos

  - Docker instalado ([Obtener Docker](https://docs.docker.com/get-docker/))
  - Git instalado
  - Un archivo de conjunto de datos llamado `web-Stanford.txt` (formato de lista de aristas / *edge list*)

-----

## Inicio Rápido

### 1\. Clonar el Repositorio

```bash
git clone https://github.com/TU_USUARIO/meli-pagerank.git
cd meli-pagerank/page_rank
```

### 2\. Construir la Imagen de Docker

```bash
docker build -t pagerank .
```

Esto crea una imagen de Docker llamada `pagerank` con todas las dependencias requeridas.

### 3\. Ejecutar el Análisis

**Importante**: Reemplaza `/absolute/path/to/data` con la **ruta absoluta** al directorio que contiene tu archivo `web-Stanford.txt`.

```bash
docker run --rm \
  -v /absolute/path/to/data:/app/data:ro \
  -v "$(pwd)/out:/app/out" \
  pagerank \
  --input /app/data/web-Stanford.txt \
  --max-edges 50000 \
  --topk 20 \
  --out-dir /app/out
```

**Ejemplo** (macOS/Linux):

```bash
docker run --rm \
  -v /Users/john/datasets:/app/data:ro \
  -v "$(pwd)/out:/app/out" \
  pagerank \
  --input /app/data/web-Stanford.txt \
  --max-edges 50000 \
  --topk 20 \
  --out-dir /app/out
```

**Ejemplo** (Windows PowerShell):

```powershell
docker run --rm `
  -v C:\Users\john\datasets:/app/data:ro `
  -v ${PWD}\out:/app/out `
  pagerank `
  --input /app/data/web-Stanford.txt `
  --max-edges 50000 `
  --topk 20 `
  --out-dir /app/out
```

-----

## Explicación de Comandos

### Montaje de Volúmenes

  - **`-v /absolute/path/to/data:/app/data:ro`**

      - Monta tu directorio de datos local en `/app/data` dentro del contenedor.
      - `:ro` = solo lectura (protege tus datos originales).
      - **Debe contener** un archivo llamado `web-Stanford.txt`.
      - Usa la ruta absoluta (ej. `/Users/john/datasets` o `C:\Users\john\datasets`).

  - **`-v "$(pwd)/out:/app/out"`**

      - Monta el directorio local `out/` para los resultados.
      - Los resultados aparecerán en tu carpeta local `page_rank/out/`.
      - Lectura y escritura habilitadas (el contenedor puede escribir archivos de salida).

### Banderas (Flags)

  - **`--rm`** - Elimina automáticamente el contenedor después de que termina (limpieza).
  - **`pagerank`** - El nombre de la imagen de Docker a ejecutar.

-----

## Argumentos Disponibles

| Argumento | Tipo | Predeterminado | Descripción |
|----------|------|---------|-------------|
| `--input` | string | **requerido** | Ruta al archivo de lista de aristas (debe ser `/app/data/web-Stanford.txt`) |
| `--max-edges` | int | 50000 | Número máximo de aristas a cargar del conjunto de datos |
| `--topk` | int | 20 | Número de nodos mejor clasificados para extraer en el CSV final |
| `--sub_k` | int | 50 | Número de nodos a incluir en la visualización del subgrafo explicativo |
| `--sampling` | string | `simple` | Estrategia de muestreo: `simple` (secuencial) o `snowball` (basado en BFS) |
| `--out-dir` | string | `out` | Ruta del directorio de salida (usar `/app/out` para Docker) |

-----

## Ejemplos de Uso

### 1\. Prueba Pequeña (1,000 aristas)

Bueno para pruebas o análisis rápidos:

```bash
docker run --rm \
  -v /absolute/path/to/data:/app/data:ro \
  -v "$(pwd)/out:/app/out" \
  pagerank \
  --input /app/data/web-Stanford.txt \
  --max-edges 1000 \
  --topk 10 \
  --out-dir /app/out
```

-----

## Archivos de Output

Después de ejecutar, revisa el directorio `out/` para encontrar:

### 1\. `top_20_pagerank.csv` (o `top_N_pagerank.csv`)

Archivo CSV con los nodos mejor clasificados y sus métricas:

| Columna | Descripción |
|--------|-------------|
| `node` | Identificador del nodo |
| `pagerank` | Puntuación PageRank (medida de importancia) |
| `authority` | Puntuación de autoridad HITS (calidad de los enlaces entrantes) |
| `hub` | Puntuación de hub HITS (calidad de los enlaces salientes) |
| `in_degree` | Número de aristas entrantes |
| `out_degree` | Número de aristas salientes |

### 2\. `explanatory_subgraph.html`

Visualización interactiva construida con PyVis:

  - Zoom, paneo, arrastrar nodos
  - Pasa el cursor para ver detalles del nodo
  - Tamaño del nodo = Puntuación PageRank
  - Abrir en cualquier navegador web

# Conclusiones -> Para 5000 Edges

### 1. Nodo 2 -> Estructura de navegacion (Router)
El **Nodo 2** tiene un PageRank (0.025) que es **10 veces superior** al del siguiente competidor.
Sin embargo:
* Tiene un equilibrio perfecto de entradas y salidas (31 in / 31 out).

* **El Insight:** El Nodo 2 es tipo *Home Page* de navegación. Todo el tráfico pasa por él, pero no retiene valor ni conocimiento. Es  infraestructura.
* **El Riesgo:** Estructuralmente, es el punto fallo critico. Si el nodo falla, la red pierde su punto de ruteo, aunque el nodo en sí no aporte contenido de valor.

### 2. Nodos Autoridad

> 226411, 105607, 234704, 38342, 167295, 76448, 41825, 180949, 13, 124470, 81435, 198090, 214128, 34573, 245659, 38, 225872, 35, 140928

Grupo de nodos (encabezados por el 226411 y el 105607) tienen un PageRank bajo (apenas 0.002), pero **Autoridades** en comparación con el Nodo 2.

* **El Insight:** Observa el **Nodo 226411**. Aunque su PageRank es bajo, tiene el **In-Degree más alto de la tabla (82)** y un puntaje de Hub ($3,11 \times 10^{10}$). Estos nodos son **Conectores Conocientes**. No solo tienen la información (Autoridad), sino que también saben a quién más enlazar (Hub).
* **La Oportunidad:** Estos nodos son los motores de la red. Tienen más enlaces entrantes que el Nodo 2, por lo que son la referencia técnica real.

### 3. Los Nodos "Clonados" (76448, 41825, 180949, 13)
Estos cuatro nodos comparten **exactamente el mismo PageRank (0.002033...)** y la misma estructura de enlaces (3 in / 3 out).

* **El Insight:** Estos nodos tienen autoridad cero o cercana. Son páginas "zombies" o contenido duplicado que el algoritmo trata por igual simplemente porque tienen la misma posicion topologica en la red. Se consideran **content leaf nodes** (blogs, paginas de producto, articulos)

---

## Recomendaciones Estratégicas

### Consideraciones y caso de uso: E-commerce
1.  **Nodo 2 (La Fachada):** Tiene el PageRank más alto (0.025), pero su Autoridad y Hub son casi **CERO**. Es pura cáscara. En un e-commerce esto suele ser la **Home Page** mal optimizada: recibe todo el tráfico, pero no transfiere valor a los productos porque su estructura de enlaces internos es deficiente.
2.  **El grupo de alta autoridad (226411, 234704, 105607):** Estos nodos tienen la Autoridad alta en comparacion con el resto. Tienen muchos enlaces entrantes (In-degree 60-80). Estos son tus **Productos Estrella (Best Sellers)**.
3.  **Los "Distribuidores Diluidos" (124470, 225872):** Tienen un Out-Degree altísimo (68 y 189 enlaces salientes) pero una autoridad baja. En e-commerce estas se ven como páginas de **Categorías Gigantes** o listas de "Ver todo". Enlazan a demasiadas cosas, diluyendo su poder.

---

### 3 Recomendaciones Estratégicas para E-commerce

#### 1. Estrategia "Hub de Autoridad": Convertir la Home (Nodo 2) en puntero hacia los Productos Estrella
* **El Problema:** El Nodo 2 (Home) tiene todo el tráfico (PageRank) pero Autoridad cero y Hub cero. Los usuarios llegan y el valor se estanca.
* **La Acción:** Dejar de tratar la Home como un menú de navegación genérico. Enlazar manualmente y con contexto (texto ancla) desde la Home directamente a los nodos del "Grupo de Oro" (226411, 105607).
* **Implementación E-commerce:**
    > Crear una sección en la Home llamada **"Los Favoritos de los Expertos"** o **"Tendencias de la Semana"**. En lugar de enlazar a la categoría "Zapatos", enlazar directamente a la ficha de producto del zapato específico (Nodo 226411). Esto transfiere el PageRank del Nodo 2 directamente al nodo de la Autoridad, potenciando el posicionamiento en el Search Engine -> explotar la iteratividad de la evaluacion pagerank. 

#### 2. Estrategia "Cross-Selling Horizontal": Romper los 
* **El Problema:** Los nodos con alta autoridad (como el 234704 y el 105607) tienen muchos enlaces entrantes (In-degree >50) pero actúan como islas. Acumulan valor de entrada pero no lo comparten entre sí para la calidad del puntaje del nodo.
* **La Acción:** Crear un anillo de enlaces entre estos productos de alta autoridad para compartir el **famoso** "link juice".
* **Implementación E-commerce:**
    * En la página de producto del **Nodo 226411** (producto más fuerte), añadir un bloque de *"Comprados frecuentemente juntos"* o *"Alternativas Premium"*.
    * Asegurarse de que este bloque enlace **exclusivamente** a los otros nodos de alta autoridad (105607, 234704).
    * **Resultado:** Creacion un clúster de relevancia temática. Si un producto sube en Google, arrastra a los demás hacia arriba.

#### 3. Estrategia Consolidación: Arreglar el Nodo 225872
* **El Problema:** El Nodo 225872 tiene 189 enlaces salientes (Out-degree) pero autoridad casi 0 -> ($10^{-6}$). Es una página de categoría o etiqueta que está "sangrando" valor. Al enlazar a 189 sitios, cada enlace recibe una fracción minúscula de valor.
* **La Acción:** Reducir la cantidad de enlaces salientes o paginar el contenido para concentrar la autoridad.
* **Implementación E-commerce:**
    * Si el Nodo 225872 es una página de categoría "Ver Todo" con 189 productos -> cambiar.
    * Divídela en sub-categorías más específicas (ej: "Zapatos Rojos", "Zapatos de Cuero").
    * El objetivo es que esta página enlace solo a los 20-30 productos más relevantes, aumentando el valor que cada producto recibe.

---

# Estructuracion del codigo

## 1. ENTITY RELATIONSHIP DIAGRAM (ERD)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA MODELS (Dataclasses)                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────┐
    │  InternalGraphRepresentation         │
    ├──────────────────────────────────────┤
    │ - edges: List[Tuple[str, str]]       │
    ├──────────────────────────────────────┤
    │ + __len__() -> int                   │
    │ + is_empty() -> bool                 │
    └──────────────────────────────────────┘
              │
              │ used by
              ▼
    ┌──────────────────────────────────────┐
    │  NetworkXGraphAdapter                │
    ├──────────────────────────────────────┤
    │ - _graph: nx.DiGraph                 │
    │ - _metrics: GraphMetrics             │
    │ - _analysis_df: pd.DataFrame         │
    ├──────────────────────────────────────┤
    │ + compute_metrics()                  │
    │ + get_dataframe_analysis()           │
    │ + extract_subgraph_topk_1hop()       │
    │ + get_networkx_graph()               │
    └──────────────────────────────────────┘
              △
              │ implements
              │
    ┌──────────────────────────────────────┐
    │    GraphOperations (ABC)             │
    ├──────────────────────────────────────┤
    │ + compute_metrics() [abstract]       │
    │ + extract_subgraph_topk_1hop()       │
    │ + get_dataframe_analysis()           │
    │ + num_nodes()                        │
    │ + num_edges()                        │
    └──────────────────────────────────────┘


    ┌──────────────────────────────────────┐
    │     GraphMetrics                     │
    ├──────────────────────────────────────┤
    │ - pagerank: Dict[str, float]         │
    │ - hubs: Dict[str, float]             │
    │ - authorities: Dict[str, float]      │
    │ - in_degree: Dict[str, int]          │
    │ - out_degree: Dict[str, int]         │
    └──────────────────────────────────────┘
              △
              │ stored in
              │
    ┌──────────────────────────────────────┐
    │    NodeAnalysis                      │
    ├──────────────────────────────────────┤
    │ - node: str                          │
    │ - pagerank: float                    │
    │ - authority: float                   │
    │ - hub: float                         │
    │ - in_degree: int                     │
    │ - out_degree: int                    │
    └──────────────────────────────────────┘
```

---

## 2. CLASS HIERARCHY DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STRATEGY PATTERN: Readers                           │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────┐
    │     EdgeReader (ABC)           │
    ├────────────────────────────────┤
    │ + read(file, max_edges)        │
    │ # _parse_line(line)            │
    └────────────────────────────────┘
            △           △
            │           │
      ┌─────┘           └──────┐
      │                        │
      ▼                        ▼
┌──────────────────┐  ┌──────────────────┐
│SimpleEdgeReader  │  │SnowballEdgeReader│
├──────────────────┤  ├──────────────────┤
│Read first N      │  │BFS-based         │
│edges sequentially│  │sampling from     │
│                  │  │first node        │
└──────────────────┘  └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                      STRATEGY PATTERN: Visualizers                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────┐
    │   GraphVisualizer (ABC)        │
    ├────────────────────────────────┤
    │ + visualize(adapter, path)     │
    └────────────────────────────────┘
            △           △
            │           │
      ┌─────┘           └──────┐
      │                        │
      ▼                        ▼
┌──────────────────┐  ┌──────────────────┐
│PyvisVisualizer   │  │MatplotlibVisualiz│
├──────────────────┤  ├──────────────────┤
│Interactive HTML  │  │Static PNG        │
│(physics-enabled) │  │(spring layout)   │
└──────────────────┘  └──────────────────┘
```

---

## 3. ADAPTER PATTERN DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTER PATTERN: Graph                               │
└─────────────────────────────────────────────────────────────────────────────┘

External Request (Library Agnostic)
         │
         ▼
┌─────────────────────────────────────────────┐
│  GraphOperations (Interface/Protocol)       │
│  - compute_metrics()                        │
│  - extract_subgraph_topk_1hop()             │
│  - get_dataframe_analysis()                 │
│  - num_nodes()                              │
│  - num_edges()                              │
└─────────────────────────────────────────────┘
         △
         │ implements
         │
┌─────────────────────────────────────────────┐
│  NetworkXGraphAdapter                       │
├─────────────────────────────────────────────┤
│  Internal: nx.DiGraph                       │
│  - All NetworkX code encapsulated here      │
│  - Translates to/from GraphOperations       │
└─────────────────────────────────────────────┘
         │
         ▼
    [nx library]
    
ADVANTAGE: If you want to swap networkx → igraph or graph-tool,
only change NetworkXGraphAdapter implementation!
```

---

## 4. COMPLETE PROGRAM FLOW DIAGRAM

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION FLOW: WebGraphAnalyzer                        │
└────────────────────────────────────────────────────────────────────────────────┘

START (CLI Arguments)
  │
  ├─ --input (file path)
  ├─ --max-edges (limit)
  ├─ --sampling (simple|snowball)
  ├─ --topk (20)
  ├─ --sub_k (50)
  └─ --out-dir (output directory)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: INITIALIZE                                          │
├──────────────────────────────────────────────────────────────┤
│  Choose EdgeReader based on --sampling                       │
│                                                              │
│  if sampling == "snowball":                                  │
│    reader = SnowballEdgeReader()                             │
│  else:                                                       │
│    reader = SimpleEdgeReader()                               │
│                                                              │
│  analyzer = WebGraphAnalyzer(edge_reader=reader)             │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: READ EDGES                                          │
├──────────────────────────────────────────────────────────────┤
│  analyzer.analyze(                                           │
│    file_path="web-Stanford.txt",                             │
│    max_edges=1000,                                           │
│    topk=20,                                                  │
│    sub_k=50                                                  │
│  )                                                           │
│                                                              │
│  reader.read(file_path, max_edges)                           │
│    ├─ Parse file line-by-line                               │
│    ├─ Skip comments (#) and empty lines                      │
│    ├─ Extract source, destination tuples                     │
│    └─ Return InternalGraphRepresentation(edges=[...])        │
└──────────────────────────────────────────────────────────────┘
      │
      │ InternalGraphRepresentation
      │ (edges: [(src, dst), ...])
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: BUILD GRAPH ADAPTER                                 │
├──────────────────────────────────────────────────────────────┤
│  graph_repr = InternalGraphRepresentation(edges=[...])        │
│  adapter = NetworkXGraphAdapter(graph_repr)                   │
│                                                              │
│  Inside NetworkXGraphAdapter.__init__:                       │
│    ├─ self._graph = nx.DiGraph()                             │
│    ├─ self._graph.add_edges_from(graph_repr.edges)           │
│    ├─ self._metrics = None (lazy compute)                    │
│    └─ self._analysis_df = None (lazy compute)                │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: COMPUTE METRICS                                     │
├──────────────────────────────────────────────────────────────┤
│  ranking_df = adapter.get_dataframe_analysis()               │
│                                                              │
│  Inside compute_metrics():                                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. PageRank (nx.pagerank)                            │   │
│  │    - Iterative voting algorithm                      │   │
│  │    - tol=1e-6 (convergence threshold)                │   │
│  │    - max_iter=200                                    │   │
│  │    - Output: Dict[node -> score]                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. HITS (nx.hits)                                    │   │
│  │    - Hubs & Authorities algorithm                    │   │
│  │    - max_iter=200, tol=1e-8                          │   │
│  │    - Output: (hubs dict, authorities dict)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Degrees (direct count)                            │   │
│  │    - in_degree = count of incoming edges             │   │
│  │    - out_degree = count of outgoing edges            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Store in GraphMetrics dataclass                             │
└──────────────────────────────────────────────────────────────┘
      │
      ▼ GraphMetrics
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: BUILD ANALYSIS DATAFRAME                            │
├──────────────────────────────────────────────────────────────┤
│  For each node in graph:                                     │
│    Create NodeAnalysis(                                      │
│      node=node_id,                                           │
│      pagerank=metrics.pagerank[node],                        │
│      authority=metrics.authorities[node],                    │
│      hub=metrics.hubs[node],                                 │
│      in_degree=metrics.in_degree[node],                      │
│      out_degree=metrics.out_degree[node]                     │
│    )                                                         │
│                                                              │
│  Convert to pd.DataFrame                                     │
│  Sort by pagerank (descending)                               │
│  Reset index                                                 │
│                                                              │
│  Output: ranking_df (all nodes with metrics)                 │
└──────────────────────────────────────────────────────────────┘
      │
      ▼ ranking_df
┌──────────────────────────────────────────────────────────────┐
│  STEP 6: SAVE TOP-K CSV                                      │
├──────────────────────────────────────────────────────────────┤
│  topk_df = ranking_df.head(topk)  # topk=20                  │
│  topk_df.to_csv("out/top_20_pagerank.csv", index=False)      │
│                                                              │
│  Output CSV columns:                                         │
│    node, pagerank, authority, hub, in_degree, out_degree    │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 7: EXTRACT SUBGRAPH (Top-K + 1-hop neighbors)          │
├──────────────────────────────────────────────────────────────┤
│  sub_adapter, _ = adapter.extract_subgraph_topk_1hop(top_n=50)
│                                                              │
│  Inside extract_subgraph_topk_1hop:                          │
│    1. top_nodes = ranking_df['node'].iloc[:50]              │
│    2. sub_nodes = set(top_nodes)                             │
│    3. For each node in top_nodes:                            │
│       ├─ Add all predecessors (incoming)                     │
│       └─ Add all successors (outgoing)                       │
│    4. sub_graph = original_graph.subgraph(sub_nodes)         │
│    5. Create new NetworkXGraphAdapter(sub_graph)             │
│    6. Return (sub_adapter, original_ranking_df)              │
│                                                              │
│  Result: Smaller graph focused on authority nodes           │
│  Size: typically 300-500 nodes (for 50 top + neighbors)      │
└──────────────────────────────────────────────────────────────┘
      │
      ▼ sub_adapter (NetworkXGraphAdapter)
┌──────────────────────────────────────────────────────────────┐
│  STEP 8: VISUALIZE SUBGRAPH (2 strategies)                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  For each visualizer in [PyvisVisualizer, MatplotlibViz]:   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ PyvisVisualizer: Interactive HTML                    │   │
│  │ ─────────────────────────────────────────────────    │   │
│  │ 1. Get ranking_df from sub_adapter                   │   │
│  │ 2. Get nx.DiGraph from sub_adapter                   │   │
│  │ 3. Create Network (height=900px, width=100%)         │   │
│  │ 4. For each node:                                    │   │
│  │    ├─ size = 40 * pagerank (bigger = higher PR)      │   │
│  │    ├─ title = hover tooltip (node, PR, in_degree)    │   │
│  │    └─ net.add_node()                                 │   │
│  │ 5. For each edge: net.add_edge()                     │   │
│  │ 6. Enable physics: true (auto-layout)                │   │
│  │ 7. net.save_graph("explanatory_subgraph.html")       │   │
│  │                                                      │   │
│  │ Output: Interactive visualization                    │   │
│  │         - Drag nodes                                 │   │
│  │         - Zoom/pan                                   │   │
│  │         - Hover for info                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MatplotlibVisualizer: Static PNG                     │   │
│  │ ──────────────────────────────────────────────────   │   │
│  │ (only if nodes <= 500)                               │   │
│  │                                                      │   │
│  │ 1. Get ranking_df and nx.DiGraph                     │   │
│  │ 2. pos = spring_layout(k=0.15, iterations=100)       │   │
│  │ 3. For each node:                                    │   │
│  │    └─ size = 200 + 3000 * pagerank                   │   │
│  │ 4. Draw nodes (sized by PageRank)                    │   │
│  │ 5. Draw edges (alpha=0.3, transparent)               │   │
│  │ 6. Draw labels (node IDs, font_size=8)               │   │
│  │ 7. plt.savefig("explanatory_subgraph.png", dpi=200)  │   │
│  │                                                      │   │
│  │ Output: Static image for reports                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 9: DONE                                                │
├──────────────────────────────────────────────────────────────┤
│  Print summary:                                              │
│    ✓ Loaded 1000 edges                                       │
│    ✓ Saved Top-20 to CSV                                     │
│    ✓ Subgraph: X nodes, Y edges                              │
│    ✓ Saved HTML visualization                                │
│    ✓ Saved PNG visualization                                 │
│    ✓ Done. Outputs in "out/" directory                       │
│                                                              │
│  Returns: ranking_df (all nodes with metrics)                │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
    END

OUTPUT FILES:
  out/top_20_pagerank.csv          (Top-K rankings)
  out/explanatory_subgraph.html    (Interactive visualization)
  out/explanatory_subgraph.png     (Static visualization)
```

---

## 5. DEPENDENCY INJECTION FLOW

```
┌────────────────────────────────────────────────────────────────────┐
│              DEPENDENCY INJECTION: Constructor Pattern              │
└────────────────────────────────────────────────────────────────────┘

WebGraphAnalyzer is created with injected dependencies:

    reader = SimpleEdgeReader()  ◄── Dependency 1
    visualizers = [              ◄── Dependency 2
        PyvisVisualizer(),
        MatplotlibVisualizer()
    ]
    
    analyzer = WebGraphAnalyzer(
        edge_reader=reader,
        graph_adapter_class=NetworkXGraphAdapter,  ◄── Dependency 3
        visualizers=visualizers
    )


BENEFIT: Easy to swap implementations without changing WebGraphAnalyzer:

    # Original
    reader = SimpleEdgeReader()
    
    # Swap to new reader (same interface)
    reader = SnowballEdgeReader()
    # WebGraphAnalyzer doesn't change!
    
    # Add new visualizer (implements GraphVisualizer)
    visualizers = [PyvisVisualizer(), NewD3Visualizer()]
    # WebGraphAnalyzer handles it automatically!
```

---

## 6. DATA FLOW THROUGH SYSTEM

```
┌────────────────────────────────────────────────────────────────────┐
│                     DATA TRANSFORMATION PIPELINE                   │
└────────────────────────────────────────────────────────────────────┘

Raw File (web-Stanford.txt)
  │
  │ [1000 lines, each: "src dst"]
  │
  ▼
EdgeReader.read()
  │
  │ Parse, filter comments, extract tuples
  │
  ▼
InternalGraphRepresentation
  │
  │ edges: [(1, 6548), (1, 15409), ..., (280935, X)]
  │
  ▼
NetworkXGraphAdapter.__init__()
  │
  │ Convert to nx.DiGraph data structure
  │
  ▼
nx.DiGraph (internal state)
  │
  │ Nodes: 1, 6548, 15409, ..., 280935 (auto-created)
  │ Edges: 1000 directed edges
  │
  ▼
compute_metrics()
  │
  ├─► PageRank: Dict[node → float]
  ├─► HITS: Dict[node → float] (hubs & authorities)
  └─► Degrees: Dict[node → int] (in & out)
      │
      ▼
    GraphMetrics (aggregated)
      │
      ▼
get_dataframe_analysis()
  │
  │ Convert metrics to pandas DataFrame
  │ Sort by PageRank descending
  │
  ▼
pd.DataFrame: ranking_df
  ┌─────────────────────────────────────────┐
  │ node    pagerank  authority  hub  in out│
  ├─────────────────────────────────────────┤
  │ 2       0.0829    0.00554    0.39 31  31│
  │ 76448   0.0067    0.00110    0.00 3   3 │
  │ 13      0.0067    0.00088    0.00 3   3 │
  │ ...                                     │
  └─────────────────────────────────────────┘
      │
      ├──► Save to CSV (top_20_pagerank.csv)
      │
      └──► extract_subgraph_topk_1hop()
           │
           │ Filter to Top-50 + neighbors
           │
           ▼
          InternalGraphRepresentation (subgraph)
           │
           ▼
          NetworkXGraphAdapter (subgraph)
           │
           ├──► PyvisVisualizer
           │    │
           │    │ Extract metrics, build Network object
           │    │
           │    ▼
           │    explanatory_subgraph.html
           │
           └──► MatplotlibVisualizer
                │
                │ Spring layout, draw with matplotlib
                │
                ▼
                explanatory_subgraph.png
```