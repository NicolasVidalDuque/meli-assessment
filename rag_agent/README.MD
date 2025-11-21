# Sistema RAG + Agente Crítico para QA de Laptops


## ¿Qué hace?

Dada una pregunta como *"¿Qué laptops tienen GPU NVIDIA RTX 3050?"*, el sistema:

1.  **Recupera** especificaciones relevantes de laptops desde un conjunto de datos CSV.
2.  **Genera** una respuesta utilizando un LLM con citas adecuadas (ej. `[id_laptop:campo]`).
3.  **Valida** la respuesta contra los datos recuperados para prevenir alucinaciones.
4.  **Regenera** la respuesta si las afirmaciones no tienen sustento, o la acepta si es validada.

-----

Claro, aquí tienes la sección completa y actualizada. He integrado la creación del entorno Conda como el **Paso 1**, reemplazando la instalación manual de dependencias (ya que el `environment.yml` debería encargarse de ello), y he mantenido el flujo lógico para los servicios y la ejecución.

-----

## Instalación y Requisitos

### 1\. Configurar Entorno (Conda)

Este proyecto utiliza un entorno aislado para gestionar las dependencias exactas.

```bash
# 1. Crear el ambiente virtual desde el archivo YAML
conda env create -f environment.yml

# 2. Activar el ambiente
conda activate meli
```

### 2\. Configurar Variables de Entorno

El agente requiere acceso a la API de OpenAI. Crea un archivo `.env` en la raíz del directorio:

```bash
# Crea el archivo .env
touch .env
```

**Contenido de `.env`:**

```ini
OPENAI_API_KEY=sk-tu_clave_de_api_aqui_...
# Opcional: Configuración de LangSmith si se requiere tracking externo
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
```

### 3\. Iniciar Servidor de Métricas (MLflow)

El script reporta trazas de ejecución y métricas de calidad a un servidor local de MLflow.

> **⚠️ Importante:** Este paso es obligatorio antes de ejecutar el script para evitar errores de conexión.

Abre una **nueva terminal**, activa el ambiente y lanza la UI:

```bash
conda activate meli
mlflow ui --port 5000
```

*Mantén esta terminal abierta en segundo plano.*

-----

## Uso

### Ejecutar el Agente (v1)

Con el entorno activado y MLflow corriendo, ejecuta el script principal:

```bash
python v1.py \
  --csv-path ../data/Laptops_with_technical_specifications.csv \
  --sample-size 300 \
  --model-name gpt-3.5-turbo-0125 \
  --top-k 5 \
  --max-iterations 3
```

### Argumentos Clave

| Flag | Por Defecto | Descripción |
|------|-------------|-------------|
| `--csv-path` | `../data/...` | Ruta relativa al dataset CSV |
| `--sample-size` | 300 | Cantidad de filas a procesar (útil para pruebas rápidas) |
| `--retrieval-backend` | `bm25` | Motor de búsqueda (`bm25` o `langchain_vector`) |
| `--top-k` | 5 | Cantidad de fragmentos de contexto a recuperar |
| `--model-name` | `gpt-3.5-turbo-0125` | Modelo LLM para generación y validación |
| `--temperature` | 0.0 | Creatividad del modelo (0 = determinista) |
| `--max-iterations` | 3 | Intentos máximos de auto-corrección (Self-Correction) |
| `--mlflow-uri` | `http://localhost:5000` | Dirección del servidor de tracking |
| `--output-path` | `rag_critical_agent_log.json` | Archivo de salida con resultados |

-----

## Salida (Output)

El script generará dos tipos de resultados:

1.  **Archivo JSON Local** (`rag_critical_agent_log.json`):

    ```json
    {
      "test_results": [
        {
          "query": "¿Qué laptop tiene mejor batería?",
          "critical_decision": { "is_valid": true, "action": "finalize" },
          "final_answer": "La Lenovo ThinkPad X1..."
        }
      ],
      "aggregate_metrics": { "avg_latency": 1.2s, "acceptance_rate": 0.85 }
    }
    ```

2.  **Dashboard de MLflow** (`http://localhost:5000`):
    Visualización de trazas, tiempos de inferencia por componente y métricas agregadas de la ejecución.

-----

## Componentes Principales

### 1\. **LaptopDataIngestion** (Ingesta de Datos)

  * **Propósito**: Transforma el CSV crudo en fragmentos (*chunks*) buscables.
  * **Salida**: Cada campo (CPU, GPU, RAM, etc.) se convierte en un fragmento separado con metadatos.

### 2\. **BM25Retrieval** (Patrón Estrategia)

  * **Propósito**: Recuperación rápida basada en palabras clave.
  * **Extensible**: Intercambiable con búsqueda vectorial implementando la interfaz `RetrievalStrategy`.

### 3\. **AnswerGenerator** (Generador de Respuestas)

  * **Propósito**: Produce respuestas concisas (≤120 palabras) con citas rastreables.
  * **Restricciones**: Debe citar todas las afirmaciones usando el formato `[id_laptop:nombre_campo]`.

### 4\. **CriticalAgent** (Cadena de Responsabilidad vía LangGraph)

  * **Propósito**: Capa de validación con autocorrección.
  * **Iteraciones Máx**: Configurable (por defecto: 3).
  * **Salida**: Aceptar, Rechazar o Respuesta Regenerada.

### 5\. **RAGPipeline** (Patrón Fachada)

  * **Propósito**: Punto de entrada único que oculta la complejidad del subsistema.
  * **Dependencias Inyectadas**: `retrieval`, `generator`, `critical_agent`.

-----

## Patrones de Diseño Utilizados

### Patrón Estrategia (Strategy)

```text
   Estrategia Abstracta
         ▲
         │
   ┌─────┴─────┬──────────────┐
   │           │              │
BM25      VectorDB      CustomRetrieval
```

  * **Dónde**: `RetrievalStrategy`, `ChunkingStrategy`
  * **Beneficio**: Intercambiar algoritmos sin modificar el código del pipeline.

### Inyección de Dependencias (Dependency Injection)

```python
   RAGPipeline(
       retrieval=BM25Retrieval(),        ← Inyectado
       generator=AnswerGenerator(),      ← Inyectado
       critical_agent=CriticalAgent()    ← Inyectado
   )
```

  * **Dónde**: `RAGPipeline.__init__()`
  * **Beneficio**: Testeable, flexible, desacoplado.

### Patrón Fachada (Facade)

```text
   Subsistema Complejo:
   ┌───────────┬─────────────┬────────────┐
   │  Ingesta  │ Recuperación│ Generación │
   └───────────┴─────────────┴────────────┘
                     │
                     ▼
              Interfaz Simplificada:
               pipeline.query()
```

  * **Dónde**: `RAGPipeline.query()`
  * **Beneficio**: API simple para un flujo de trabajo complejo.

### Cadena de Responsabilidad (Chain of Responsibility)

```text
   validar → decidir → regenerar → validar → ...
       ▲                                   │
       │                                   │
       └───────────────────────────────────┘
         (bucle hasta ser válido o máx. iteraciones)
```

  * **Dónde**: `CriticalAgent._build_graph()`
  * **Beneficio**: Procesamiento secuencial con ramificación condicional.


-----

## Dependencias

```bash
# Recuperación y ML
rank-bm25
pandas
numpy

# LLM y Agentes
langchain-ollama
langchain-core
langgraph

# Seguimiento (Tracking)
mlflow
```
