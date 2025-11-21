# **Framework Híbrido de Detección de Anomalías**

Un framework de nivel producción diseñado para detectar anomalías de precios en series temporales. Orquesta una comparación entre métodos estadísticos tradicionales (Z-Score) y un enfoque basado en LLM con alta concurrencia.

## 🏗 **Principios de Diseño y Modularidad**

El sistema sigue el **Patrón Strategy** y **Dependency Injection** para asegurar que el código sea extensible y testeable.

* **Clases Base Abstractas (`ABC`):**

  * **Qué:** Todos los modelos heredan de `BaseAnomalyModel`.
  * **Por qué:** Define un contrato estricto (`fit`, `score`). Puedes intercambiar un modelo Z-Score por un Isolation Forest o un modelo LLM sin modificar una sola línea del motor de ejecución.
* **Inyección de Dependencias:**

  * **Qué:** `AnomalySystem` no crea los modelos internamente; los recibe completamente configurados vía el diccionario `base_structure`.
  * **Por qué:** Desacopla la lógica de orquestación de la configuración de los modelos, facilitando enormemente las pruebas unitarias y el ajuste de hiperparámetros.
* **Seguridad de Tipos:**

  * **Qué:** Uso extensivo de `TypedDict` (`ItemData`, `ModelData`) y dataclasses.
  * **Por qué:** Garantiza consistencia en los datos a lo largo del pipeline, evitando errores en tiempo de ejecución por diccionarios mal formados.

## ⚡ **Arquitectura Async para LLM**

La innovación central de este proyecto es `NewLLMAnomalyDetector`. El procesamiento secuencial tradicional de series temporales con LLM es demasiado lento; este módulo lo resuelve mediante un **enfoque Map-Reduce con AsyncIO**.

### El Problema

Analizar miles de puntos de precios de forma secuencial con un LLM tomaría horas debido a la latencia de la API.

### La Solución: Batching Concurrente

1. **Fragmentación (Map):** Series largas se dividen en ventanas pequeñas con contexto (p. ej., 10 puntos + estadísticas globales).
2. **Ejecución Asíncrona:** `asyncio` y `.abatch` de LangChain permiten lanzar múltiples solicitudes al LLM simultáneamente.
3. **Reensamblaje (Reduce):** Los resultados se recolectan, validan (JSON), y se mapean de vuelta a sus índices originales.

**Impacto en Rendimiento:** Reduce tiempos de horas a minutos manejando la latencia de red en paralelo.

```
                                 [ RAW DATAFRAME ]
                                         |
                                         v
                                [ ANOMALY SYSTEM ]
                         (Orchestrator / Dependency Injection)
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      STRATEGY A: Z-SCORE                                 STRATEGY B: LLM
      (Synchronous / Vectorized)                          (Async / Map-Reduce)
               |                                                   |
    1. FIT METADATA (Fast)                              1. FIT GLOBAL STATS
       (Mean, Std, Median)                                 (Calculate once for whole item)
               |                                                   |
    2. DETECT (Vectorized)                              2. CHUNK TIME SERIES
       |X - Mean| / Std                                    (Split to manage token limits)
       [Array Operation]                                   [Chunk 1] [Chunk 2] [Chunk 3]...
               |                                                   |
               |                                        3. ASYNC MAP (Parallel Execution)
               |                                           +-------+   +-------+
               |                                           |  LLM  |   |  LLM  |  (Max Conc: 5)
               |                                           +-------+   +-------+
               |                                                   |
               |                                        4. REDUCE (Stitch Results)
               |                                           Reassemble chunks into original
               |                                           array indices in-place.
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                                [ SCORE CONVERTER ]
                          (Standardize outputs to Binary)
                       If Z-Score > 3.0  OR  LLM says "ANOMALOUS"
                                         |
                                         v
                            [ FINAL REPORT & SUMMARY ]
                            - ZScore: 12 Anomalies found
                            - LLM:     8 Anomalies found
```

## 🛠 **Stack Tecnológico y Observabilidad**

* **LangChain:** Manejo del prompting y parsing estructurado en JSON.
* **MLflow:** Trazabilidad de extremo a extremo. Cada batch async, llamada LLM y uso de tokens queda registrado, permitiendo identificar dónde se disparan la latencia o los costos.
* **Pandas/NumPy:** Operaciones vectorizadas para cálculos estadísticos de alta velocidad (Z-Score).

## 📂 **Estructura del Código**

* `BaseAnomalyModel`: Interfaz para todos los detectores.
* `ZScoreModel`: Detector estadístico base, rápido.
* `NewLLMAnomalyDetector`: Detector concurrente usando OpenAI/LangChain.
* `AnomalySystem`: Orquestador que ejecuta todos los modelos inyectados y agrega resultados.

## 📊 UML

```
+-----------------------------------------------------------+
|                      AnomalySystem                        |
+-----------------------------------------------------------+
| - base_structure: Dict[str, ModelData]                    | 1
| - converter: ScoreConverter                               |<>-------+
+-----------------------------------------------------------+         |
| + fit_and_detect()                                        |         | (Composes)
| + print_summary()                                         |         |
+-----------------------------------------------------------+         |
          | 1                                                         |
          | (Aggregates)                                              |
          v * v
+------------------------+                               +------------------------+
|       ModelData        |                               |     ScoreConverter     |
|      (TypedDict)       |                               +------------------------+
+------------------------+                               | + percentile: float    |
| + name: str            |                               | + convert(scores)      |
| + items: List[ItemData]|                               +------------------------+
| + model_object: Base   |
+-----------+------------+
            |
            | (Contains Strategy)
            v
+-----------------------------------------------------------+
|           <<Abstract>> BaseAnomalyModel                   |
+-----------------------------------------------------------+
| + metadata: Dict                                          |
| + fit(X, item_id)                                         |
| + score(X, item_id)                                       |
+-----------------------------^-----------------------------+
                              |
              +---------------+---------------+
              | (Inherits)                    | (Inherits)
+---------------------------+   +---------------------------+
|        ZScoreModel        |   |   NewLLMAnomalyDetector   |
+---------------------------+   +---------------------------+
| + threshold: float        |   | + llm: BaseChatModel      |
|                           |   | + max_concurrent: int     |
| + _compute_anomaly_scores |   |                           |
|   (Math implementation)   |   | + LLMfit_and_detect()     |
+---------------------------+   |   (Async implementation)  |
                                | + _create_item_chain()    |
                                +---------------------------+
```

## 🚀 **Inicio Rápido**


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

### Ejecutar el Agente (v2)

Con el entorno activado y MLflow corriendo, ejecuta el script principal:

```bash
python v2.py --absolute_path_to_csv /ruta/absoluta/a/tu/archivo.csv
```

**Parámetros:**
- `--absolute_path_to_csv` (requerido): Ruta absoluta al archivo CSV con los datos históricos de precios.

**Ejemplo:**
```bash
python v2.py --absolute_path_to_csv /Users/vidal/Desktop/meli/data/precios_historicos.csv
```


# Next Steps

## Bootrap Evaluator

```
PHASE 1: DETECTION                               PHASE 2: PROXY METRIC BOOTSTRAP (Unlabeled)
==================                               ===========================================

[ Model A (Z-Score) ]   [ Model B (LLM) ]                 [ RAW TIME SERIES DATA (X) ]
        |                       |                                       |
        v                       v                                       |
  [ Binary Flags ]        [ Binary Flags ]                              |
  [ 0, 0, 1, 0... ]       [ 0, 1, 1, 0... ]                             |
        |                       |                                       |
        +-----------+-----------+---------------------------------------+
                    |
                    v
    +-----------------------------------------------------------------------+
    |                  BOOTSTRAP PROXY EVALUATOR                            |
    |   (Compares "Quality of Anomalies" not "Correctness")                 |
    +-----------------------------------------------------------------------+
                    |
                    | < LOOP n=1000 Iterations >
                    |
      +-------------+---------------------------------------------------+
      | 1. RESAMPLE (With Replacement)                                  |
      |    Generate random indices `idx` size of N                      |
      |                                                                 |
      | 2. APPLY MASKS (Model A vs Model B)                             |
      |    Get data values flagged by A:  X_A = Data[idx][Preds_A[idx]] |
      |    Get data values flagged by B:  X_B = Data[idx][Preds_B[idx]] |
      |                                                                 |
      | 3. COMPUTE PROXY METRICS (Internal Quality)                     |
      |    a) Sparsity Score:                                           |
      |       len(X_A) / len(sample)  (Too high = over-sensitive)       |
      |                                                                 |
      |    b) Signal-to-Noise (SNR) / Severity:                         |
      |       Mean distance of X_A from Rolling Median                  |
      |       (Are the flagged points actually extreme?)                |
      |                                                                 |
      |    c) Consensus/Stability:                                      |
      |       Jaccard(Preds_A, Preds_B) on this sample                  |
      +-------------+---------------------------------------------------+
                    |
                    v
          [ DISTRIBUTIONS OF QUALITY ]
      SNR(A): [3.1σ, 3.2σ, 2.9σ...]  vs  SNR(B): [4.5σ, 4.8σ, 4.4σ...]
                    |
                    v
      +-------------------------------------------+
      |          INTERPRETATION LOGIC             |
      +-------------------------------------------+
      | "Model B (LLM) flags fewer points         |
      |  (Sparsity: 1% vs 5%), but the points     |
      |  it flags are significantly more extreme  |
      |  (SNR: 4.5σ vs 3.1σ).                     |
      |  -> Conclusion: B is more precise/useful."|
      +-------------------------------------------+
```

```
+---------------------+
|    AnomalySystem    |
+---------------------+
| + fit_and_detect()  |
| + results: Dict     |
+----------+----------+
           |
           | (Feeds)
           v
+-------------------------------------------------------+
|                BootstrapProxyEvaluator                |
+-------------------------------------------------------+
| + n_iterations: int = 1000                            |
| + metric_type: Enum [SNR, SPARSITY, CONSENSUS]        |
|                                                       |
| + compare_models(raw_data, preds_a, preds_b)          |
|   -> ProxyComparisonResult                            |
|                                                       |
| # Internal Bootstrap Logic                            |
| - _resample_indices(n_samples)                        |
| - _calc_sparsity(preds_sample)                        |
| - _calc_snr(data_sample, preds_sample)                |
|   (Calculates |val - rolling_mean| for flags)         |
+---------------------------+---------------------------+
                            |
                            | (Returns)
                            v
            +-----------------------------------+
            |       ProxyComparisonResult       |
            |           (Dataclass)             |
            +-----------------------------------+
            | + metric_name: str ("SNR")        |
            | + dist_a: List[float]             |
            | + dist_b: List[float]             |
            | + median_a: float                 |
            | + median_b: float                 |
            | + overlap_score: float            |
            +-----------------------------------+
```