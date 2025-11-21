======================================================================
OFFLINE ANOMALY DETECTION - PER-ITEM RETROSPECTIVE ANALYSIS
======================================================================

Loading historical price data...

Dataset: 224 price points across 3 items
Overall statistics: mean=487.48, std=371.76
Price range: [47.57, 960.63]

======================================================================
INITIALIZING COMPONENTS (Dependency Injection)
======================================================================
INITIALIZATION COMPLETE
======================================================================

Fitting models per item and detecting anomalies...
(Retrospective analysis - each item processed independently)

  Processing 23 items concurrently...
  Max concurrent requests: 5
  Batch 1/5: Processing 5 items...
    Completed batch 1 in 8.79s
  Batch 2/5: Processing 5 items...
    Completed batch 2 in 8.79s
  Batch 3/5: Processing 5 items...
    Completed batch 3 in 7.76s
  Batch 4/5: Processing 5 items...
    Completed batch 4 in 7.55s
  Batch 5/5: Processing 3 items...
    Completed batch 5 in 6.66s


======================================================================
ANOMALY DETECTION SYSTEM SUMMARY
======================================================================

Total observations: 224
Models evaluated: 2
Items processed: 3

----------------------------------------------------------------------
Model                     Anomalies       Rate       Score Range         
----------------------------------------------------------------------
ZScore                    23              10.27     % [0.01, 2.89]
LLM                       77              34.38     % [0.80, 0.99]
----------------------------------------------------------------------

======================================================================
SAMPLE ANOMALIES PER MODEL (First 3 per item)
======================================================================

ZScore:
  Item MLB4281133192: 10 anomalies detected
    [2024-01-24] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-25] Price=$55.94, Score=2.11, Reason: N/A
    [2024-01-23] Price=$53.02, Score=2.38, Reason: N/A
    [2024-01-27] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-30] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-22] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-28] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-29] Price=$56.09, Score=2.10, Reason: N/A
    [2024-01-23] Price=$47.57, Score=2.89, Reason: N/A
    [2024-01-23] Price=$56.09, Score=2.10, Reason: N/A
  Item MLB4184697886: 6 anomalies detected
    [2023-12-23] Price=$960.63, Score=1.62, Reason: N/A
    [2023-12-24] Price=$951.40, Score=1.52, Reason: N/A
    [2023-12-23] Price=$951.40, Score=1.52, Reason: N/A
    [2023-12-27] Price=$950.69, Score=1.51, Reason: N/A
    [2023-12-27] Price=$957.08, Score=1.58, Reason: N/A
    [2023-12-27] Price=$954.95, Score=1.56, Reason: N/A
  Item MLB4097203966: 7 anomalies detected
    [2024-01-26] Price=$741.24, Score=1.79, Reason: N/A
    [2024-01-28] Price=$733.43, Score=1.96, Reason: N/A
    [2024-02-05] Price=$728.46, Score=2.07, Reason: N/A
    [2024-01-02] Price=$921.58, Score=2.24, Reason: N/A
    [2024-02-04] Price=$729.88, Score=2.04, Reason: N/A
    [2024-02-18] Price=$729.88, Score=2.04, Reason: N/A
    [2024-01-29] Price=$736.27, Score=1.90, Reason: N/A

LLM:
  Item MLB4281133192: 40 anomalies detected
    [2023-12-18] Price=$91.59, Score=0.95, Reason: 1.22 std above mean significant spike
    [2023-12-15] Price=$88.75, Score=0.92, Reason: 0.95 std above mean notable increase
    [2024-01-18] Price=$72.42, Score=0.99, Reason: 2.3 std below mean extreme drop
    [2024-01-24] Price=$56.09, Score=0.98, Reason: 2.09 std below mean significant drop
    [2023-12-14] Price=$95.85, Score=0.97, Reason: 1.62 std above mean significant spike
    [2024-01-25] Price=$55.94, Score=0.99, Reason: 2.11 std below mean extreme drop
    [2023-12-14] Price=$86.62, Score=0.95, Reason: 0.69 std above mean unusual high
    [2023-12-16] Price=$95.85, Score=0.97, Reason: 1.62 std above mean significant spike
    [2023-12-19] Price=$92.30, Score=0.97, Reason: 1.28 std above mean significant spike
    [2024-01-23] Price=$53.02, Score=0.99, Reason: 2.38 std below mean extreme drop
    [2024-01-27] Price=$56.09, Score=0.99, Reason: 2.09 std below mean extreme drop
    [2024-01-30] Price=$56.09, Score=0.98, Reason: 2.02 std below mean significant drop
    [2023-12-13] Price=$95.85, Score=0.95, Reason: 3.21 std above mean massive spike
    [2024-01-10] Price=$85.20, Score=0.95, Reason: 0.06 std above Q75 threshold
    [2024-01-20] Price=$66.34, Score=0.99, Reason: 2.3 std below mean extreme drop
    [2024-01-22] Price=$60.35, Score=0.99, Reason: 2.7 std below mean extreme drop
    [2024-01-22] Price=$56.09, Score=0.99, Reason: 2.1 std below mean extreme drop
    [2023-12-19] Price=$90.17, Score=0.98, Reason: 1.1 std above mean significant spike
    [2024-01-22] Price=$57.51, Score=0.95, Reason: 2 std below mean significant drop
    [2023-12-19] Price=$88.75, Score=0.97, Reason: 1.55 std above mean extreme spike
    [2023-12-13] Price=$95.14, Score=0.93, Reason: 1.5 std above mean unusual high
    [2024-01-06] Price=$74.55, Score=0.95, Reason: 1.5 std below mean unusual low
    [2023-12-19] Price=$91.59, Score=0.98, Reason: 1.2 std above mean significant spike
    [2024-01-28] Price=$56.09, Score=0.99, Reason: 2.1 std below mean extreme drop
    [2024-01-19] Price=$71.71, Score=0.96, Reason: 1.0 std below mean unusual low
    [2023-12-19] Price=$89.46, Score=0.97, Reason: 1.0 std above mean significant spike
    [2023-12-13] Price=$90.88, Score=0.95, Reason: 1.15 std above mean significant spike
    [2024-01-20] Price=$65.67, Score=0.99, Reason: 1.2 std below mean extreme drop
    [2024-01-17] Price=$72.42, Score=0.95, Reason: 2.8 std below mean significant drop
    [2024-01-19] Price=$67.71, Score=0.99, Reason: 10.1 std below mean extreme drop
    [2023-12-18] Price=$92.30, Score=0.95, Reason: 1.28 std above mean significant spike
    [2023-12-14] Price=$90.88, Score=0.99, Reason: 3.0 std below mean extreme drop
    [2024-01-29] Price=$56.09, Score=0.98, Reason: 2.9 std below mean extreme drop
    [2024-01-23] Price=$47.57, Score=0.95, Reason: 2.5 std above mean significant spike
    [2024-01-17] Price=$75.26, Score=0.95, Reason: 2.9 std below mean significant drop
    [2024-01-23] Price=$56.09, Score=0.99, Reason: 2.1 std below mean extreme drop
    [2024-01-22] Price=$56.80, Score=0.95, Reason: 2.02 std below mean significant drop
    [2024-01-14] Price=$75.26, Score=0.92, Reason: 3.03 std below mean unusual low
    [2024-01-20] Price=$67.71, Score=0.94, Reason: 1.25 std below mean significant drop
    [2023-12-13] Price=$90.17, Score=0.96, Reason: 1.09 std above mean significant spike
  Item MLB4184697886: 18 anomalies detected
    [2024-02-04] Price=$729.17, Score=0.95, Reason: 1.1 std below mean unusual low
    [2024-03-04] Price=$706.45, Score=0.96, Reason: 1.2 std below mean significant drop
    [2024-03-02] Price=$710.00, Score=0.95, Reason: 2.2 std below mean significant drop
    [2023-12-23] Price=$960.63, Score=0.98, Reason: 1.6 std above mean extreme spike
    [2023-12-25] Price=$936.49, Score=0.96, Reason: 1.4 std above mean significant spike
    [2023-12-28] Price=$943.59, Score=0.95, Reason: 1.44 std above mean significant spike
    [2024-03-05] Price=$706.45, Score=0.98, Reason: 1.56 std below mean extreme drop
    [2023-12-24] Price=$951.40, Score=0.95, Reason: 1.53 std above mean significant spike
    [2023-12-23] Price=$951.40, Score=0.97, Reason: 1.43 std above mean significant spike
    [2023-12-27] Price=$950.69, Score=0.95, Reason: 1.6 std above mean significant spike
    [2023-12-27] Price=$957.08, Score=0.96, Reason: 1.6 std above mean significant spike
    [2024-03-05] Price=$702.90, Score=0.99, Reason: 1.3 std below mean extreme drop
    [2024-03-06] Price=$695.80, Score=0.99, Reason: 1.3 std below mean extreme drop
    [2023-12-27] Price=$954.95, Score=0.95, Reason: 1.57 std above mean significant spike
    [2024-01-30] Price=$726.33, Score=0.85, Reason: Below Q25 threshold unusual low
    [2024-03-07] Price=$702.19, Score=0.99, Reason: 1.20 std below mean extreme drop
    [2024-03-09] Price=$702.19, Score=0.95, Reason: Below Q25 threshold unusual low
    [2024-01-06] Price=$943.59, Score=0.92, Reason: 1.43 std above mean significant spike
  Item MLB4097203966: 19 anomalies detected
    [2024-01-26] Price=$741.24, Score=0.95, Reason: 3 std below mean significant drop
    [2023-12-21] Price=$865.49, Score=0.92, Reason: Higher than 1.5 IQR above Q75
    [2024-01-17] Price=$774.61, Score=0.98, Reason: 2.1 std below mean significant drop
    [2024-01-02] Price=$890.34, Score=0.95, Reason: 1.5 std above mean unusual spike
    [2024-01-21] Price=$753.31, Score=0.99, Reason: 3.0 std below mean extreme drop
    [2024-01-28] Price=$733.43, Score=0.97, Reason: 2.0 std below mean significant drop
    [2024-01-19] Price=$741.95, Score=0.95, Reason: 1.77 std below mean significant drop
    [2024-01-20] Price=$766.09, Score=0.90, Reason: 1.23 std below mean unusual low
    [2024-01-20] Price=$754.02, Score=0.97, Reason: 1.93 std below mean extreme drop
    [2024-02-05] Price=$728.46, Score=0.99, Reason: 2.07 std below mean extreme drop
    [2024-01-02] Price=$921.58, Score=0.95, Reason: 2.25 std above mean significant spike
    [2024-01-28] Price=$772.48, Score=0.96, Reason: 1.09 std below mean unusual drop
    [2024-02-04] Price=$729.88, Score=0.95, Reason: 2.03 std below mean significant drop
    [2023-12-18] Price=$883.24, Score=0.96, Reason: 1.38 std above mean significant spike
    [2024-02-18] Price=$729.88, Score=0.95, Reason: 2.03 std below mean significant drop
    [2024-02-12] Price=$791.65, Score=0.92, Reason: 1.5 std below mean unusual low
    [2024-01-19] Price=$779.58, Score=0.94, Reason: 1.9 std below mean significant drop
    [2024-01-29] Price=$736.27, Score=0.95, Reason: 1.9 std below mean significant drop
    [2024-01-22] Price=$749.05, Score=0.90, Reason: 1.6 std below mean unusual low

======================================================================
DETAILED ANOMALY DETECTION RESULTS
======================================================================

======================================================================
FINAL
======================================================================
(meli) vidal@MQXVL6D5VW ts_model_vs % 