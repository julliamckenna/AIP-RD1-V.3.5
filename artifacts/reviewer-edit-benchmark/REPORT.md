# Extractor benchmark — 24 synthetic cases (0 failed)

Overall: recall 0.66 · precision 0.85 · ownership 0.85 · recall on clear markers 0.80 · on crowded markers 0.57 · centre error P50 1.31 px / P95 3.81 px · y error P50 0.0052 mmol/g

Definitions: a truth marker is *crowded* when another marker (any series) lies within 2.2 marker radii. Matching tolerance = 1 marker radius. Ownership = matched markers with the right series label. Centre error is measured on correctly owned matches only.


## by n_series

| n_series | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 6 | 0.63 | 0.82 | 0.86 | 0.83 | 0.54 | 1.28 | 3.74 | 0.0041 |
| 17 | 6 | 0.45 | 0.75 | 0.71 | 0.46 | 0.47 | 1.52 | 4.53 | 0.0056 |
| 4 | 6 | 0.82 | 0.96 | 1.00 | 0.97 | 0.64 | 1.23 | 3.15 | 0.0057 |
| 8 | 6 | 0.75 | 0.86 | 0.85 | 0.94 | 0.64 | 1.22 | 3.82 | 0.0056 |

## by palette

| palette | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distinct | 12 | 0.78 | 0.91 | 0.92 | 0.96 | 0.64 | 1.22 | 3.48 | 0.0056 |
| origin | 12 | 0.54 | 0.79 | 0.79 | 0.64 | 0.50 | 1.40 | 4.14 | 0.0048 |

## by width

| width | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 8 | 0.68 | 0.82 | 0.72 | 0.88 | 0.58 | 1.15 | 3.16 | 0.0055 |
| 1200 | 8 | 0.66 | 0.82 | 0.89 | 0.74 | 0.60 | 1.16 | 3.74 | 0.0050 |
| 1600 | 8 | 0.63 | 0.91 | 0.95 | 0.78 | 0.53 | 1.62 | 4.53 | 0.0052 |

## by jpeg_q

| jpeg_q | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 8 | 0.68 | 0.82 | 0.72 | 0.88 | 0.58 | 1.15 | 3.16 | 0.0055 |
| 78 | 8 | 0.66 | 0.82 | 0.89 | 0.74 | 0.60 | 1.16 | 3.74 | 0.0050 |
| 92 | 8 | 0.63 | 0.91 | 0.95 | 0.78 | 0.53 | 1.62 | 4.53 | 0.0052 |

## by marker_pt

| marker_pt | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 12 | 0.52 | 0.85 | 0.89 | 0.78 | 0.49 | 1.48 | 3.49 | 0.0062 |
| 7 | 12 | 0.80 | 0.84 | 0.82 | 0.82 | 0.65 | 1.14 | 4.14 | 0.0043 |

## by density

| density | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 12 | 0.52 | 0.85 | 0.89 | 0.78 | 0.49 | 1.48 | 3.49 | 0.0062 |
| normal | 12 | 0.80 | 0.84 | 0.82 | 0.82 | 0.65 | 1.14 | 4.14 | 0.0043 |

## by loops

| loops | cases | recall | precision | ownership | recall clear | recall crowded | centre P50 px | centre P95 px | y err P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| False | 12 | 0.80 | 0.84 | 0.82 | 0.82 | 0.65 | 1.14 | 4.14 | 0.0043 |
| True | 12 | 0.52 | 0.85 | 0.89 | 0.78 | 0.49 | 1.48 | 3.49 | 0.0062 |


## per case

| case | truth | pred | crowded | recall | precision | ownership | rec clear | rec crowded | P50 px | P95 px | s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c00_s4_distinct_w1600_q92_m7_normal | 112 | 110 | 0.04 | 0.98 | 1.00 | 0.99 | 1.00 | 0.50 | 2.35 | 4.27 | 12.0 |
| c01_s4_distinct_w1600_q92_m5_dense_loops | 432 | 258 | 0.90 | 0.58 | 0.97 | 1.00 | 0.95 | 0.54 | 1.24 | 3.18 | 12.3 |
| c02_s4_distinct_w1200_q78_m7_normal | 112 | 108 | 0.04 | 0.96 | 1.00 | 0.99 | 0.97 | 0.75 | 0.90 | 3.63 | 4.9 |
| c03_s4_distinct_w1200_q78_m5_dense_loops | 432 | 315 | 0.88 | 0.67 | 0.92 | 1.00 | 0.94 | 0.64 | 0.87 | 2.61 | 7.3 |
| c04_s4_distinct_w1000_q70x2_m7_normal | 112 | 111 | 0.04 | 0.99 | 1.00 | 1.00 | 1.00 | 0.75 | 1.08 | 2.93 | 2.8 |
| c05_s4_distinct_w1000_q70x2_m5_dense_loops | 432 | 363 | 0.74 | 0.73 | 0.87 | 1.00 | 0.94 | 0.66 | 0.95 | 2.28 | 5.5 |
| c06_s8_distinct_w1600_q92_m7_normal | 224 | 254 | 0.18 | 0.95 | 0.84 | 1.00 | 0.97 | 0.88 | 1.13 | 5.00 | 20.8 |
| c07_s8_distinct_w1600_q92_m5_dense_loops | 864 | 573 | 0.98 | 0.64 | 0.97 | 0.99 | 1.00 | 0.64 | 1.86 | 3.98 | 37.1 |
| c08_s8_distinct_w1200_q78_m7_normal | 224 | 275 | 0.08 | 0.99 | 0.80 | 1.00 | 1.00 | 0.82 | 0.89 | 3.64 | 13.1 |
| c09_s8_distinct_w1200_q78_m5_dense_loops | 864 | 535 | 0.96 | 0.54 | 0.87 | 0.99 | 0.91 | 0.52 | 1.43 | 3.71 | 15.0 |
| c10_s8_distinct_w1000_q70x2_m7_normal | 224 | 197 | 0.12 | 0.75 | 0.86 | 0.15 | 0.79 | 0.46 | 0.87 | 3.58 | 12.6 |
| c11_s8_distinct_w1000_q70x2_m5_dense_loops | 864 | 622 | 0.88 | 0.60 | 0.83 | 0.94 | 0.99 | 0.54 | 1.12 | 2.99 | 17.3 |
| c12_s11_origin_w1600_q92_m7_normal | 308 | 253 | 0.25 | 0.68 | 0.82 | 0.95 | 0.75 | 0.45 | 0.99 | 4.85 | 94.9 |
| c13_s11_origin_w1600_q92_m5_dense_loops | 1188 | 657 | 0.94 | 0.51 | 0.93 | 0.94 | 0.99 | 0.48 | 2.01 | 4.10 | 104.1 |
| c14_s11_origin_w1200_q78_m7_normal | 308 | 318 | 0.21 | 0.76 | 0.74 | 0.76 | 0.78 | 0.69 | 0.84 | 3.71 | 45.9 |
| c15_s11_origin_w1200_q78_m5_dense_loops | 1188 | 726 | 0.85 | 0.51 | 0.84 | 0.91 | 0.64 | 0.49 | 1.50 | 3.43 | 79.3 |
| c16_s11_origin_w1000_q70x2_m7_normal | 308 | 325 | 0.11 | 0.83 | 0.79 | 0.85 | 0.85 | 0.70 | 0.90 | 3.13 | 25.1 |
| c17_s11_origin_w1000_q70x2_m5_dense_loops | 1188 | 712 | 0.96 | 0.47 | 0.79 | 0.74 | 0.96 | 0.45 | 1.41 | 3.25 | 36.9 |
| c18_s17_origin_w1600_q92_m7_normal | 476 | 290 | 0.28 | 0.49 | 0.80 | 0.84 | 0.46 | 0.56 | 1.13 | 5.92 | 317.6 |
| c19_s17_origin_w1600_q92_m5_dense_loops | 1836 | 470 | 0.96 | 0.23 | 0.92 | 0.85 | 0.09 | 0.24 | 2.25 | 4.92 | 273.4 |
| c20_s17_origin_w1200_q78_m7_normal | 476 | 368 | 0.38 | 0.57 | 0.74 | 0.80 | 0.56 | 0.59 | 1.16 | 5.05 | 107.2 |
| c21_s17_origin_w1200_q78_m5_dense_loops | 1836 | 879 | 0.92 | 0.31 | 0.65 | 0.69 | 0.15 | 0.32 | 1.68 | 4.15 | 218.1 |
| c22_s17_origin_w1000_q70x2_m7_normal | 476 | 439 | 0.47 | 0.66 | 0.72 | 0.51 | 0.65 | 0.67 | 1.44 | 3.91 | 108.8 |
| c23_s17_origin_w1000_q70x2_m5_dense_loops | 1836 | 1148 | 0.95 | 0.44 | 0.70 | 0.58 | 0.82 | 0.42 | 1.44 | 3.24 | 84.6 |