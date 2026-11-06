
### mat_338: 338 books, vocab 5,588, nnz 238,645 (mean 706 distinct words/book), CSR 1.8 MB

PMI transform (Shifted PPMI, alpha=0.85):

| backend | time |
|---|---|
| numba_csr | 0.002 s |
| scipy | 0.01 s |
| pf | 0.053 s (est) |
| torch_sparse | 0.001 s |

**cos_tf**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 0.17 ms | 0.06 s (est) | 1.000 | 1.8 |  |
| scipy |  | 0.35 ms | 0.07 s (est) |  | 1.8 |  |
| pf | 0.02 s | 1.97 ms | 0.66 s (est) | 1.000 | 2.3 |  |
| torch_sparse | 0.001 s | 0.16 ms | 0.06 s | 1.000 |  | 10.1 |
| dense5k_fp32 | 1.28 s | 0.04 ms | 0.02 s | 1.000 | 7.2 | 23.1 |
| dense5k_fp16 | 0.003 s | 0.06 ms | 0 s | 0.991 | 3.6 | 16.3 |

**cos_ppmi**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 4.67 ms | 1.58 s (est) | 1.000 | 0.6 |  |
| scipy |  | 0.14 ms | 0.01 s (est) |  | 0.6 |  |
| pf | 0.01 s | 1.26 ms | 0.43 s (est) | 1.000 | 0.8 |  |
| torch_sparse | 0.001 s | 0.16 ms | 0.01 s | 1.000 |  | 17.0 |
| dense5k_fp32 | 0.001 s | 0.04 ms | 0 s | 1.000 | 7.2 | 23.1 |
| dense5k_fp16 | 0.001 s | 0.06 ms | 0 s | 0.999 | 3.6 | 15.7 |
| rp256 | 0.099 s |  | 0.002 s | 0.461 | 0.2 | 12.1 |
| rp1024 | 0.001 s |  | 0.001 s | 0.689 | 0.7 | 13.3 |
| svd256 | 0.088 s |  | 0.001 s | 0.871 | 0.2 | 28.6 |
| svd1024 | 0.058 s |  | 0.001 s | 0.999 | 0.2 | 33.3 |
| hybrid_rp256_C50 |  | 0.23 ms (cand 0.01 + rerank 0.22; explain top-10 0.4 ms) | 0.1 s (est) | 0.766 |  |  |
| hybrid_rp256_C200 |  | 0.83 ms (cand 0.01 + rerank 0.83; explain top-10 0.4 ms) | 0.32 s (est) | 0.971 |  |  |
| hybrid_svd256_C50 |  | 0.22 ms (cand 0.00 + rerank 0.22; explain top-10 0.4 ms) | 0.1 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 0.88 ms (cand 0.00 + rerank 0.88; explain top-10 0.4 ms) | 0.33 s (est) | 1.000 |  |  |

**hellinger**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 0.18 ms | 0.06 s (est) | 1.000 | 1.8 |  |
| scipy |  | 0.35 ms | 0.07 s (est) |  | 1.8 |  |
| pf | 0.02 s | 1.99 ms | 0.67 s (est) | 1.000 | 2.3 |  |
| torch_sparse | 0.002 s | 0.16 ms | 0.01 s | 1.000 |  | 19.5 |
| dense5k_fp32 | 0.003 s | 0.04 ms | 0 s | 1.000 | 7.2 | 24.3 |
| dense5k_fp16 | 0.002 s | 0.06 ms | 0 s | 0.994 | 3.6 | 17.5 |
| rp256 | 0.002 s |  | 0 s | 0.710 | 0.2 | 19.9 |
| rp1024 | 0.002 s |  | 0.001 s | 0.857 | 0.7 | 21.1 |
| svd256 | 0.055 s |  | 0.001 s | 0.901 | 0.2 | 32.3 |
| svd1024 | 0.062 s |  | 0 s | 0.996 | 0.2 | 37.0 |
| hybrid_rp256_C50 |  | 0.44 ms (cand 0.00 + rerank 0.43; explain top-10 0.7 ms) | 0.21 s (est) | 0.987 |  |  |
| hybrid_rp256_C200 |  | 1.38 ms (cand 0.01 + rerank 1.37; explain top-10 0.6 ms) | 0.53 s (est) | 1.000 |  |  |
| hybrid_svd256_C50 |  | 0.39 ms (cand 0.00 + rerank 0.39; explain top-10 0.6 ms) | 0.19 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 1.42 ms (cand 0.01 + rerank 1.41; explain top-10 0.6 ms) | 0.55 s (est) | 1.000 |  |  |

**jsd**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 1.44 ms | 0.49 s (est) | 1.000 | 1.8 |  |
| scipy |  | 5.55 ms |  |  |  |  |
| pf | 0.02 s | 3.57 ms | 1.21 s (est) | 1.000 | 2.3 |  |
| torch_sparse | 0.002 s | 0.42 ms | 0.17 s | 1.000 |  | 17.3 |
| dense5k_fp32 | 0.002 s | 0.38 ms | 0.16 s | 1.000 | 7.2 | 48.7 |
| dense5k_fp16 | 0.003 s | 0.44 ms | 0.16 s | 0.998 | 3.6 | 52.1 |
| rp256 | 0.002 s |  | 0.001 s | 0.703 | 0.2 | 10.9 |
| rp1024 | 0.002 s |  | 0.001 s | 0.845 | 0.7 | 11.4 |
| svd256 | 0.055 s |  | 0 s | 0.875 | 0.2 | 10.9 |
| svd1024 | 0.062 s |  | 0 s | 0.936 | 0.2 | 11.0 |
| hybrid_rp256_C50 |  | 0.82 ms (cand 0.00 + rerank 0.82; explain top-10 19.4 ms) | 0.34 s (est) | 0.987 |  |  |
| hybrid_rp256_C200 |  | 2.62 ms (cand 0.01 + rerank 2.61; explain top-10 2.5 ms) | 0.96 s (est) | 1.000 |  |  |
| hybrid_svd256_C50 |  | 0.85 ms (cand 0.01 + rerank 0.84; explain top-10 2.6 ms) | 0.35 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 2.64 ms (cand 0.01 + rerank 2.63; explain top-10 2.5 ms) | 0.96 s (est) | 1.000 |  |  |

gpu total_peak: 11.0 MB, process max_rss: 1615.6 MB

### synthetic_1000: 1,000 books, vocab 500,000, nnz 15,526,558 (mean 15,527 distinct words/book), CSR 118.5 MB

PMI transform (Shifted PPMI, alpha=0.85):

| backend | time |
|---|---|
| numba_csr | 0.037 s |
| scipy | 0.653 s |
| pf | 1.62 s (est) |
| torch_sparse | 0.004 s |

**cos_tf**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 2.96 ms | 2.96 s (est) | 1.000 | 118.5 |  |
| scipy |  | 31.45 ms | 12 s (est) |  | 118.5 |  |
| pf | 1.11 s | 148.52 ms | 149 s (est) | 1.000 | 133.8 |  |
| torch_sparse | 0.05 s | 0.35 ms | 0.13 s | 1.000 |  | 1107.7 |
| dense50k_fp32 | 1.34 s | 0.31 ms | 0.01 s | 0.788 | 190.7 | 398.3 |
| dense50k_fp16 | 0.074 s | 0.19 ms | 0 s | 0.784 | 95.4 | 311.3 |

**cos_ppmi**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 1.96 ms | 1.96 s (est) | 1.000 | 50.2 |  |
| scipy |  | 16.01 ms | 1.63 s (est) |  | 50.2 |  |
| pf | 0.43 s | 65.16 ms | 65.2 s (est) | 1.000 | 62.7 |  |
| torch_sparse | 0.014 s | 0.23 ms | 0.07 s | 1.000 |  | 1043.1 |
| dense50k_fp32 | 0.06 s | 0.32 ms | 0.01 s | 0.467 | 190.7 | 393.8 |
| dense50k_fp16 | 0.037 s | 0.19 ms | 0 s | 0.467 | 95.4 | 202.4 |
| rp256 | 0.082 s |  | 0.002 s | 0.026 | 0.5 | 307.8 |
| rp1024 | 0.019 s |  | 0.001 s | 0.067 | 2.0 | 311.2 |
| svd256 | 1.12 s |  | 0.001 s | 0.351 | 0.5 | 718.8 |
| svd1024 | 3.56 s |  | 0.001 s | 0.741 | 1.9 | 2393.2 |
| hybrid_rp256_C50 |  | 4.01 ms (cand 0.03 + rerank 3.98; explain top-10 1.4 ms) | 8.47 s (est) | 0.118 |  |  |
| hybrid_rp256_C200 |  | 12.99 ms (cand 0.01 + rerank 12.97; explain top-10 1.3 ms) | 17.8 s (est) | 0.356 |  |  |
| hybrid_svd256_C50 |  | 4.20 ms (cand 0.01 + rerank 4.20; explain top-10 1.3 ms) | 8.7 s (est) | 0.729 |  |  |
| hybrid_svd256_C200 |  | 16.53 ms (cand 0.01 + rerank 16.52; explain top-10 1.3 ms) | 21.4 s (est) | 0.890 |  |  |

**hellinger**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 4.24 ms | 4.24 s (est) | 1.000 | 118.5 |  |
| scipy |  | 31.33 ms | 12.1 s (est) |  | 118.5 |  |
| pf | 1.05 s | 147.85 ms | 148 s (est) | 1.000 | 133.8 |  |
| torch_sparse | 0.052 s | 0.34 ms | 0.11 s | 1.000 |  | 1121.6 |
| dense50k_fp32 | 0.109 s | 0.31 ms | 0.01 s | 0.731 | 190.7 | 411.3 |
| dense50k_fp16 | 0.085 s | 0.19 ms | 0 s | 0.726 | 95.4 | 316.1 |
| rp256 | 0.045 s |  | 0.001 s | 0.710 | 0.5 | 688.3 |
| rp1024 | 0.048 s |  | 0.001 s | 0.831 | 2.0 | 692.7 |
| svd256 | 2.5 s |  | 0.001 s | 0.889 | 0.5 | 862.7 |
| svd1024 | 7.26 s |  | 0.001 s | 0.894 | 1.9 | 2537.1 |
| hybrid_rp256_C50 |  | 8.61 ms (cand 0.01 + rerank 8.61; explain top-10 4.0 ms) | 18.6 s (est) | 0.995 |  |  |
| hybrid_rp256_C200 |  | 33.23 ms (cand 0.01 + rerank 33.23; explain top-10 3.9 ms) | 43.4 s (est) | 1.000 |  |  |
| hybrid_svd256_C50 |  | 8.47 ms (cand 0.01 + rerank 8.46; explain top-10 3.9 ms) | 18.3 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 35.19 ms (cand 0.01 + rerank 35.18; explain top-10 3.9 ms) | 45.5 s (est) | 1.000 |  |  |

**jsd**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 11.64 ms | 11.6 s (est) | 1.000 | 118.5 |  |
| scipy |  | 415.28 ms |  |  |  |  |
| pf | 1.02 s | 196.31 ms | 196 s (est) | 1.000 | 133.8 |  |
| torch_sparse | 0.051 s | 3.50 ms | 3.59 s | 1.000 |  | 454.7 |
| dense50k_fp32 | 0.078 s | 7.51 ms | 7.59 s | 0.757 | 190.7 | 1020.9 |
| dense50k_fp16 | 0.074 s | 7.96 ms | 8.04 s | 0.757 | 95.4 | 1116.7 |
| rp256 | 0.045 s |  | 0.001 s | 0.704 | 0.5 | 22.2 |
| rp1024 | 0.048 s |  | 0.001 s | 0.824 | 2.0 | 22.6 |
| svd256 | 2.5 s |  | 0.001 s | 0.876 | 0.5 | 22.2 |
| svd1024 | 7.26 s |  | 0 s | 0.877 | 1.9 | 22.6 |
| hybrid_rp256_C50 |  | 11.60 ms (cand 0.00 + rerank 11.60; explain top-10 17.9 ms) | 21.9 s (est) | 0.994 |  |  |
| hybrid_rp256_C200 |  | 45.08 ms (cand 0.01 + rerank 45.07; explain top-10 17.9 ms) | 55.2 s (est) | 0.999 |  |  |
| hybrid_svd256_C50 |  | 11.96 ms (cand 0.01 + rerank 11.95; explain top-10 17.5 ms) | 21.7 s (est) | 0.997 |  |  |
| hybrid_svd256_C200 |  | 47.57 ms (cand 0.01 + rerank 47.56; explain top-10 17.6 ms) | 57.7 s (est) | 1.000 |  |  |

gpu total_peak: 22.6 MB, process max_rss: 2670.7 MB

### synthetic_10000: 10,000 books, vocab 500,000, nnz 154,742,874 (mean 15,474 distinct words/book), CSR 1,180.7 MB

PMI transform (Shifted PPMI, alpha=0.85):

| backend | time |
|---|---|
| numba_csr | 0.322 s |
| scipy | 6.85 s |
| pf | 15.7 s (est) |
| torch_sparse | 0.035 s |

**cos_tf**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 24.96 ms | 250 s (est) | 1.000 | 1180.7 |  |
| scipy |  | 305.67 ms | 16.8 min (est) |  | 1180.7 |  |
| pf | 10.7 s | 2155.79 ms | 359 min (est) | 1.000 | 1347.2 |  |
| torch_sparse | 0.307 s | 2.00 ms | 7.61 s | 1.000 |  | 2253.0 |
| dense50k_fp32 | 0.432 s | 2.86 ms | 0.58 s | 0.811 | 1907.3 | 3819.0 |
| dense50k_fp16 | 0.387 s | 1.52 ms | 0.18 s | 0.791 | 953.7 | 2869.3 |

**cos_ppmi**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 14.45 ms | 144 s (est) | 1.000 | 463.8 |  |
| scipy |  | 152.42 ms | 64.7 s (est) |  | 463.8 |  |
| pf | 4.15 s | 632.74 ms | 105 min (est) | 1.000 | 584.4 |  |
| torch_sparse | 0.135 s | 1.05 ms | 3.7 s | 1.000 |  | 1512.7 |
| dense50k_fp32 | 0.111 s | 2.85 ms | 0.57 s | 0.484 | 1907.3 | 2858.3 |
| dense50k_fp16 | 0.095 s | 1.47 ms | 0.17 s | 0.485 | 953.7 | 1434.8 |
| rp256 | 0.208 s |  | 0.007 s | 0.193 | 4.9 | 2592.4 |
| rp1024 | 0.185 s |  | 0.005 s | 0.485 | 19.5 | 2626.8 |
| svd256 | 7.61 s |  | 0.004 s | 0.622 | 4.9 | 1406.7 |
| svd1024 | 23.2 s |  | 0.005 s | 0.638 | 19.5 | 3233.3 |
| hybrid_rp256_C50 |  | 3.25 ms (cand 0.01 + rerank 3.23; explain top-10 1.3 ms) | 197 s (est) | 0.395 |  |  |
| hybrid_rp256_C200 |  | 14.76 ms (cand 0.02 + rerank 14.74; explain top-10 1.5 ms) | 504 s (est) | 0.603 |  |  |
| hybrid_svd256_C50 |  | 3.41 ms (cand 0.01 + rerank 3.40; explain top-10 1.5 ms) | 206 s (est) | 0.998 |  |  |
| hybrid_svd256_C200 |  | 16.80 ms (cand 0.01 + rerank 16.79; explain top-10 1.5 ms) | 526 s (est) | 1.000 |  |  |

**hellinger**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 26.93 ms | 269 s (est) | 1.000 | 1180.7 |  |
| scipy |  | 293.58 ms | 17.1 min (est) |  | 1180.7 |  |
| pf | 10.4 s | 1493.77 ms | 249 min (est) | 1.000 | 1347.2 |  |
| torch_sparse | 0.219 s | 2.00 ms | 7.57 s | 1.000 |  | 2309.2 |
| dense50k_fp32 | 0.69 s | 2.85 ms | 0.58 s | 0.718 | 1907.3 | 3876.0 |
| dense50k_fp16 | 0.709 s | 1.48 ms | 0.17 s | 0.717 | 953.7 | 2918.1 |
| rp256 | 0.443 s |  | 0.004 s | 0.696 | 4.9 | 4285.8 |
| rp1024 | 0.457 s |  | 0.005 s | 0.847 | 19.5 | 4320.0 |
| svd256 | 19.5 s |  | 0.004 s | 0.702 | 4.9 | 2915.9 |
| svd1024 | 60.6 s |  | 0.005 s | 0.857 | 19.5 | 4741.3 |
| hybrid_rp256_C50 |  | 8.79 ms (cand 0.01 + rerank 8.78; explain top-10 4.4 ms) | 555 s (est) | 0.991 |  |  |
| hybrid_rp256_C200 |  | 38.37 ms (cand 0.01 + rerank 38.36; explain top-10 4.3 ms) | 18.7 min (est) | 1.000 |  |  |
| hybrid_svd256_C50 |  | 8.25 ms (cand 0.01 + rerank 8.23; explain top-10 4.3 ms) | 487 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 32.90 ms (cand 0.01 + rerank 32.89; explain top-10 4.3 ms) | 17.1 min (est) | 1.000 |  |  |

**jsd**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 87.99 ms | 14.7 min (est) | 1.000 | 1180.7 |  |
| scipy |  | 5242.06 ms |  |  |  |  |
| pf | 10.6 s | 2133.16 ms | 356 min (est) | 1.000 | 1347.2 |  |
| torch_sparse | 0.312 s | 33.38 ms | 335 s (est) | 1.000 |  | 4392.7 |
| dense50k_fp32 | 0.485 s | 74.36 ms | 12.4 min (est) | 0.745 | 1907.3 | 8663.0 |
| dense50k_fp16 | 0.456 s | 78.81 ms | 13.1 min (est) | 0.744 | 953.7 | 9267.9 |
| rp256 | 0.443 s |  | 0.004 s | 0.699 | 4.9 | 265.3 |
| rp1024 | 0.457 s |  | 0.005 s | 0.840 | 19.5 | 271.3 |
| svd256 | 19.5 s |  | 0.004 s | 0.730 | 4.9 | 265.3 |
| svd1024 | 60.6 s |  | 0.005 s | 0.883 | 19.5 | 271.3 |
| hybrid_rp256_C50 |  | 12.44 ms (cand 0.01 + rerank 12.44; explain top-10 19.5 ms) | 566 s (est) | 0.991 |  |  |
| hybrid_rp256_C200 |  | 51.82 ms (cand 0.02 + rerank 51.80; explain top-10 19.3 ms) | 20.8 min (est) | 1.000 |  |  |
| hybrid_svd256_C50 |  | 11.28 ms (cand 0.02 + rerank 11.26; explain top-10 19.0 ms) | 517 s (est) | 1.000 |  |  |
| hybrid_svd256_C200 |  | 46.33 ms (cand 0.02 + rerank 46.32; explain top-10 19.2 ms) | 19.2 min (est) | 1.000 |  |  |

gpu total_peak: 271.3 MB, process max_rss: 9537.9 MB

### shard_8969: 8,969 books, vocab 11,812,683, nnz 77,586,651 (mean 8,650 distinct words/book), CSR 592.0 MB

PMI transform (Shifted PPMI, alpha=0.85):

| backend | time |
|---|---|
| numba_csr | 0.369 s |
| scipy | 3.81 s |
| pf | 14.6 s (est) |
| torch_sparse | 0.019 s |

**cos_tf**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 37.77 ms | 339 s (est) | 1.000 | 592.0 |  |
| scipy |  | 248.13 ms | 210 s (est) |  | 592.0 |  |
| pf | 5.81 s | 912.85 ms | 136 min (est) | 1.000 | 813.7 |  |
| torch_sparse | 0.166 s | 2.01 ms | 4.11 s | 1.000 |  | 4458.0 |
| dense50k_fp32 | 0.284 s | 2.56 ms | 0.47 s | 0.977 | 1710.7 | 2644.5 |
| dense50k_fp16 | 0.183 s | 1.30 ms | 0.14 s | 0.968 | 855.4 | 1628.8 |

**cos_ppmi**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 35.40 ms | 317 s (est) | 1.000 | 370.0 |  |
| scipy |  | 222.90 ms | 27.2 s (est) |  | 370.0 |  |
| pf | 3.57 s | 668.13 ms | 99.9 min (est) | 1.000 | 559.3 |  |
| torch_sparse | 0.115 s | 1.76 ms | 3.18 s | 1.000 |  | 4233.8 |
| dense50k_fp32 | 0.121 s | 2.64 ms | 0.47 s | 0.632 | 1710.7 | 2644.5 |
| dense50k_fp16 | 0.085 s | 1.30 ms | 0.14 s | 0.631 | 855.4 | 1328.3 |
| rp256 | 0.185 s |  | 0.01 s | 0.099 | 4.4 | 2592.6 |
| rp1024 | 0.186 s |  | 0.006 s | 0.234 | 17.5 | 2623.2 |
| svd256 | 3.82 s |  | 0.003 s | 0.502 | 4.4 | 1035.0 |
| svd1024 | 12 s |  | 0.006 s | 0.538 | 17.5 | 2853.8 |
| hybrid_rp256_C50 |  | 4.61 ms (cand 0.02 + rerank 4.59; explain top-10 1.1 ms) | 175 s (est) | 0.161 |  |  |
| hybrid_rp256_C200 |  | 23.26 ms (cand 0.01 + rerank 23.24; explain top-10 1.5 ms) | 494 s (est) | 0.251 |  |  |
| hybrid_svd256_C50 |  | 7.55 ms (cand 0.01 + rerank 7.54; explain top-10 2.0 ms) | 209 s (est) | 0.817 |  |  |
| hybrid_svd256_C200 |  | 20.52 ms (cand 0.01 + rerank 20.50; explain top-10 1.6 ms) | 466 s (est) | 0.949 |  |  |

**hellinger**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 36.72 ms | 329 s (est) | 1.000 | 592.0 |  |
| scipy |  | 236.60 ms | 210 s (est) |  | 592.0 |  |
| pf | 5.69 s | 928.47 ms | 139 min (est) | 1.000 | 813.7 |  |
| torch_sparse | 0.117 s | 2.03 ms | 4.1 s | 1.000 |  | 4509.0 |
| dense50k_fp32 | 0.341 s | 2.56 ms | 0.47 s | 0.822 | 1710.7 | 2688.3 |
| dense50k_fp16 | 0.299 s | 1.30 ms | 0.14 s | 0.819 | 855.4 | 1672.6 |
| rp256 | 0.237 s |  | 0.004 s | 0.549 | 4.4 | 3857.4 |
| rp1024 | 0.24 s |  | 0.006 s | 0.748 | 17.5 | 3888.0 |
| svd256 | 7.14 s |  | 0.004 s | 0.544 | 4.4 | 1533.3 |
| svd1024 | 22.3 s |  | 0.006 s | 0.643 | 17.5 | 3352.8 |
| hybrid_rp256_C50 |  | 10.44 ms (cand 0.01 + rerank 10.43; explain top-10 4.9 ms) | 316 s (est) | 0.843 |  |  |
| hybrid_rp256_C200 |  | 31.64 ms (cand 0.01 + rerank 31.63; explain top-10 3.0 ms) | 11.4 min (est) | 0.943 |  |  |
| hybrid_svd256_C50 |  | 6.19 ms (cand 0.01 + rerank 6.17; explain top-10 2.9 ms) | 269 s (est) | 0.867 |  |  |
| hybrid_svd256_C200 |  | 24.57 ms (cand 0.01 + rerank 24.55; explain top-10 2.9 ms) | 10.5 min (est) | 0.978 |  |  |

**jsd**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 51.68 ms | 464 s (est) | 1.000 | 592.0 |  |
| scipy |  | 1788.56 ms |  |  |  |  |
| pf | 5.41 s | 692.13 ms | 103 min (est) | 1.000 | 813.7 |  |
| torch_sparse | 0.2 s | 17.79 ms | 161 s (est) | 1.000 |  | 2291.9 |
| dense50k_fp32 | 0.189 s | 66.25 ms | 595 s (est) | 0.822 | 1710.7 | 8454.3 |
| dense50k_fp16 | 0.264 s | 70.23 ms | 10.5 min (est) | 0.822 | 855.4 | 9158.8 |
| rp256 | 0.237 s |  | 0.004 s | 0.550 | 4.4 | 238.5 |
| rp1024 | 0.24 s |  | 0.006 s | 0.735 | 17.5 | 245.3 |
| svd256 | 7.14 s |  | 0.003 s | 0.565 | 4.4 | 238.5 |
| svd1024 | 22.3 s |  | 0.005 s | 0.664 | 17.5 | 245.3 |
| hybrid_rp256_C50 |  | 7.89 ms (cand 0.01 + rerank 7.89; explain top-10 13.7 ms) | 296 s (est) | 0.842 |  |  |
| hybrid_rp256_C200 |  | 27.63 ms (cand 0.02 + rerank 27.61; explain top-10 13.7 ms) | 10.7 min (est) | 0.938 |  |  |
| hybrid_svd256_C50 |  | 6.55 ms (cand 0.02 + rerank 6.53; explain top-10 13.1 ms) | 267 s (est) | 0.890 |  |  |
| hybrid_svd256_C200 |  | 23.20 ms (cand 0.02 + rerank 23.18; explain top-10 13.4 ms) | 10.2 min (est) | 0.981 |  |  |

gpu total_peak: 245.3 MB, process max_rss: 5647.1 MB

### synthetic_100000: 100,000 books, vocab 500,000, nnz 1,548,295,806 (mean 15,483 distinct words/book), CSR 11,813.3 MB

PMI transform (Shifted PPMI, alpha=0.85):

| backend | time |
|---|---|
| numba_csr | 3.57 s |
| pf | 152 s (est) |
| torch_sparse | 0.352 s |

**cos_tf**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 224.76 ms | 375 min (est) | 1.000 | 11813.3 |  |
| scipy |  | 3358.75 ms |  |  | 11813.3 |  |
| pf | 110 s (est) | 15548.47 ms | 432 h (est) | 1.000 | 13462.7 |  |
| torch_sparse | 2.95 s | 19.77 ms | 12.1 min (est) | 1.000 |  | 13714.5 |
| dense50k_fp32 | 6.36 s | 28.23 ms | 59.2 s | 0.791 | 19073.5 | 22164.0 |
| dense50k_fp16 | 4.45 s | 15.89 ms | 8.99 s | 0.773 | 9536.7 | 12596.7 |

**cos_ppmi**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 125.99 ms | 210 min (est) | 1.000 | 4247.5 |  |
| scipy |  | 1457.32 ms |  |  | 4247.5 |  |
| pf | 39.5 s (est) | 3936.49 ms | 109 h (est) | 1.000 | 5343.0 |  |
| torch_sparse | 1.05 s | 8.92 ms | 311 s (est) | 1.000 |  | 5841.5 |
| dense50k_fp32 | 0.77 s | 28.22 ms | 58.2 s | 0.691 | 19073.5 | 21464.8 |
| dense50k_fp16 | 0.761 s | 15.90 ms | 8.65 s | 0.691 | 9536.7 | 11927.6 |
| rp256 | 1.83 s |  | 0.245 s | 0.598 | 48.8 | 4324.8 |
| rp1024 | 1.51 s |  | 0.361 s | 0.797 | 195.3 | 4666.6 |
| svd256 | 15 s |  | 0.239 s | 0.664 | 48.8 | 2211.2 |
| svd1024 | 44.1 s |  | 0.358 s | 0.767 | 195.3 | 4140.5 |
| hybrid_rp256_C50 |  | 2.17 ms (cand 0.01 + rerank 2.16; explain top-10 1.1 ms) | 36.4 min (est) | 0.929 |  |  |
| hybrid_rp256_C200 |  | 8.24 ms (cand 0.01 + rerank 8.23; explain top-10 1.1 ms) | 134 min (est) | 0.980 |  |  |
| hybrid_svd256_C50 |  | 2.16 ms (cand 0.01 + rerank 2.15; explain top-10 1.1 ms) | 35 min (est) | 0.982 |  |  |
| hybrid_svd256_C200 |  | 8.14 ms (cand 0.01 + rerank 8.13; explain top-10 1.1 ms) | 127 min (est) | 0.999 |  |  |

**hellinger**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 235.33 ms | 392 min (est) | 1.000 | 11813.3 |  |
| scipy |  | 3101.02 ms |  |  | 11813.3 |  |
| pf | 106 s (est) | 14432.26 ms | 401 h (est) | 1.000 | 13462.7 |  |
| torch_sparse | 2.9 s | 19.82 ms | 12.1 min (est) | 1.000 |  | 14209.6 |
| dense50k_fp32 | 13.7 s | 28.25 ms | 59.2 s | 0.738 | 19073.5 | 22660.4 |
| dense50k_fp16 | 9.78 s | 15.89 ms | 8.99 s | 0.740 | 9536.7 | 13085.0 |
| rp256 | 4.6 s |  | 0.243 s | 0.690 | 48.8 | 4813.1 |
| rp1024 | 3.99 s |  | 0.354 s | 0.854 | 195.3 | 5154.9 |
| svd256 | 49.9 s |  | 0.244 s | 0.467 | 48.8 | 5778.6 |
| svd1024 | 137 s |  | 0.362 s | 0.637 | 195.3 | 7707.9 |
| hybrid_rp256_C50 |  | 15.29 ms (cand 0.01 + rerank 15.28; explain top-10 4.6 ms) | 126 min (est) | 0.962 |  |  |
| hybrid_rp256_C200 |  | 49.24 ms (cand 0.02 + rerank 49.22; explain top-10 4.3 ms) | 341 min (est) | 0.987 |  |  |
| hybrid_svd256_C50 |  | 8.56 ms (cand 0.24 + rerank 8.32; explain top-10 4.1 ms) | 101 min (est) | 0.812 |  |  |
| hybrid_svd256_C200 |  | 36.47 ms (cand 0.02 + rerank 36.46; explain top-10 4.3 ms) | 357 min (est) | 0.933 |  |  |

**jsd**

| backend | build | one-vs-many / query | all-vs-all top-10 | recall@10 | repr MB | GPU peak MB |
|---|---|---|---|---|---|---|
| numba_csr |  | 827.47 ms | 23 h (est) | 1.000 | 11813.3 |  |
| pf | 106 s (est) | 19749.90 ms | 549 h (est) | 1.000 | 13462.7 |  |
| torch_sparse | 2.08 s | 331.96 ms | 554 min (est) | 1.000 |  | 17604.1 |
| dense50k_fp32 | 3.63 s | 742.85 ms | 20.6 h (est) | 0.765 | 19073.5 | 28312.2 |
| dense50k_fp16 | 4.13 s | 786.55 ms | 21.9 h (est) | 0.764 | 9536.7 | 20302.5 |
| rp256 | 4.6 s |  | 0.247 s | 0.678 | 48.8 | 2557.9 |
| rp1024 | 3.99 s |  | 0.357 s | 0.848 | 195.3 | 2563.9 |
| svd256 | 49.9 s |  | 0.239 s | 0.491 | 48.8 | 2557.9 |
| svd1024 | 137 s |  | 0.357 s | 0.655 | 195.3 | 2563.9 |
| hybrid_rp256_C50 |  | 13.31 ms (cand 0.01 + rerank 13.31; explain top-10 18.9 ms) | 123 min (est) | 0.959 |  |  |
| hybrid_rp256_C200 |  | 57.00 ms (cand 0.01 + rerank 56.98; explain top-10 18.0 ms) | 348 min (est) | 0.983 |  |  |
| hybrid_svd256_C50 |  | 11.83 ms (cand 0.24 + rerank 11.60; explain top-10 18.0 ms) | 105 min (est) | 0.826 |  |  |
| hybrid_svd256_C200 |  | 46.56 ms (cand 0.01 + rerank 46.55; explain top-10 18.5 ms) | 349 min (est) | 0.936 |  |  |

gpu total_peak: 2563.9 MB, process max_rss: 40925.0 MB
